package main

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type testPoolState struct {
	Keys      []testKeyState
	Anonymous []testKeyState
	Healthy   []bool
}

type testKeyState struct {
	Failures uint32
	Cooldown int64
	Proxy    int64
}

func snapshotTestPools(g *Gateway) testPoolState {
	state := testPoolState{}
	for _, pool := range []*nodePool{g.zenNodes, g.goNodes} {
		for _, node := range pool.nodes {
			state.Keys = append(state.Keys, testKeyState{
				Failures: node.failures.Load(), Cooldown: node.cooldownUntil.Load(),
				Proxy: node.proxyIndex.Load(),
			})
		}
	}
	for _, node := range g.anonymous.nodes {
		state.Anonymous = append(state.Anonymous, testKeyState{
			Failures: node.failures.Load(), Cooldown: node.cooldownUntil.Load(),
			Proxy: int64(node.proxy.index),
		})
	}
	for _, proxy := range g.transports.items {
		state.Healthy = append(state.Healthy, proxy.healthy.Load())
	}
	return state
}

func runDebugTest(t *testing.T, manager *RuntimeManager, key DebugKeySelection, model string) DebugInferenceResult {
	t.Helper()
	admin := NewAdminServer(manager, manager.monitor, manager.hub, manager.logger)
	payload, err := json.Marshal(DebugInferenceRequest{
		Protocol: ProtocolChat, Key: key,
		Request: map[string]any{
			"model": model, "stream": true,
			"messages": []any{map[string]any{"role": "user", "content": "hello"}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/api/debug/inference", strings.NewReader(string(payload)))
	recorder := httptest.NewRecorder()
	admin.handleDebugInference(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("management status = %d: %s", recorder.Code, recorder.Body.String())
	}
	var result DebugInferenceResult
	if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	return result
}

func TestAutomaticDiagnosticsPreserveProductionPools(t *testing.T) {
	for _, lane := range []string{"key", "anonymous"} {
		for _, outcome := range []string{"success", "unauthorized", "rate_limited", "transport_error"} {
			t.Run(lane+"/"+outcome, func(t *testing.T) {
				cfg := handlerTestConfig()
				cfg.Proxies = []string{"http://first.invalid:8080", "http://second.invalid:8080"}
				model := "example-model"
				if lane == "anonymous" {
					cfg.Anonymous = true
					cfg.ZenKeys = nil
					model = "example-free"
				}
				var probes atomic.Int32
				var calls atomic.Int32
				manager, gateway := newHandlerTestRuntime(t, cfg, func(r *http.Request) (*http.Response, error) {
					if r.URL.Host == "cloudflare.com" {
						probes.Add(1)
						return stubResponse(200, "ok"), nil
					}
					calls.Add(1)
					var request map[string]any
					if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
						t.Fatal(err)
					}
					if boolAt(request, "stream") {
						t.Fatal("Playground must force a non-streaming request")
					}
					switch outcome {
					case "unauthorized":
						return stubResponse(401, `{"error":{"message":"invalid key"}}`), nil
					case "rate_limited":
						return stubResponse(429, `{"error":{"message":"rate limited"}}`), nil
					case "transport_error":
						return nil, context.DeadlineExceeded
					default:
						return stubResponse(200, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`), nil
					}
				})
				// Successful diagnostics must not clear prior production failures.
				for _, node := range gateway.zenNodes.nodes {
					node.failures.Store(2)
					node.cooldownUntil.Store(time.Now().Add(time.Minute).UnixNano())
				}
				for _, node := range gateway.anonymous.nodes {
					node.failures.Store(2)
				}
				before := snapshotTestPools(gateway)
				result := runDebugTest(t, manager, DebugKeySelection{Mode: "auto"}, model)
				if after := snapshotTestPools(gateway); !reflect.DeepEqual(after, before) {
					t.Fatalf("diagnostic mutated pools:\nbefore: %+v\nafter: %+v", before, after)
				}
				if probes.Load() != 0 {
					t.Fatal("diagnostic triggered a proxy health probe")
				}
				if calls.Load() == 0 || result.Route.Attempts != int(calls.Load()) {
					t.Fatalf("attempt trace = %+v, upstream calls = %d", result.Route, calls.Load())
				}
				if result.OK != (outcome == "success") {
					t.Fatalf("unexpected result: %+v", result)
				}
			})
		}
	}
}

func TestProductionFailuresStillUpdateKeyCooldown(t *testing.T) {
	manager, gateway := newHandlerTestRuntime(t, handlerTestConfig(), func(r *http.Request) (*http.Response, error) {
		if r.URL.Host == "cloudflare.com" {
			return stubResponse(200, "ok"), nil
		}
		return stubResponse(401, `{"error":{"message":"invalid key"}}`), nil
	})
	recorder := serveTestRequest(manager.Handler(), http.MethodPost, "/v1/chat/completions",
		`{"model":"example-model","messages":[{"role":"user","content":"hello"}]}`)
	node := gateway.zenNodes.nodes[0]
	if recorder.Code != 401 || node.failures.Load() != 1 || node.cooldownUntil.Load() <= time.Now().UnixNano() {
		t.Fatalf("production failure was not recorded: status=%d failures=%d", recorder.Code, node.failures.Load())
	}
}

func TestSelectedDiagnosticClassifiesActualAttempt(t *testing.T) {
	for _, test := range []struct {
		name   string
		status int
		err    error
		want   string
	}{
		{name: "connection failure", err: errors.New("connection failed"), want: "transport_error"},
		{name: "header timeout", err: context.DeadlineExceeded, want: "transport_error"},
		{name: "upstream error", status: 503, want: "upstream_error"},
		{name: "invalid key", status: 401, want: "rejected"},
		{name: "rate limit", status: 429, want: "rate_limited"},
		{name: "bad request", status: 422, want: "request_error"},
		{name: "success", status: 200, want: "usable"},
	} {
		t.Run(test.name, func(t *testing.T) {
			manager, gateway := newHandlerTestRuntime(t, handlerTestConfig(), func(_ *http.Request) (*http.Response, error) {
				if test.err != nil {
					return nil, test.err
				}
				return stubResponse(test.status, `{"choices":[{"message":{"content":"ok"}}]}`), nil
			})
			result := runDebugTest(t, manager, DebugKeySelection{
				Mode: "selected", Tier: TierZen, ID: secretFingerprint(gateway.cfg.ZenKeys[0]),
			}, "example-model")
			if result.KeyTest != test.want || result.Route.Attempts != 1 || result.RequestID == "" {
				t.Fatalf("diagnostic = %+v, want key_test=%s with one traced attempt", result, test.want)
			}
			assertPoolUntouched(t, gateway)
		})
	}
}

func TestAutomaticDiagnosticPreservesTierFallback(t *testing.T) {
	cfg := handlerTestConfig()
	cfg.Anonymous = true
	cfg.GoKeys = []string{"go-handler-test-key"}
	cfg.Prefer = TierGo
	cfg.Proxies = []string{"http://first.invalid:8080", "http://second.invalid:8080"}
	cfg.Models.Protocols = nil
	var attempts []string
	manager, gateway := newHandlerTestRuntime(t, cfg, func(r *http.Request) (*http.Response, error) {
		switch {
		case r.Header.Get("x-api-key") == anonymousZenKey:
			attempts = append(attempts, "anonymous")
			return stubResponse(429, `{"error":{"message":"anonymous limit"}}`), nil
		case r.Header.Get("Authorization") == "Bearer "+cfg.GoKeys[0]:
			attempts = append(attempts, "go")
			if r.URL.Path != "/zen/go/v1/chat/completions" {
				t.Fatalf("Go request used the wrong protocol: %s", r.URL.Path)
			}
			return stubResponse(503, `{"error":{"message":"unavailable"}}`), nil
		default:
			attempts = append(attempts, "zen")
			if r.Header.Get("x-api-key") != cfg.ZenKeys[0] || r.URL.Path != "/zen/v1/messages" {
				t.Fatalf("Zen request used the wrong credential or protocol: %s", r.URL.Path)
			}
			return stubResponse(200, `{"id":"msg_test","model":"example-free","type":"message","role":"assistant","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`), nil
		}
	})
	gateway.catalog.ReplaceWithCapabilities([]string{"example-free"}, []string{"example-free"},
		map[Tier]map[string]Protocol{
			TierZen: {"example-free": ProtocolAnthropic},
			TierGo:  {"example-free": ProtocolChat},
		}, nil, nil)
	before := snapshotTestPools(gateway)
	result := runDebugTest(t, manager, DebugKeySelection{Mode: "auto"}, "example-free")
	if want := []string{"anonymous", "anonymous", "go", "zen"}; !reflect.DeepEqual(attempts, want) {
		t.Fatalf("fallback attempts = %v, want %v", attempts, want)
	}
	if !result.OK || result.Route.Tier != TierZen || result.Route.NativeProtocol != ProtocolAnthropic || result.Route.Anonymous || result.Route.Attempts != 4 {
		t.Fatalf("incorrect final route: %+v", result)
	}
	if after := snapshotTestPools(gateway); !reflect.DeepEqual(after, before) {
		t.Fatalf("fallback changed production state: before=%+v after=%+v", before, after)
	}
}

func TestSelectedDiagnosticDoesNotReplayStaleReasoning(t *testing.T) {
	cfg := handlerTestConfig()
	cfg.Models.Protocols["example-model"] = "responses"
	var calls int
	manager, gateway := newHandlerTestRuntime(t, cfg, func(_ *http.Request) (*http.Response, error) {
		calls++
		return stubResponse(400, `{"error":{"message":"Referenced reasoning item rs_old was not found"}}`), nil
	})
	admin := NewAdminServer(manager, manager.monitor, manager.hub, manager.logger)
	body, err := json.Marshal(DebugInferenceRequest{
		Protocol: ProtocolResponses,
		Key:      DebugKeySelection{Mode: "selected", Tier: TierZen, ID: secretFingerprint(gateway.cfg.ZenKeys[0])},
		Request: map[string]any{
			"model":                "example-model",
			"previous_response_id": "resp_old",
			"input":                []any{map[string]any{"type": "reasoning", "id": "rs_old", "summary": []any{}}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	recorder := httptest.NewRecorder()
	admin.handleDebugInference(recorder, httptest.NewRequest(http.MethodPost, "/api/debug/inference", strings.NewReader(string(body))))
	if recorder.Code != 200 || calls != 1 {
		t.Fatalf("selected diagnostic replayed: calls=%d response=%s", calls, recorder.Body.String())
	}
}
