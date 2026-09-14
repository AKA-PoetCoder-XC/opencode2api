package main

import (
	"bytes"
	"context"
	"log/slog"
	"testing"
	"time"
)

func testConfig() Config {
	return Config{
		Upstream: UpstreamConfig{Zen: "https://example.invalid/zen", Go: "https://example.invalid/zen/go"},
		Retry:    RetryConfig{MaxAttempts: 3, TimeoutSeconds: 30},
		Performance: PerformanceConfig{
			MaxIdleConns: 16, MaxIdleConnsPerHost: 8, MaxConnsPerHost: 0,
			IdleConnTimeoutSeconds: 30, ConnectTimeoutSeconds: 1, FailureCooldownSeconds: 15,
		},
		Logging: LoggingConfig{Level: "error", RingSize: 2000},
		Models:  ModelsConfig{RefreshSeconds: 300, Protocols: map[string]string{}},
		Prefer:  TierGo,
	}
}

func newTestGateway(t *testing.T, sink *bytes.Buffer) *Gateway {
	t.Helper()
	cfg := testConfig()
	transports, err := newTransportPool([]string{"direct"}, cfg.Performance, time.Second)
	if err != nil {
		t.Fatalf("newTransportPool: %v", err)
	}
	cooldown := time.Duration(cfg.Performance.FailureCooldownSeconds) * time.Second
	zenNodes, err := newNodePool([]string{"sk-zen-test"}, transports, cooldown)
	if err != nil {
		t.Fatalf("newNodePool: %v", err)
	}
	goNodes, err := newNodePool([]string{"sk-go-test"}, transports, cooldown)
	if err != nil {
		t.Fatalf("newNodePool: %v", err)
	}
	var handler slog.Handler = slog.NewTextHandler(sink, &slog.HandlerOptions{Level: slog.LevelDebug})
	return &Gateway{
		cfg: cfg, logger: slog.New(handler), transports: transports, zenNodes: zenNodes, goNodes: goNodes,
		anonymous: newAnonymousPool(true, transports, cooldown),
	}
}

func assertPoolUntouched(t *testing.T, g *Gateway) {
	t.Helper()
	for _, pool := range []*nodePool{g.zenNodes, g.goNodes} {
		for _, node := range pool.nodes {
			if failures := node.failures.Load(); failures != 0 {
				t.Fatalf("key %s recorded %d failures", node.key, failures)
			}
			if until := node.cooldownUntil.Load(); until != 0 {
				t.Fatalf("key %s was cooled until %d", node.key, until)
			}
		}
	}
	for _, proxy := range g.transports.items {
		if !proxy.healthy.Load() {
			t.Fatal("healthy proxy was evicted")
		}
	}
}

// An exhausted request budget must not be spent rotating keys: every attempt
// would fail instantly, cool a healthy key and evict a healthy proxy.
func TestExpiredContextDoesNotCoolKeysOrEvictProxies(t *testing.T) {
	g := newTestGateway(t, &bytes.Buffer{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	route := modelRoute{ID: "example-model", Tier: TierZen, Protocol: ProtocolChat, KeyTiers: []Tier{TierZen}}
	bodies := map[Tier][]byte{TierZen: []byte(`{"model":"example-model"}`)}
	ids := requestIDs{Session: "ses_test", Request: "req_test", Project: "prj_test"}

	resp, err, attempts := g.doKeyUpstream(ctx, route, bodies, ids, 0)
	if resp != nil {
		t.Fatalf("response = %v, want nil", resp)
	}
	if err == nil {
		t.Fatal("expected an error from the exhausted context")
	}
	if attempts != 0 {
		t.Fatalf("attempts = %d, want 0", attempts)
	}
	assertPoolUntouched(t, g)
}

// The same guard applies to the anonymous phase, and the authenticated fallback
// must not run once the budget is gone.
func TestExpiredContextSkipsAnonymousScanAndKeyFallback(t *testing.T) {
	g := newTestGateway(t, &bytes.Buffer{})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	route := modelRoute{
		ID: "example-model", Tier: TierZen, Protocol: ProtocolChat,
		Anonymous: true, KeyTiers: []Tier{TierZen, TierGo},
	}
	bodies := map[Tier][]byte{
		TierZen: []byte(`{"model":"example-model"}`),
		TierGo:  []byte(`{"model":"example-model"}`),
	}
	ids := requestIDs{Session: "ses_test", Request: "req_test", Project: "prj_test"}

	resp, _, attempts, err := g.doUpstreamTiers(ctx, route, bodies, ids, 0)
	if resp != nil {
		t.Fatalf("response = %v, want nil", resp)
	}
	if err == nil {
		t.Fatal("expected an error from the exhausted context")
	}
	if attempts != 0 {
		t.Fatalf("attempts = %d, want 0", attempts)
	}
	assertPoolUntouched(t, g)
	for _, node := range g.anonymous.nodes {
		if until := node.cooldownUntil.Load(); until != 0 {
			t.Fatalf("anonymous node was cooled until %d", until)
		}
	}
}

// The body dump is off unless configured, and when enabled must carry the
// reference ids a retry investigation needs.
func TestDumpOutboundBodiesIsOptIn(t *testing.T) {
	var sink bytes.Buffer
	g := newTestGateway(t, &sink)
	route := modelRoute{ID: "example-model", Tier: TierZen, Protocol: ProtocolResponses}
	bodies := map[Tier][]byte{TierZen: []byte(`{"model":"example-model","previous_response_id":"resp_1"}`)}
	ids := requestIDs{Request: "req_test"}

	g.dumpOutboundBodies(route, bodies, ids, 0)
	if sink.Len() != 0 {
		t.Fatalf("dump emitted while disabled: %s", sink.String())
	}

	g.cfg.Logging.DumpRequestBodies = true
	g.dumpOutboundBodies(route, bodies, ids, 1)
	dumped := sink.String()
	for _, want := range []string{"upstream_request_body", "previous_response_id", "phase=retry"} {
		if !bytes.Contains([]byte(dumped), []byte(want)) {
			t.Fatalf("dump is missing %q: %s", want, dumped)
		}
	}
}
