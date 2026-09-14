package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func streamTestPayload(protocol Protocol) string {
	payload := map[string]any{"model": "example-model", "stream": true}
	if protocol == ProtocolResponses {
		payload["input"] = "hello"
	} else {
		payload["messages"] = []any{map[string]any{"role": "user", "content": "hello"}}
		payload["max_tokens"] = 16
	}
	body, _ := json.Marshal(payload)
	return string(body)
}

func streamTestFixture(protocol Protocol, outcome string) string {
	if outcome == "malformed" {
		return "data: {invalid JSON}\n\n"
	}
	if outcome == "error" {
		switch protocol {
		case ProtocolAnthropic:
			return "event: error\ndata: " + `{"type":"error","error":{"type":"overloaded_error","message":"mock stream failure"}}` + "\n\n"
		case ProtocolResponses:
			return "event: response.failed\ndata: " + `{"type":"response.failed","response":{"id":"resp_test","status":"failed","error":{"code":"server_error","message":"mock stream failure"}}}` + "\n\n"
		default:
			return "data: " + `{"error":{"type":"server_error","message":"mock stream failure"}}` + "\n\ndata: [DONE]\n\n"
		}
	}
	var prefix, suffix string
	switch protocol {
	case ProtocolAnthropic:
		prefix = "event: message_start\ndata: " + `{"type":"message_start","message":{"id":"msg_test","model":"example-model","role":"assistant","content":[],"usage":{"input_tokens":3,"output_tokens":0}}}` + "\n\n" +
			"event: content_block_start\ndata: " + `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
			"event: content_block_delta\ndata: " + `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}` + "\n\n"
		suffix = "event: content_block_stop\ndata: " + `{"type":"content_block_stop","index":0}` + "\n\n" +
			"event: message_delta\ndata: " + `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}` + "\n\n" +
			"event: message_stop\ndata: " + `{"type":"message_stop"}` + "\n\n"
	case ProtocolResponses:
		prefix = "event: response.created\ndata: " + `{"type":"response.created","response":{"id":"resp_test","model":"example-model","status":"in_progress"}}` + "\n\n" +
			"event: response.output_text.delta\ndata: " + `{"type":"response.output_text.delta","item_id":"msg_test","output_index":0,"content_index":0,"delta":"hello"}` + "\n\n"
		suffix = "event: response.completed\ndata: " + `{"type":"response.completed","response":{"id":"resp_test","model":"example-model","status":"completed","output":[{"type":"message","id":"msg_test","role":"assistant","content":[{"type":"output_text","text":"hello"}]}],"usage":{"input_tokens":3,"output_tokens":2,"total_tokens":5}}}` + "\n\n"
	default:
		prefix = "data: " + `{"id":"chatcmpl_test","model":"example-model","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"}}]}` + "\n\n"
		suffix = "data: " + `{"id":"chatcmpl_test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}` + "\n\ndata: [DONE]\n\n"
	}
	if outcome == "truncated" {
		return prefix
	}
	return prefix + suffix
}

func TestStreamingResultsAcrossProtocols(t *testing.T) {
	protocols := []Protocol{ProtocolChat, ProtocolResponses, ProtocolAnthropic}
	for _, upstream := range protocols {
		for _, client := range protocols {
			for _, outcome := range []string{"success", "error", "malformed", "truncated"} {
				t.Run(string(upstream)+"/"+string(client)+"/"+outcome, func(t *testing.T) {
					cfg := handlerTestConfig()
					cfg.Models.Protocols["example-model"] = string(upstream)
					manager, _ := newHandlerTestRuntime(t, cfg, func(r *http.Request) (*http.Response, error) {
						if want := "/zen" + protocolPath(upstream); r.URL.Path != want {
							t.Fatalf("upstream path = %q, want %q", r.URL.Path, want)
						}
						response := stubResponse(200, streamTestFixture(upstream, outcome))
						response.Header.Set("Content-Type", "text/event-stream")
						return response, nil
					})
					recorder := serveTestRequest(manager.Handler(), http.MethodPost, protocolPath(client), streamTestPayload(client))
					if recorder.Code != 200 {
						t.Fatalf("HTTP status = %d: %s", recorder.Code, recorder.Body.String())
					}
					observer := newStreamUsageObserver(client)
					_, _ = observer.Write(recorder.Body.Bytes())
					observer.Finish()
					snapshot := manager.monitor.Snapshot()
					if snapshot.Lifetime.Total != 1 || snapshot.Active != 0 || snapshot.ActiveStreams != 0 || len(snapshot.Upstream.Requests) != 1 {
						t.Fatalf("unexpected completed request metrics: %+v", snapshot)
					}
					request := snapshot.Upstream.Requests[0]
					if request.Status != 200 || request.RequestID == "" {
						t.Fatalf("HTTP status or request trace lost: %+v", request)
					}
					if outcome == "success" {
						if !observer.NormalTermination() || observer.ErrorTermination() || observer.ParseError() != nil {
							t.Fatalf("invalid successful client stream: %s", recorder.Body.String())
						}
						if snapshot.Lifetime.Success != 1 || snapshot.Lifetime.Errors != 0 || !request.Success || request.Outcome != "success" {
							t.Fatalf("normal stream misclassified: %+v, %+v", snapshot.Lifetime, request)
						}
						tokens := snapshot.Usage.Lifetime.Tokens
						if tokens.Input != 3 || tokens.Output != 2 || tokens.Total != 5 {
							t.Fatalf("usage was lost in the stream bridge: %+v", tokens)
						}
					} else {
						if !observer.ErrorTermination() {
							t.Fatalf("client did not receive a terminal error: %s", recorder.Body.String())
						}
						if snapshot.Lifetime.Success != 0 || snapshot.Lifetime.Errors != 1 || request.Success || request.Outcome != "stream_error" {
							t.Fatalf("failed stream misclassified: %+v, %+v", snapshot.Lifetime, request)
						}
						if snapshot.Window.Errors != 1 {
							t.Fatalf("rolling metrics missed the error: %+v", snapshot.Window)
						}
					}
				})
			}
		}
	}
}

type terminalTestReader struct {
	err    error
	cancel context.CancelFunc
}

func (reader terminalTestReader) Read([]byte) (int, error) {
	if reader.cancel != nil {
		reader.cancel()
	}
	return 0, reader.err
}

func TestStreamReadFailureAndClientCancellation(t *testing.T) {
	for _, client := range []Protocol{ProtocolChat, ProtocolResponses, ProtocolAnthropic} {
		for _, canceled := range []bool{false, true} {
			name := string(client) + "/read_error"
			if canceled {
				name = string(client) + "/canceled"
			}
			t.Run(name, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				terminal := terminalTestReader{err: io.ErrUnexpectedEOF}
				if canceled {
					terminal.err, terminal.cancel = context.Canceled, cancel
				}
				manager, _ := newHandlerTestRuntime(t, handlerTestConfig(), func(_ *http.Request) (*http.Response, error) {
					response := stubResponse(200, "")
					response.Body = io.NopCloser(io.MultiReader(strings.NewReader(streamTestFixture(ProtocolChat, "truncated")), terminal))
					return response, nil
				})
				request := httptest.NewRequest(http.MethodPost, protocolPath(client), strings.NewReader(streamTestPayload(client))).WithContext(ctx)
				request.Header.Set("Authorization", "Bearer "+testServerKey)
				recorder := httptest.NewRecorder()
				manager.Handler().ServeHTTP(recorder, request)
				snapshot := manager.monitor.Snapshot()
				wantOutcome := "stream_error"
				if canceled {
					wantOutcome = "client_canceled"
					if strings.Contains(recorder.Body.String(), "upstream_error") {
						t.Fatal("client cancellation emitted a synthetic upstream error")
					}
				}
				if recorder.Code != 200 || snapshot.Lifetime.Errors != 1 || snapshot.Lifetime.Success != 0 || snapshot.Upstream.Requests[0].Outcome != wantOutcome {
					t.Fatalf("unexpected termination: %+v, %+v", snapshot.Lifetime, snapshot.Upstream.Requests)
				}
			})
		}
	}
}

func TestStreamErrorsAreReturnedAfterDelivery(t *testing.T) {
	for _, target := range []Protocol{ProtocolChat, ProtocolResponses, ProtocolAnthropic} {
		recorder := httptest.NewRecorder()
		_, _, err := transcodeStreamWithUsage(recorder, strings.NewReader(streamTestFixture(ProtocolChat, "error")), ProtocolChat, target, "example-model")
		if !errors.Is(err, errStreamUpstreamFailure) {
			t.Fatalf("%s: terminal error = %v", target, err)
		}
	}
}
