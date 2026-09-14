package main

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
)

// A per-key diagnostic runs the selected key and nothing else: no anonymous
// lane, no other key, no other tier.
func TestSelectedKeyRunsExactlyOneAttempt(t *testing.T) {
	var hits int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&hits, 1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"bad request"}}`))
	}))
	defer upstream.Close()

	g := newTestGateway(t, &bytes.Buffer{})
	g.cfg.Upstream = UpstreamConfig{Zen: upstream.URL, Go: upstream.URL}
	node := g.zenNodes.nodes[0]

	route := modelRoute{
		ID: "example-model", Tier: TierZen, Protocol: ProtocolChat,
		Protocols: map[Tier]Protocol{TierZen: ProtocolChat}, KeyTiers: []Tier{TierZen},
	}
	bodies := map[Tier][]byte{TierZen: []byte(`{"model":"example-model"}`)}
	ids := requestIDs{Session: "ses_test", Request: "req_test", Project: "prj_test"}

	resp, err, attempts := g.doSelectedKeyUpstream(context.Background(), route, bodies, ids,
		debugKeyOverride{Tier: TierZen, KeyID: secretFingerprint(node.key)}, 0)
	if err != nil {
		t.Fatalf("doSelectedKeyUpstream: %v", err)
	}
	if resp == nil {
		t.Fatal("expected the upstream response")
	}
	defer drainAndClose(resp.Body)
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
	if attempts != 1 {
		t.Fatalf("attempts = %d, want 1", attempts)
	}
	if got := atomic.LoadInt32(&hits); got != 1 {
		t.Fatalf("upstream hits = %d, want exactly 1", got)
	}
}

// The diagnostic must never cool a production key or evict a proxy: that is
// what made a bad Playground request degrade live traffic.
func TestSelectedKeyDiagnosticLeavesPoolStateUntouched(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"invalid key"}}`))
	}))
	defer upstream.Close()

	g := newTestGateway(t, &bytes.Buffer{})
	g.cfg.Upstream = UpstreamConfig{Zen: upstream.URL, Go: upstream.URL}
	node := g.zenNodes.nodes[0]

	route := modelRoute{
		ID: "example-model", Tier: TierZen, Protocol: ProtocolChat,
		Protocols: map[Tier]Protocol{TierZen: ProtocolChat}, KeyTiers: []Tier{TierZen},
	}
	bodies := map[Tier][]byte{TierZen: []byte(`{"model":"example-model"}`)}
	ids := requestIDs{Session: "ses_test", Request: "req_test", Project: "prj_test"}

	for attempt := 0; attempt < 3; attempt++ {
		resp, err, _ := g.doSelectedKeyUpstream(context.Background(), route, bodies, ids,
			debugKeyOverride{Tier: TierZen, KeyID: secretFingerprint(node.key)}, 0)
		if err != nil {
			t.Fatalf("doSelectedKeyUpstream: %v", err)
		}
		drainAndClose(resp.Body)
	}

	// Repeated 401s would normally trip the cooldown and mark the proxy failed.
	assertPoolUntouched(t, g)
}

func TestSelectedKeyRejectsUnknownFingerprint(t *testing.T) {
	g := newTestGateway(t, &bytes.Buffer{})
	route := modelRoute{ID: "example-model", Tier: TierZen, Protocol: ProtocolChat}
	bodies := map[Tier][]byte{TierZen: []byte(`{"model":"example-model"}`)}

	resp, err, attempts := g.doSelectedKeyUpstream(context.Background(), route, bodies,
		requestIDs{Request: "req_test"}, debugKeyOverride{Tier: TierZen, KeyID: "deadbeef"}, 0)
	if resp != nil || err == nil {
		t.Fatalf("response = %v, err = %v; want a lookup failure", resp, err)
	}
	if attempts != 0 {
		t.Fatalf("attempts = %d, want 0", attempts)
	}
}

// The pinned route must describe the selected tier only, and must carry that
// tier's own native protocol so the body is re-encoded for it.
func TestRouteForTierPinsOneTier(t *testing.T) {
	catalog := newModelCatalog(TierGo, nil)
	catalog.ReplaceWithCapabilities(
		[]string{"zen-only", "shared"}, []string{"go-only", "shared"},
		map[Tier]map[string]Protocol{
			TierZen: {"zen-only": ProtocolChat, "shared": ProtocolChat},
			TierGo:  {"go-only": ProtocolAnthropic, "shared": ProtocolAnthropic},
		}, nil, nil,
	)

	route, err := catalog.RouteForTier("zen-only", TierZen, true, true)
	if err != nil {
		t.Fatalf("RouteForTier: %v", err)
	}
	if route.Tier != TierZen || route.Anonymous {
		t.Fatalf("route = %+v, want a non-anonymous zen route", route)
	}
	if len(route.KeyTiers) != 1 || route.KeyTiers[0] != TierZen {
		t.Fatalf("key tiers = %v, want exactly [zen]", route.KeyTiers)
	}

	shared, err := catalog.RouteForTier("shared", TierGo, true, true)
	if err != nil {
		t.Fatalf("RouteForTier: %v", err)
	}
	if shared.Protocol != ProtocolAnthropic {
		t.Fatalf("protocol = %q, want the Go tier's own %q", shared.Protocol, ProtocolAnthropic)
	}

	if _, err := catalog.RouteForTier("zen-only", TierGo, true, true); err == nil {
		t.Fatal("a model missing from the selected tier must be rejected")
	}
	if _, err := catalog.RouteForTier("shared", TierGo, true, false); err == nil {
		t.Fatal("a tier without a configured key must be rejected")
	}
}

func TestClassifyKeyTest(t *testing.T) {
	cases := []struct {
		name     string
		status   int
		outcome  string
		expected string
	}{
		{"success", http.StatusOK, "success", "usable"},
		{"unauthorized", http.StatusUnauthorized, "rejected", "rejected"},
		{"forbidden", http.StatusForbidden, "rejected", "rejected"},
		{"rate limited", http.StatusTooManyRequests, "retryable_failure", "rate_limited"},
		{"transport", 0, "transport_error", "transport_error"},
		{"upstream 500", http.StatusInternalServerError, "retryable_failure", "upstream_error"},
		{"bad request", http.StatusBadRequest, "rejected", "request_error"},
	}
	for _, item := range cases {
		if got := classifyKeyTest(item.status, item.outcome); got != item.expected {
			t.Errorf("%s: classifyKeyTest(%d, %q) = %q, want %q", item.name, item.status, item.outcome, got, item.expected)
		}
	}
}

// The selected tier decides where the request goes, even when the preferred
// tier also advertises the model.
func TestSelectedKeyUsesItsOwnTier(t *testing.T) {
	var zenHits, goHits int32
	zen := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&zenHits, 1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"zen"}}]}`))
	}))
	defer zen.Close()
	goUpstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&goHits, 1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"go"}}]}`))
	}))
	defer goUpstream.Close()

	g := newTestGateway(t, &bytes.Buffer{})
	g.cfg.Upstream = UpstreamConfig{Zen: zen.URL, Go: goUpstream.URL}
	node := g.goNodes.nodes[0]

	route := modelRoute{
		ID: "example-model", Tier: TierGo, Protocol: ProtocolChat,
		Protocols: map[Tier]Protocol{TierZen: ProtocolChat, TierGo: ProtocolChat}, KeyTiers: []Tier{TierGo},
	}
	bodies := map[Tier][]byte{
		TierZen: []byte(`{"model":"example-model"}`),
		TierGo:  []byte(`{"model":"example-model"}`),
	}

	resp, err, _ := g.doSelectedKeyUpstream(context.Background(), route, bodies,
		requestIDs{Request: "req_test"}, debugKeyOverride{Tier: TierGo, KeyID: secretFingerprint(node.key)}, 0)
	if err != nil {
		t.Fatalf("doSelectedKeyUpstream: %v", err)
	}
	drainAndClose(resp.Body)

	if got := atomic.LoadInt32(&goHits); got != 1 {
		t.Fatalf("go tier hits = %d, want 1", got)
	}
	if got := atomic.LoadInt32(&zenHits); got != 0 {
		t.Fatalf("zen tier hits = %d, want 0", got)
	}
}
