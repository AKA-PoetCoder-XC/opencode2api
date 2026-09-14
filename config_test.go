package main

import (
	"testing"
	"time"
)

// The knob must be inert by default: existing configs keep using the
// request-level retry timeout.
func TestAttemptTimeoutDefaultsToRequestTimeout(t *testing.T) {
	request := 45 * time.Second
	if got := (PerformanceConfig{}).AttemptTimeout(request); got != request {
		t.Fatalf("AttemptTimeout = %v, want %v", got, request)
	}
	if got := (PerformanceConfig{AttemptTimeoutSeconds: -1}).AttemptTimeout(request); got != request {
		t.Fatalf("negative value = %v, want %v", got, request)
	}
}

func TestAttemptTimeoutIsBoundedByRequestTimeout(t *testing.T) {
	request := 45 * time.Second
	if got := (PerformanceConfig{AttemptTimeoutSeconds: 10}).AttemptTimeout(request); got != 10*time.Second {
		t.Fatalf("AttemptTimeout = %v, want 10s", got)
	}
	if got := (PerformanceConfig{AttemptTimeoutSeconds: 600}).AttemptTimeout(request); got != request {
		t.Fatalf("AttemptTimeout = %v, want the request timeout %v", got, request)
	}
}

func TestNormalizeConfigRejectsNegativeAttemptTimeout(t *testing.T) {
	cfg := testConfig()
	cfg.Listen = "127.0.0.1:8080"
	cfg.ServerKeys = []string{"local-key"}
	cfg.ZenKeys = []string{"sk-zen"}
	cfg.Performance.AttemptTimeoutSeconds = -5
	if _, err := NormalizeConfig("config.json", cfg); err == nil {
		t.Fatal("negative attempt_timeout_seconds must be rejected")
	}
}

// Adding a field to the logging section must not make the management API reject
// configs that carry the new key.
func TestNormalizeConfigAcceptsBodyDumpFlag(t *testing.T) {
	cfg := testConfig()
	cfg.Listen = "127.0.0.1:8080"
	cfg.ServerKeys = []string{"local-key"}
	cfg.ZenKeys = []string{"sk-zen"}
	cfg.Logging.DumpRequestBodies = true
	normalized, err := NormalizeConfig("config.json", cfg)
	if err != nil {
		t.Fatalf("NormalizeConfig: %v", err)
	}
	if !normalized.Logging.DumpRequestBodies {
		t.Fatal("dump_request_bodies was dropped during normalization")
	}
}
