package main

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
)

const testServerKey = "local-handler-test-key"

type stubTransport func(*http.Request) (*http.Response, error)

func (transport stubTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return transport(r)
}

func stubResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func handlerTestConfig() Config {
	cfg := testConfig()
	cfg.Listen = "127.0.0.1:8080"
	cfg.ServerKeys = []string{testServerKey}
	cfg.ZenKeys = []string{"zen-handler-test-key"}
	cfg.Proxies = []string{"direct"}
	cfg.Prefer = TierZen
	cfg.Retry.MaxAttempts = 1
	cfg.Models.Protocols = map[string]string{
		"example-model": "chat",
		"example-free":  "chat",
	}
	return cfg
}

// The handler harness exercises authentication, routing, conversion, and
// monitoring without starting background refreshes or contacting real services.
func newHandlerTestRuntime(t *testing.T, cfg Config, transport stubTransport) (*RuntimeManager, *Gateway) {
	t.Helper()
	configPath := filepath.Join(t.TempDir(), "config.json")
	cfg, err := NormalizeConfig(configPath, cfg)
	if err != nil {
		t.Fatal(err)
	}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	monitor := NewMonitor()
	gateway, err := NewGateway(cfg, logger, monitor)
	if err != nil {
		t.Fatal(err)
	}
	for _, proxy := range gateway.transports.items {
		proxy.client = &http.Client{Transport: transport}
	}
	gateway.catalog.Replace([]string{"example-model", "example-free"}, nil)
	redactor := NewSecretRedactor()
	redactor.Replace(cfg)
	manager := &RuntimeManager{
		configPath: configPath, root: context.Background(),
		logger: logger, monitor: monitor, hub: NewLogHub(100),
		redactor: redactor, level: new(slog.LevelVar),
		effective: effectiveListeners{API: cfg.Listen, WebUI: cfg.WebUI.Listen, WebUIEnabled: cfg.WebUI.Enabled},
	}
	manager.current.Store(&gatewayRuntime{
		config: cfg, gateway: gateway, handler: gateway.Handler(), cancel: func() {},
	})
	return manager, gateway
}

func serveTestRequest(handler http.Handler, method, path, body string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(method, path, strings.NewReader(body))
	request.Header.Set("Authorization", "Bearer "+testServerKey)
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, request)
	return recorder
}
