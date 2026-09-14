package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestEmbeddedWebAssets(t *testing.T) {
	manager, _ := newHandlerTestRuntime(t, handlerTestConfig(), nil)
	admin := NewAdminServer(manager, manager.monitor, manager.hub, manager.logger)
	for _, test := range []struct {
		path        string
		contentType string
		contains    string
	}{
		{path: "/", contentType: "text/html", contains: `<script src="/app.js" defer></script>`},
		{path: "/app.js", contentType: "javascript", contains: "async function boot()"},
		{path: "/styles.css", contentType: "text/css", contains: "#account-form"},
	} {
		t.Run(test.path, func(t *testing.T) {
			recorder := httptest.NewRecorder()
			admin.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, test.path, nil))
			if recorder.Code != 200 || !strings.Contains(recorder.Header().Get("Content-Type"), test.contentType) || !strings.Contains(recorder.Body.String(), test.contains) {
				t.Fatalf("asset %s: status=%d content-type=%s", test.path, recorder.Code, recorder.Header().Get("Content-Type"))
			}
			if policy := recorder.Header().Get("Content-Security-Policy"); strings.Contains(policy, "unsafe-inline") || !strings.Contains(policy, "script-src 'self'") {
				t.Fatalf("unexpected asset policy: %s", policy)
			}
		})
	}
}

func TestManagementEndpointsRequireSession(t *testing.T) {
	manager, _ := newHandlerTestRuntime(t, handlerTestConfig(), nil)
	admin := NewAdminServer(manager, manager.monitor, manager.hub, manager.logger)
	for _, path := range []string{"/api/config", "/api/monitor", "/api/debug/models", "/api/logs", "/api/logs/stream"} {
		recorder := httptest.NewRecorder()
		admin.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, path, nil))
		if recorder.Code != http.StatusUnauthorized {
			t.Errorf("%s: unauthenticated status = %d", path, recorder.Code)
		}
	}
}
