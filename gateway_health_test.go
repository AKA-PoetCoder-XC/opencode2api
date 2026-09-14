package main

import (
	"encoding/json"
	"net/http"
	"reflect"
	"testing"
	"time"
)

func TestHealthAndModelsUseTheSameRoutes(t *testing.T) {
	for _, test := range []struct {
		name       string
		anonymous  bool
		zenKeys    bool
		zenModels  []string
		goModels   []string
		stale      bool
		wantModels []string
		wantStatus int
	}{
		{name: "anonymous paid only", anonymous: true, zenModels: []string{"example-model"}, wantStatus: 503},
		{name: "anonymous free", anonymous: true, zenModels: []string{"example-model", "example-free"}, wantModels: []string{"example-free"}, wantStatus: 200},
		{name: "configured key", zenKeys: true, zenModels: []string{"example-model"}, wantModels: []string{"example-model"}, wantStatus: 200},
		{name: "cached models for a removed tier", zenKeys: true, goModels: []string{"example-model"}, wantStatus: 503},
		{name: "stale usable catalog", zenKeys: true, zenModels: []string{"example-model"}, stale: true, wantModels: []string{"example-model"}, wantStatus: 200},
	} {
		t.Run(test.name, func(t *testing.T) {
			cfg := handlerTestConfig()
			cfg.Anonymous = test.anonymous
			if !test.zenKeys {
				cfg.ZenKeys = nil
			}
			manager, gateway := newHandlerTestRuntime(t, cfg, func(_ *http.Request) (*http.Response, error) {
				t.Fatal("discovery and health must not make upstream requests")
				return nil, nil
			})
			gateway.catalog.Replace([]string{}, []string{})
			gateway.catalog.Replace(test.zenModels, test.goModels)
			if test.stale {
				gateway.catalog.updatedAt = time.Now().Add(-time.Hour)
			}
			modelsRecorder := serveTestRequest(manager.Handler(), http.MethodGet, "/v1/models", "")
			var models struct {
				Data []struct {
					ID string `json:"id"`
				} `json:"data"`
			}
			if err := json.Unmarshal(modelsRecorder.Body.Bytes(), &models); err != nil {
				t.Fatal(err)
			}
			var ids []string
			for _, model := range models.Data {
				ids = append(ids, model.ID)
			}
			if !reflect.DeepEqual(ids, test.wantModels) {
				t.Fatalf("models = %v, want %v", ids, test.wantModels)
			}
			before := manager.monitor.Snapshot().Lifetime.Total
			healthRecorder := serveTestRequest(manager.Handler(), http.MethodGet, "/healthz", "")
			var health healthResponse
			if err := json.Unmarshal(healthRecorder.Body.Bytes(), &health); err != nil {
				t.Fatal(err)
			}
			if healthRecorder.Code != test.wantStatus || health.Ready != (test.wantStatus == 200) || health.Models.Exposed != len(ids) {
				t.Fatalf("health status=%d body=%s, discovered=%v", healthRecorder.Code, healthRecorder.Body.String(), ids)
			}
			if health.Models.Stale != test.stale || manager.Resources().Models.Exposed != len(ids) {
				t.Fatalf("resource snapshot disagrees with discovery: %+v", health)
			}
			if manager.monitor.Snapshot().Lifetime.Total != before {
				t.Fatal("health check changed request metrics")
			}
		})
	}
}
