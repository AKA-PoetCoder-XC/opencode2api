package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"
)

type requestMeta struct {
	Model          string
	Tier           string
	Protocol       Protocol
	Request        string
	KeyID          string
	Channel        string
	Anonymous      bool
	Proxy          string
	Attempts       int
	Stream         bool
	Usage          bridgeUsage
	UsageReported  bool
	AttemptOutcome string
	Outcome        string
}

type requestMetaKey struct{}

func metaFromRequest(r *http.Request) *requestMeta {
	meta, _ := r.Context().Value(requestMetaKey{}).(*requestMeta)
	return meta
}

type statusWriter struct {
	http.ResponseWriter
	status int
	bytes  int64
}

func (w *statusWriter) WriteHeader(status int) {
	if w.status != 0 {
		return
	}
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}

func (w *statusWriter) Write(data []byte) (int, error) {
	if w.status == 0 {
		w.WriteHeader(http.StatusOK)
	}
	n, err := w.ResponseWriter.Write(data)
	w.bytes += int64(n)
	return n, err
}

func (w *statusWriter) Flush() {
	_ = http.NewResponseController(w.ResponseWriter).Flush()
}

func (w *statusWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

func monitorMiddleware(monitor *Monitor, logger *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			next.ServeHTTP(w, r)
			return
		}
		started := time.Now()
		meta := metaFromRequest(r)
		if meta == nil {
			meta = &requestMeta{}
			r = r.WithContext(context.WithValue(r.Context(), requestMetaKey{}, meta))
		}
		writer := &statusWriter{ResponseWriter: w}
		monitor.active.Add(1)
		defer func() {
			monitor.active.Add(-1)
			status := writer.status
			if status == 0 {
				status = http.StatusOK
			}
			duration := time.Since(started)
			monitor.Record(r.URL.Path, status, duration, meta)
			outcome := meta.Outcome
			if outcome == "" {
				outcome = requestOutcome(status, meta.Channel)
			}
			if meta.Channel != "" {
				logger.Info("request routed", "component", "http", "event", "request_routed", "method", r.Method,
					"path", r.URL.Path, "status", status, "duration_ms", duration.Milliseconds(), "request_id", meta.Request,
					"model", meta.Model, "tier", meta.Tier, "key_id", meta.KeyID, "channel", meta.Channel,
					"anonymous", meta.Anonymous, "attempts", meta.Attempts, "stream", meta.Stream, "outcome", outcome)
			}
			logger.Debug("request completed", "component", "http", "event", "request_complete", "method", r.Method,
				"path", r.URL.Path, "status", status, "duration_ms", duration.Milliseconds(), "bytes", writer.bytes,
				"request_id", meta.Request, "model", meta.Model, "tier", meta.Tier, "key_id", meta.KeyID,
				"channel", meta.Channel, "anonymous", meta.Anonymous, "attempts", meta.Attempts, "stream", meta.Stream, "outcome", outcome)
		}()
		next.ServeHTTP(writer, r)
	})
}

func encodeSSE(w http.ResponseWriter, event string, id uint64, value any) error {
	data, err := json.Marshal(value)
	if err != nil {
		return err
	}
	if id > 0 {
		_, _ = fmt.Fprintf(w, "id: %d\n", id)
	}
	if event != "" {
		_, _ = fmt.Fprintf(w, "event: %s\n", event)
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", data)
	return err
}
