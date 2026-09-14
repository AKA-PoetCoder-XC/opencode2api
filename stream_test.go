package main

import (
	"testing"
)

func newResponsesParser() *bridgeStreamParser {
	return &bridgeStreamParser{
		protocol: ProtocolResponses, tools: map[string]bool{}, toolIDs: map[string]string{},
		toolNames: map[string]string{}, responseArgs: map[string]bool{}, responseReasoning: map[string]bool{},
	}
}

func parseAll(t *testing.T, parser *bridgeStreamParser, event string) []bridgeStreamEvent {
	t.Helper()
	events, err := parser.Parse("", event)
	if err != nil {
		t.Fatalf("parse %s: %v", event, err)
	}
	return events
}

func finishStop(t *testing.T, events []bridgeStreamEvent) string {
	t.Helper()
	for _, event := range events {
		if event.Kind == "finish" {
			return event.Stop
		}
	}
	t.Fatalf("no finish event in %v", events)
	return ""
}

// A completed Responses turn that carries function_call items ended waiting for
// tool results. Reporting "stop" made every tool step end the downstream turn
// instead of chaining inside it.
func TestResponsesCompletedWithToolCallFinishesAsToolCalls(t *testing.T) {
	parser := newResponsesParser()
	parseAll(t, parser, `{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}}`)

	events := parseAll(t, parser, `{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}]}}`)
	if stop := finishStop(t, events); stop != "tool_calls" {
		t.Fatalf("stop = %q, want tool_calls", stop)
	}
}

// The completed event is authoritative even when no deltas were observed first.
func TestResponsesCompletedWithToolCallOnlyInOutput(t *testing.T) {
	parser := newResponsesParser()
	events := parseAll(t, parser, `{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}]}}`)
	if stop := finishStop(t, events); stop != "tool_calls" {
		t.Fatalf("stop = %q, want tool_calls", stop)
	}
}

func TestResponsesCompletedWithoutToolsFinishesAsStop(t *testing.T) {
	parser := newResponsesParser()
	parseAll(t, parser, `{"type":"response.output_text.delta","delta":"hello"}`)

	events := parseAll(t, parser, `{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}]}}`)
	if stop := finishStop(t, events); stop != "stop" {
		t.Fatalf("stop = %q, want stop", stop)
	}
}

// An incomplete turn keeps its mapped reason instead of being reclassified by
// the tool-call scan.
func TestResponsesIncompleteKeepsMappedReason(t *testing.T) {
	parser := newResponsesParser()
	events := parseAll(t, parser, `{"type":"response.incomplete","response":{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[{"id":"fc_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}]}}`)
	if stop := finishStop(t, events); stop != "length" {
		t.Fatalf("stop = %q, want length", stop)
	}
}
