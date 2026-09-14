package main

import (
	"encoding/json"
	"testing"
)

func mustConvert(t *testing.T, from, to Protocol, input map[string]any) map[string]any {
	t.Helper()
	output, err := convertRequest(from, to, input)
	if err != nil {
		t.Fatalf("convertRequest(%s -> %s): %v", from, to, err)
	}
	return output
}

// firstContentBlock returns the leading content block of an Anthropic-style
// request body built by the tests above.
func firstContentBlock(t *testing.T, input map[string]any) map[string]any {
	t.Helper()
	messages := sliceAt(input, "messages")
	if len(messages) == 0 {
		t.Fatal("converted request has no messages")
	}
	message, ok := messages[0].(map[string]any)
	if !ok {
		t.Fatalf("messages[0] is %T, want object", messages[0])
	}
	content := sliceAt(message, "content")
	if len(content) == 0 {
		t.Fatal("messages[0] has no content blocks")
	}
	block, ok := content[0].(map[string]any)
	if !ok {
		t.Fatalf("messages[0].content[0] is %T, want object", content[0])
	}
	return block
}

// A client that asks for JSON mode must not silently receive plain text. The
// sampling and response-shape knobs have to survive the bridge.
func TestChatToResponsesKeepsSamplingKnobs(t *testing.T) {
	output := mustConvert(t, ProtocolChat, ProtocolResponses, map[string]any{
		"model":               "example-model",
		"messages":            []any{map[string]any{"role": "user", "content": "hi"}},
		"response_format":     map[string]any{"type": "json_object"},
		"parallel_tool_calls": false,
		"temperature":         0.2,
	})

	format := mapAt(output, "text", "format")
	if stringAt(format, "type") != "json_object" {
		t.Fatalf("text.format.type = %q, want json_object (output: %v)", stringAt(format, "type"), output)
	}
	if value, exists := output["parallel_tool_calls"]; !exists || value != false {
		t.Fatalf("parallel_tool_calls = %v (present=%v), want explicit false", value, exists)
	}
}

func TestResponsesToChatFlattensJSONSchema(t *testing.T) {
	output := mustConvert(t, ProtocolResponses, ProtocolChat, map[string]any{
		"model": "example-model",
		"input": "hi",
		"text": map[string]any{"format": map[string]any{
			"type":   "json_schema",
			"name":   "answer",
			"strict": true,
			"schema": map[string]any{"type": "object"},
		}},
	})

	schema := mapAt(output, "response_format", "json_schema")
	if stringAt(schema, "name") != "answer" {
		t.Fatalf("response_format.json_schema.name = %q, want answer (output: %v)", stringAt(schema, "name"), output)
	}
	if stringAt(schema, "schema", "type") != "object" {
		t.Fatalf("response_format.json_schema.schema.type = %q, want object", stringAt(schema, "schema", "type"))
	}
}

func TestChatToAnthropicOmitsUnmappableKnobs(t *testing.T) {
	output := mustConvert(t, ProtocolChat, ProtocolAnthropic, map[string]any{
		"model":           "example-model",
		"messages":        []any{map[string]any{"role": "user", "content": "hi"}},
		"response_format": map[string]any{"type": "json_object"},
		"seed":            7,
	})

	// Anthropic has no native shape for these; they must be omitted rather than
	// forwarded as unknown fields.
	for _, key := range []string{"response_format", "seed", "frequency_penalty", "presence_penalty", "parallel_tool_calls"} {
		if _, exists := output[key]; exists {
			t.Fatalf("Anthropic request unexpectedly carries %q: %v", key, output)
		}
	}
}

// Chat history has no server-side reasoning id. Synthesizing one makes the
// upstream reject the request as a stale reference, which used to force every
// tool-bearing turn through the strip-and-retry path.
func TestChatReasoningReplayOmitsSyntheticReference(t *testing.T) {
	output := mustConvert(t, ProtocolChat, ProtocolResponses, map[string]any{
		"model": "example-model",
		"messages": []any{
			map[string]any{"role": "user", "content": "run it"},
			map[string]any{
				"role":              "assistant",
				"content":           "",
				"reasoning_content": "I should call the tool",
				"tool_calls": []any{map[string]any{
					"id":       "call_1",
					"type":     "function",
					"function": map[string]any{"name": "lookup", "arguments": `{"q":"x"}`},
				}},
			},
			map[string]any{"role": "tool", "tool_call_id": "call_1", "content": "result"},
		},
	})

	var sawFunctionCall bool
	for _, raw := range sliceAt(output, "input") {
		item, _ := raw.(map[string]any)
		switch stringAt(item, "type") {
		case "reasoning":
			t.Fatalf("replayed reasoning item has no upstream id and must be omitted: %v", item)
		case "function_call":
			sawFunctionCall = true
		}
	}
	if !sawFunctionCall {
		t.Fatalf("function_call item missing from converted input: %v", output["input"])
	}
}

// A reasoning block the upstream actually issued must still be replayed with
// its original id.
func TestResponsesReasoningReplayKeepsServerID(t *testing.T) {
	output := mustConvert(t, ProtocolResponses, ProtocolChat, map[string]any{
		"model": "example-model",
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "hi"},
		},
	})
	if _, err := json.Marshal(output); err != nil {
		t.Fatalf("converted request is not encodable: %v", err)
	}

	block := bridgeBlock{Kind: "reasoning", ID: "rs_abc", Text: "because"}
	encoded, ok := encodeResponsesReasoningInput(block)
	if !ok {
		t.Fatal("reasoning block with a server id must be replayed")
	}
	if stringAt(encoded, "id") != "rs_abc" {
		t.Fatalf("reasoning id = %q, want rs_abc", stringAt(encoded, "id"))
	}

	if _, ok := encodeResponsesReasoningInput(bridgeBlock{Kind: "reasoning", Text: "plain"}); ok {
		t.Fatal("reasoning block without an id or encrypted content must be skipped")
	}
}

// Deleting thinking signatures or rewriting redacted_thinking is a repair for
// specific vendors. Applying it to a native Anthropic endpoint discards state
// that endpoint re-validates and shifts the cached prompt prefix.
func TestAnthropicThinkingRepairIsVendorGated(t *testing.T) {
	newInput := func() map[string]any {
		return map[string]any{
			"messages": []any{map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "thinking", "thinking": "", "signature": "sig-real"},
					map[string]any{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": map[string]any{}},
				},
			}},
		}
	}

	native := newInput()
	if normalizeToolReasoningHistory(ProtocolAnthropic, "claude-sonnet-5", "https://opencode.ai/zen", native) {
		t.Fatal("native Anthropic model must not have its thinking history rewritten")
	}
	thinking := firstContentBlock(t, native)
	if stringAt(thinking, "signature") != "sig-real" {
		t.Fatalf("signature was dropped for a native Anthropic model: %v", thinking)
	}
	if stringAt(thinking, "thinking") != "" {
		t.Fatalf("empty thinking text was overwritten for a native Anthropic model: %v", thinking)
	}

	vendor := newInput()
	if !normalizeToolReasoningHistory(ProtocolAnthropic, "deepseek-chat", "https://opencode.ai/zen", vendor) {
		t.Fatal("vendor model should still receive the thinking repair")
	}
	vendorThinking := firstContentBlock(t, vendor)
	if _, exists := vendorThinking["signature"]; exists {
		t.Fatalf("vendor repair must remove the signature: %v", vendorThinking)
	}
	if stringAt(vendorThinking, "thinking") != toolReasoningPlaceholder {
		t.Fatalf("vendor repair must fill empty thinking text: %v", vendorThinking)
	}
}
