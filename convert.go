package main

import (
	"encoding/json"
	"strings"
)

const (
	toolReasoningPlaceholder             = "tool call"
	anthropicRedactedThinkingPlaceholder = "[redacted thinking]"
)

var reasoningVendorHints = [...]string{"moonshot", "kimi", "deepseek", "mimo", "xiaomimimo"}

type bridgeBlock struct {
	Kind          string
	Text          string
	Signature     string
	Encrypted     string
	URL           string
	MediaType     string
	Data          string
	FileID        string
	Filename      string
	ID            string
	Name          string
	Arguments     any
	ArgumentsJSON string
	CallID        string
	Result        any
	IsError       bool
}

type bridgeMessage struct {
	Role   string
	Blocks []bridgeBlock
}

type bridgeTool struct {
	Name        string
	Description string
	Schema      any
	Strict      bool
}

type bridgeToolChoice struct {
	Mode string
	Name string
}

type bridgeRequest struct {
	Model       string
	System      []bridgeBlock
	Developer   []bridgeBlock
	Messages    []bridgeMessage
	Tools       []bridgeTool
	ToolChoice  bridgeToolChoice
	Stream      bool
	Temperature any
	TopP        any
	MaxTokens   any
	Stop        any
	Reasoning   any
	Metadata    any
	// Sampling and response-shape knobs are carried across the bridge as
	// opaque values. Each protocol accepts a different shape, so a value that
	// cannot be expressed natively is omitted rather than rejected: dropping a
	// knob silently was the old behavior and it made clients believe (for
	// example) that JSON mode had been honored.
	ResponseFormat    any
	ParallelToolCalls any
	FrequencyPenalty  any
	PresencePenalty   any
	Seed              any
}

type bridgeUsage struct {
	// Input is the total prompt input, including cache reads and writes. This
	// matches OpenAI's prompt_tokens semantics and lets each output encoder
	// split the provider-specific cache fields exactly once.
	Input         int
	Output        int
	Total         int
	Cached        int
	CacheCreation int
	Reasoning     int
}

type bridgeResponse struct {
	ID        string
	Model     string
	Text      string
	Reasoning []bridgeBlock
	Tools     []bridgeBlock
	Stop      string
	Error     string
	Usage     bridgeUsage
	Created   int64
}

func convertRequest(from, to Protocol, input map[string]any) (map[string]any, error) {
	if from == to {
		return cloneMap(input), nil
	}
	request, err := decodeBridgeRequest(from, input)
	if err != nil {
		return nil, err
	}
	return encodeBridgeRequest(to, request)
}

func firstNonZero(values ...int) int {
	for _, value := range values {
		if value != 0 {
			return value
		}
	}
	return 0
}

func normalizeResponsesRole(role string) string {
	if role == "assistant" {
		return "assistant"
	}
	return "user"
}

func schemaOrDefault(value any) any {
	if value == nil {
		return map[string]any{"type": "object", "properties": map[string]any{}}
	}
	return value
}

// chatResponseFormat converts a bridge-level response format into the OpenAI
// Chat Completions shape. Chat and Responses agree on the format type names but
// not on the json_schema nesting: Responses carries name/schema/strict inline,
// while Chat nests them under "json_schema". A value that has no Chat
// equivalent (the "text" type) is omitted instead of failing the request.
func chatResponseFormat(format any) any {
	object, ok := format.(map[string]any)
	if !ok {
		return nil
	}
	switch stringAt(object, "type") {
	case "json_object":
		return map[string]any{"type": "json_object"}
	case "json_schema":
		if nested, ok := object["json_schema"].(map[string]any); ok {
			return map[string]any{"type": "json_schema", "json_schema": nested}
		}
		nested := make(map[string]any, 4)
		for _, key := range []string{"name", "description", "schema", "strict"} {
			if value, exists := object[key]; exists {
				nested[key] = value
			}
		}
		if len(nested) == 0 {
			return nil
		}
		return map[string]any{"type": "json_schema", "json_schema": nested}
	default:
		return nil
	}
}

// responsesTextFormat converts a bridge-level response format into the object
// accepted by the Responses "text.format" field. A Chat-style nested
// json_schema is flattened into the inline shape Responses expects.
func responsesTextFormat(format any) any {
	object, ok := format.(map[string]any)
	if !ok {
		return nil
	}
	switch stringAt(object, "type") {
	case "text", "json_object":
		return object
	case "json_schema":
		if nested, ok := object["json_schema"].(map[string]any); ok {
			flattened := make(map[string]any, len(nested)+1)
			for key, value := range nested {
				flattened[key] = value
			}
			flattened["type"] = "json_schema"
			return flattened
		}
		return object
	default:
		return nil
	}
}

func reasoningEffort(value any) any {
	if object, ok := value.(map[string]any); ok {
		if effort := object["effort"]; effort != nil {
			return effort
		}
		switch stringAt(object, "type") {
		case "disabled":
			return nil
		case "enabled", "adaptive":
			return effortForThinkingBudget(intAt(object, "budget_tokens"))
		default:
			return object["type"]
		}
	}
	return value
}

func anthropicThinking(value any) (any, map[string]any) {
	if value == nil {
		return nil, nil
	}
	if object, ok := value.(map[string]any); ok {
		kind := stringAt(object, "type")
		if kind == "enabled" || kind == "adaptive" || kind == "disabled" {
			return value, nil
		}
		value = firstAny(object["effort"], object["type"])
	}
	effort, _ := value.(string)
	effort = strings.ToLower(strings.TrimSpace(effort))
	if effort == "" || effort == "none" || effort == "disabled" {
		return nil, nil
	}
	budget := 4096
	switch effort {
	case "minimal", "low":
		budget = 1024
	case "medium":
		budget = 4096
	case "high", "xhigh", "max":
		budget = 8192
	}
	return map[string]any{"type": "enabled", "budget_tokens": budget}, map[string]any{"effort": effort}
}

func effortForThinkingBudget(budget int) string {
	if budget <= 0 || budget >= 8192 {
		return "high"
	}
	if budget <= 2048 {
		return "low"
	}
	return "medium"
}

func bridgeBlocksText(blocks []bridgeBlock) string {
	var text strings.Builder
	for _, block := range blocks {
		if block.Kind == "text" {
			text.WriteString(block.Text)
		}
	}
	return text.String()
}

func parseJSONOrString(value string) any {
	if value == "" {
		return map[string]any{}
	}
	var decoded any
	if json.Unmarshal([]byte(value), &decoded) == nil {
		return decoded
	}
	return value
}

func asPrefix(id, prefix string) string {
	if strings.HasPrefix(id, prefix+"_") {
		return id
	}
	for _, current := range []string{"chatcmpl_", "resp_", "msg_"} {
		id = strings.TrimPrefix(id, current)
	}
	return prefix + "_" + id
}

func put(object map[string]any, key string, value any) {
	if value != nil {
		object[key] = value
	}
}

func firstAny(values ...any) any {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func cloneMap(input map[string]any) map[string]any {
	data, _ := json.Marshal(input)
	var output map[string]any
	_ = json.Unmarshal(data, &output)
	return output
}

func asSlice(value any) []any {
	if values, ok := value.([]any); ok {
		return values
	}
	if value == nil {
		return nil
	}
	return []any{value}
}

func sliceAt(object map[string]any, path ...string) []any {
	values, _ := anyAt(object, path...).([]any)
	return values
}

func mapAt(object map[string]any, path ...string) map[string]any {
	value, _ := anyAt(object, path...).(map[string]any)
	return value
}

func stringAt(object map[string]any, path ...string) string {
	value, _ := anyAt(object, path...).(string)
	return value
}

func boolAt(object map[string]any, path ...string) bool {
	value, _ := anyAt(object, path...).(bool)
	return value
}

func intAt(object map[string]any, path ...string) int {
	value := anyAt(object, path...)
	switch number := value.(type) {
	case float64:
		return int(number)
	case int:
		return number
	case json.Number:
		integer, _ := number.Int64()
		return int(integer)
	default:
		return 0
	}
}

func int64At(object map[string]any, path ...string) int64 {
	return int64(intAt(object, path...))
}

func anyAt(object map[string]any, path ...string) any {
	var current any = object
	for _, key := range path {
		next, ok := current.(map[string]any)
		if !ok {
			return nil
		}
		current = next[key]
	}
	return current
}
