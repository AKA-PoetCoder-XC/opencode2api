package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

type bridgeStreamParser struct {
	protocol          Protocol
	started           bool
	tools             map[string]bool
	toolIDs           map[string]string
	toolNames         map[string]string
	toolOrder         []string
	responseArgs      map[string]bool
	responseReasoning map[string]bool
}

func (parser *bridgeStreamParser) Parse(eventName, data string) ([]bridgeStreamEvent, error) {
	if data == "[DONE]" {
		if parser.protocol != ProtocolChat {
			return nil, fmt.Errorf("unexpected [DONE] terminator for %s stream", parser.protocol)
		}
		return []bridgeStreamEvent{{Kind: "done"}}, nil
	}
	var value map[string]any
	if err := json.Unmarshal([]byte(data), &value); err != nil {
		return nil, fmt.Errorf("invalid upstream SSE JSON: %w", err)
	}
	switch parser.protocol {
	case ProtocolChat:
		return parser.parseChatEvent(eventName, value), nil
	case ProtocolAnthropic:
		return parser.parseAnthropicEvent(eventName, value)
	case ProtocolResponses:
		return parser.parseResponses(eventName, value), nil
	default:
		return nil, fmt.Errorf("unsupported stream protocol %q", parser.protocol)
	}
}

func (parser *bridgeStreamParser) parseChat(value map[string]any) []bridgeStreamEvent {
	return parser.parseChatEvent("", value)
}

func (parser *bridgeStreamParser) parseChatEvent(eventName string, value map[string]any) []bridgeStreamEvent {
	events := make([]bridgeStreamEvent, 0, 4)
	if eventName == "error" {
		return []bridgeStreamEvent{{Kind: "error", Error: streamErrorMessage(value, "upstream Chat stream error"), ErrorType: streamErrorType(value)}}
	}
	if rawError, exists := value["error"]; exists && rawError != nil {
		return []bridgeStreamEvent{{Kind: "error", Error: streamErrorMessage(value, "upstream Chat stream error"), ErrorType: streamErrorType(value)}}
	}
	if !parser.started {
		if id := stringAt(value, "id"); id != "" {
			parser.started = true
			events = append(events, bridgeStreamEvent{Kind: "start", ResponseID: id, Model: stringAt(value, "model")})
		}
	}
	if usageMap := mapAt(value, "usage"); len(usageMap) > 0 {
		usage := decodeOpenAIUsage(usageMap)
		events = append(events, bridgeStreamEvent{Kind: "usage", Usage: &usage})
	}
	for _, raw := range sliceAt(value, "choices") {
		choice, _ := raw.(map[string]any)
		delta := mapAt(choice, "delta")
		if reasoning := firstString(stringAt(delta, "reasoning_content"), stringAt(delta, "reasoning")); reasoning != "" {
			events = append(events, bridgeStreamEvent{Kind: "reasoning", Text: reasoning})
		}
		if text := stringAt(delta, "content"); text != "" {
			events = append(events, bridgeStreamEvent{Kind: "text", Text: text})
		}
		for _, rawCall := range sliceAt(delta, "tool_calls") {
			call, _ := rawCall.(map[string]any)
			key := fmt.Sprint(firstAny(call["index"], stringAt(call, "id")))
			function := mapAt(call, "function")
			id := stringAt(call, "id")
			if _, seen := parser.tools[key]; !seen {
				parser.tools[key] = false
				parser.toolOrder = append(parser.toolOrder, key)
			}
			if id != "" {
				parser.toolIDs[key] = id
			}
			if name := stringAt(function, "name"); name != "" {
				parser.toolNames[key] = mergeToolName(parser.toolNames[key], name)
			}
			if arguments := stringAt(function, "arguments"); arguments != "" {
				if !parser.tools[key] && parser.toolNames[key] != "" {
					parser.tools[key] = true
					events = append(events, bridgeStreamEvent{Kind: "tool_start", ToolKey: key, ToolID: parser.toolIDs[key], ToolName: parser.toolNames[key]})
				}
				events = append(events, bridgeStreamEvent{Kind: "tool_delta", ToolKey: key, ToolID: parser.toolIDs[key], ToolName: parser.toolNames[key], Text: arguments})
			}
		}
		if stop := stringAt(choice, "finish_reason"); stop != "" {
			if isStreamErrorFinish(stop) {
				events = append(events, bridgeStreamEvent{Kind: "error", Error: firstString(stringAt(choice, "error", "message"), "upstream Chat stream failed"), ErrorType: "upstream_error"})
				continue
			}
			for _, key := range parser.toolOrder {
				if !parser.tools[key] && parser.toolNames[key] != "" {
					parser.tools[key] = true
					events = append(events, bridgeStreamEvent{Kind: "tool_start", ToolKey: key, ToolID: parser.toolIDs[key], ToolName: parser.toolNames[key]})
				}
			}
			events = append(events, bridgeStreamEvent{Kind: "finish", Stop: stop})
		}
	}
	return events
}

func (parser *bridgeStreamParser) parseAnthropic(value map[string]any) ([]bridgeStreamEvent, error) {
	return parser.parseAnthropicEvent("", value)
}

func (parser *bridgeStreamParser) parseAnthropicEvent(eventName string, value map[string]any) ([]bridgeStreamEvent, error) {
	typeName := stringAt(value, "type")
	if typeName == "" {
		typeName = eventName
	}
	switch typeName {
	case "message_start":
		message := mapAt(value, "message")
		events := []bridgeStreamEvent{{Kind: "start", ResponseID: stringAt(message, "id"), Model: stringAt(message, "model")}}
		if usageMap := mapAt(message, "usage"); len(usageMap) > 0 {
			usage := decodeAnthropicUsage(usageMap)
			events = append(events, bridgeStreamEvent{Kind: "usage", Usage: &usage})
		}
		return events, nil
	case "content_block_start":
		block := mapAt(value, "content_block")
		key := fmt.Sprint(value["index"])
		switch stringAt(block, "type") {
		case "thinking":
			events := make([]bridgeStreamEvent, 0, 2)
			if thinking := stringAt(block, "thinking"); thinking != "" {
				events = append(events, bridgeStreamEvent{Kind: "reasoning", Text: thinking})
			}
			if signature := stringAt(block, "signature"); signature != "" {
				events = append(events, bridgeStreamEvent{Kind: "reasoning_signature", Signature: signature})
			}
			return events, nil
		case "redacted_thinking":
			return []bridgeStreamEvent{{Kind: "reasoning", Encrypted: stringAt(block, "data")}}, nil
		case "tool_use":
			parser.tools[key] = true
			return []bridgeStreamEvent{{Kind: "tool_start", ToolKey: key, ToolID: stringAt(block, "id"), ToolName: stringAt(block, "name")}}, nil
		case "text":
			if text := stringAt(block, "text"); text != "" {
				return []bridgeStreamEvent{{Kind: "text", Text: text}}, nil
			}
		}
	case "content_block_delta":
		delta := mapAt(value, "delta")
		key := fmt.Sprint(value["index"])
		switch stringAt(delta, "type") {
		case "thinking_delta":
			return []bridgeStreamEvent{{Kind: "reasoning", Text: stringAt(delta, "thinking")}}, nil
		case "signature_delta":
			return []bridgeStreamEvent{{Kind: "reasoning_signature", Signature: stringAt(delta, "signature")}}, nil
		case "text_delta":
			return []bridgeStreamEvent{{Kind: "text", Text: stringAt(delta, "text")}}, nil
		case "input_json_delta":
			return []bridgeStreamEvent{{Kind: "tool_delta", ToolKey: key, Text: stringAt(delta, "partial_json")}}, nil
		}
	case "message_delta":
		events := make([]bridgeStreamEvent, 0, 2)
		if usageMap := mapAt(value, "usage"); len(usageMap) > 0 {
			usage := decodeAnthropicUsage(usageMap)
			events = append(events, bridgeStreamEvent{Kind: "usage", Usage: &usage})
		}
		if stop := stringAt(value, "delta", "stop_reason"); stop != "" {
			if isStreamErrorFinish(stop) {
				events = append(events, bridgeStreamEvent{Kind: "error", Error: "upstream Anthropic stream failed", ErrorType: "upstream_error"})
				return events, nil
			}
			events = append(events, bridgeStreamEvent{Kind: "finish", Stop: stop})
		}
		return events, nil
	case "message_stop":
		return []bridgeStreamEvent{{Kind: "done"}}, nil
	case "error":
		message := firstString(stringAt(value, "error", "message"), "upstream Anthropic stream error")
		return []bridgeStreamEvent{{Kind: "error", Error: message, ErrorType: firstString(stringAt(value, "error", "type"), "upstream_error")}}, nil
	}
	return nil, nil
}

func (parser *bridgeStreamParser) parseResponses(eventName string, value map[string]any) []bridgeStreamEvent {
	typeName := stringAt(value, "type")
	if typeName == "" {
		typeName = eventName
	}
	switch typeName {
	case "error":
		return []bridgeStreamEvent{{Kind: "error", Error: streamErrorMessage(value, "upstream Responses stream error"), ErrorType: streamErrorType(value)}}
	case "response.created":
		response := mapAt(value, "response")
		return []bridgeStreamEvent{{Kind: "start", ResponseID: stringAt(response, "id"), Model: stringAt(response, "model")}}
	case "response.output_text.delta":
		return []bridgeStreamEvent{{Kind: "text", Text: stringAt(value, "delta")}}
	case "response.reasoning_summary_text.delta":
		key := responseToolKey(value, nil)
		parser.responseReasoning[key] = true
		return []bridgeStreamEvent{{Kind: "reasoning", Text: stringAt(value, "delta")}}
	case "response.output_item.added":
		item := mapAt(value, "item")
		if stringAt(item, "type") == "function_call" {
			key := responseToolKey(value, item)
			parser.rememberTool(key, stringAt(item, "call_id"), stringAt(item, "name"))
			if parser.toolNames[key] != "" {
				parser.tools[key] = true
				return []bridgeStreamEvent{{Kind: "tool_start", ToolKey: key, ToolID: parser.toolIDs[key], ToolName: parser.toolNames[key]}}
			}
		}
	case "response.function_call_arguments.delta":
		key := responseToolKey(value, nil)
		parser.responseArgs[key] = true
		return []bridgeStreamEvent{{Kind: "tool_delta", ToolKey: key, ToolID: parser.toolIDs[key], ToolName: parser.toolNames[key], Text: stringAt(value, "delta")}}
	case "response.function_call_arguments.done":
		key := responseToolKey(value, nil)
		if !parser.responseArgs[key] {
			parser.responseArgs[key] = true
			return []bridgeStreamEvent{{Kind: "tool_delta", ToolKey: key, ToolID: parser.toolIDs[key], ToolName: parser.toolNames[key], Text: stringAt(value, "arguments")}}
		}
	case "response.output_item.done":
		item := mapAt(value, "item")
		if stringAt(item, "type") == "reasoning" {
			key := responseToolKey(value, item)
			if parser.responseReasoning[key] {
				return nil
			}
			blocks := decodeResponsesReasoning(item)
			if len(blocks) > 0 {
				return []bridgeStreamEvent{{Kind: "reasoning", Text: blocks[0].Text, Encrypted: blocks[0].Encrypted}}
			}
			return nil
		}
		if stringAt(item, "type") == "function_call" {
			key := responseToolKey(value, item)
			parser.rememberTool(key, stringAt(item, "call_id"), stringAt(item, "name"))
			events := make([]bridgeStreamEvent, 0, 2)
			if !parser.tools[key] {
				parser.tools[key] = true
				events = append(events, bridgeStreamEvent{Kind: "tool_start", ToolKey: key, ToolID: parser.toolIDs[key], ToolName: parser.toolNames[key]})
			}
			if !parser.responseArgs[key] {
				if arguments := stringAt(item, "arguments"); arguments != "" {
					events = append(events, bridgeStreamEvent{Kind: "tool_delta", ToolKey: key, Text: arguments})
				}
			}
			return events
		}
	case "response.completed", "response.incomplete", "response.failed", "response.done":
		response := mapAt(value, "response")
		usageMap := mapAt(response, "usage")
		if typeName == "response.failed" || stringAt(response, "status") == "failed" {
			message := firstString(stringAt(response, "error", "message"), stringAt(value, "error", "message"), "upstream Responses request failed")
			events := []bridgeStreamEvent{{Kind: "error", Error: message, ErrorType: firstString(stringAt(response, "error", "code"), "upstream_error")}}
			if len(usageMap) > 0 {
				usage := decodeOpenAIUsage(usageMap)
				events = append([]bridgeStreamEvent{{Kind: "usage", Usage: &usage}}, events...)
			}
			return events
		}
		stop := "stop"
		if typeName == "response.incomplete" {
			stop = canonicalResponsesIncomplete(stringAt(response, "incomplete_details", "reason"))
		} else {
			stop = parser.responsesCompletedStop(response)
		}
		events := make([]bridgeStreamEvent, 0, 3)
		if len(usageMap) > 0 {
			usage := decodeOpenAIUsage(usageMap)
			events = append(events, bridgeStreamEvent{Kind: "usage", Usage: &usage})
		}
		return append(events, bridgeStreamEvent{Kind: "finish", Stop: stop}, bridgeStreamEvent{Kind: "done"})
	}
	return nil
}

// responsesCompletedStop decides how a finished Responses turn is reported
// downstream. The completed event carries no stop reason, so the function_call
// items in its output — or the tool calls observed earlier on the stream — are
// the only signal that the turn ended waiting for tool results. Defaulting to
// "stop" made every tool step terminate the downstream turn instead of chaining
// inside it.
func (parser *bridgeStreamParser) responsesCompletedStop(response map[string]any) string {
	for _, raw := range sliceAt(response, "output") {
		item, _ := raw.(map[string]any)
		if stringAt(item, "type") == "function_call" {
			return "tool_calls"
		}
	}
	if len(parser.toolOrder) > 0 {
		return "tool_calls"
	}
	return "stop"
}

func streamErrorMessage(value map[string]any, fallback string) string {
	if message := stringAt(value, "error", "message"); message != "" {
		return message
	}
	if message := stringAt(value, "message"); message != "" {
		return message
	}
	if message, ok := value["error"].(string); ok && message != "" {
		return message
	}
	return fallback
}

func streamErrorType(value map[string]any) string {
	return firstString(stringAt(value, "error", "type"), stringAt(value, "error", "code"), "upstream_error")
}

func isStreamErrorFinish(stop string) bool {
	switch strings.ToLower(strings.TrimSpace(stop)) {
	case "error", "network_error", "server_error":
		return true
	default:
		return false
	}
}

func (parser *bridgeStreamParser) rememberTool(key, id, name string) {
	if _, seen := parser.tools[key]; !seen {
		parser.tools[key] = false
		parser.toolOrder = append(parser.toolOrder, key)
	}
	if id != "" {
		parser.toolIDs[key] = id
	}
	if name != "" {
		parser.toolNames[key] = name
	}
}

func mergeToolName(current, fragment string) string {
	if current == "" || fragment == current {
		return fragment
	}
	if strings.HasPrefix(fragment, current) {
		return fragment
	}
	return current + fragment
}

func responseToolKey(event map[string]any, item map[string]any) string {
	if item != nil {
		if id := stringAt(item, "id"); id != "" {
			return id
		}
	}
	if id := stringAt(event, "item_id"); id != "" {
		return id
	}
	return fmt.Sprint(event["output_index"])
}
