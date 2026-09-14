package main

import (
	"errors"
	"fmt"
	"sort"
	"strings"
)

// prepareUpstreamRequest is the single request preparation path for both
// pass-through and transcoded requests. Same-protocol requests are cloned so
// provider-specific fields survive, while cross-protocol requests go through
// the bridge. Target-protocol normalization then repairs reasoning history in
// either case.
func prepareUpstreamRequest(from, to Protocol, input map[string]any, upstreamURL string) (map[string]any, error) {
	output, err := convertRequest(from, to, input)
	if err != nil {
		return nil, err
	}
	normalizeToolReasoningHistory(to, stringAt(output, "model"), upstreamURL, output)
	return output, nil
}

// normalizeToolReasoningHistory applies only to endpoints that are known to
// require reasoning replay, or to requests that explicitly enable reasoning.
// Normalizing the target shape makes the behavior independent of the client
// protocol used to reach the gateway.
func normalizeToolReasoningHistory(protocol Protocol, model, upstreamURL string, input map[string]any) bool {
	vendor := isReasoningVendorIdentifier(model) || isReasoningVendorIdentifier(upstreamURL)
	widened := vendor || requestEnablesReasoning(input)
	switch protocol {
	case ProtocolChat:
		// The Chat repair is additive: it only supplies reasoning_content that
		// a compatible endpoint ignores when it does not understand the field.
		// The widened gate is therefore safe here.
		if !widened {
			return false
		}
		return normalizeChatToolReasoningHistory(input)
	case ProtocolAnthropic:
		// Every repair below is destructive for a native Anthropic endpoint:
		// it deletes thinking signatures, discards redacted_thinking payloads
		// and overwrites empty thinking text. Those are precisely the shapes a
		// native endpoint re-validates and replays verbatim, and rewriting them
		// also shifts the cached prompt prefix mid-history. Only endpoints
		// identified as reasoning vendors get the rewrite.
		if !vendor {
			return false
		}
		return normalizeAnthropicToolThinkingHistory(input)
	default:
		return false
	}
}

func isReasoningVendorIdentifier(value string) bool {
	value = strings.ToLower(value)
	for _, hint := range reasoningVendorHints {
		if strings.Contains(value, hint) {
			return true
		}
	}
	return false
}

func requestEnablesReasoning(input map[string]any) bool {
	for _, key := range []string{"reasoning_effort", "reasoning", "thinking", "effort"} {
		value, exists := input[key]
		if !exists || value == nil {
			continue
		}
		switch typed := value.(type) {
		case string:
			mode := strings.ToLower(strings.TrimSpace(typed))
			if mode != "" && mode != "none" && mode != "disabled" {
				return true
			}
		case bool:
			if typed {
				return true
			}
		case map[string]any:
			mode := strings.ToLower(strings.TrimSpace(firstString(stringAt(typed, "type"), stringAt(typed, "effort"))))
			if mode == "none" || mode == "disabled" {
				continue
			}
			return true
		default:
			return true
		}
	}
	return false
}

// normalizeChatToolReasoningHistory ensures every assistant tool-call turn
// carries reasoning_content. Some clients discard this non-standard field
// while retaining tool_calls, which otherwise makes the next thinking-mode
// request invalid. A legacy reasoning string is promoted when available.
func normalizeChatToolReasoningHistory(input map[string]any) bool {
	messages, ok := input["messages"].([]any)
	if !ok {
		return false
	}

	changed := false
	for _, rawMessage := range messages {
		message, ok := rawMessage.(map[string]any)
		if !ok || stringAt(message, "role") != "assistant" || len(sliceAt(message, "tool_calls")) == 0 {
			continue
		}
		if reasoning, ok := message["reasoning_content"].(string); ok && strings.TrimSpace(reasoning) != "" {
			continue
		}
		reasoning, _ := message["reasoning"].(string)
		if strings.TrimSpace(reasoning) == "" {
			reasoning = toolReasoningPlaceholder
		}
		message["reasoning_content"] = reasoning
		changed = true
	}
	return changed
}

// normalizeAnthropicToolThinkingHistory repairs only assistant turns that
// contain tool_use. DeepSeek, Kimi/Moonshot and MiMo reject signed, redacted,
// empty or missing thinking history on those turns even though Anthropic
// clients can legitimately send each of those shapes.
func normalizeAnthropicToolThinkingHistory(input map[string]any) bool {
	messages, ok := input["messages"].([]any)
	if !ok {
		return false
	}

	changed := false
	for _, rawMessage := range messages {
		message, ok := rawMessage.(map[string]any)
		if !ok || stringAt(message, "role") != "assistant" {
			continue
		}
		content, ok := message["content"].([]any)
		if !ok || !anthropicContentHasType(content, "tool_use") {
			continue
		}

		hasThinking := false
		for i, rawBlock := range content {
			block, ok := rawBlock.(map[string]any)
			if !ok {
				continue
			}
			switch stringAt(block, "type") {
			case "thinking":
				hasThinking = true
				if _, exists := block["signature"]; exists {
					delete(block, "signature")
					changed = true
				}
				thinking, ok := block["thinking"].(string)
				if !ok || strings.TrimSpace(thinking) == "" {
					block["thinking"] = toolReasoningPlaceholder
					changed = true
				}
			case "redacted_thinking":
				hasThinking = true
				content[i] = map[string]any{
					"type":     "thinking",
					"thinking": anthropicRedactedThinkingPlaceholder,
				}
				changed = true
			}
		}
		if !hasThinking {
			content = append([]any{map[string]any{
				"type":     "thinking",
				"thinking": toolReasoningPlaceholder,
			}}, content...)
			message["content"] = content
			changed = true
		}
	}
	return changed
}

func anthropicContentHasType(content []any, kind string) bool {
	for _, rawBlock := range content {
		block, _ := rawBlock.(map[string]any)
		if stringAt(block, "type") == kind {
			return true
		}
	}
	return false
}

func decodeBridgeRequest(protocol Protocol, input map[string]any) (bridgeRequest, error) {
	request := bridgeRequest{
		Model:       stringAt(input, "model"),
		Stream:      boolAt(input, "stream"),
		Temperature: input["temperature"],
		TopP:        input["top_p"],
		Metadata:    input["metadata"],
	}
	switch protocol {
	case ProtocolChat:
		request.MaxTokens = firstAny(input["max_completion_tokens"], input["max_tokens"])
		request.Stop = input["stop"]
		request.Reasoning = firstAny(input["reasoning_effort"], input["reasoning"])
		for i, raw := range sliceAt(input, "messages") {
			message, ok := raw.(map[string]any)
			if !ok {
				return request, fmt.Errorf("messages[%d] must be an object", i)
			}
			role := stringAt(message, "role")
			blocks, err := decodeOpenAIBlocksChecked(message["content"])
			if err != nil {
				return request, fmt.Errorf("messages[%d]: %w", i, err)
			}
			if role == "assistant" {
				blocks = append(decodeChatReasoning(message), blocks...)
			}
			for j, rawCall := range sliceAt(message, "tool_calls") {
				call, ok := rawCall.(map[string]any)
				if !ok {
					return request, fmt.Errorf("messages[%d].tool_calls[%d] must be an object", i, j)
				}
				function := mapAt(call, "function")
				blocks = append(blocks, bridgeBlock{
					Kind:          "tool_call",
					ID:            stringAt(call, "id"),
					Name:          stringAt(function, "name"),
					ArgumentsJSON: stringAt(function, "arguments"),
				})
			}
			switch role {
			case "system":
				request.System = append(request.System, blocks...)
			case "developer":
				request.Developer = append(request.Developer, blocks...)
			case "tool":
				request.Messages = append(request.Messages, bridgeMessage{Role: "user", Blocks: []bridgeBlock{{
					Kind:   "tool_result",
					CallID: stringAt(message, "tool_call_id"),
					Result: message["content"],
				}}})
			case "user", "assistant":
				request.Messages = append(request.Messages, bridgeMessage{Role: role, Blocks: blocks})
			default:
				return request, fmt.Errorf("messages[%d] has unsupported role %q", i, role)
			}
		}
		for i, raw := range sliceAt(input, "tools") {
			tool, ok := raw.(map[string]any)
			if !ok || stringAt(tool, "type") != "function" {
				return request, fmt.Errorf("tools[%d] must be a function tool", i)
			}
			function := mapAt(tool, "function")
			request.Tools = append(request.Tools, bridgeTool{
				Name:        stringAt(function, "name"),
				Description: stringAt(function, "description"),
				Schema:      function["parameters"],
				Strict:      boolAt(function, "strict"),
			})
		}
		request.ToolChoice = decodeChatToolChoice(input["tool_choice"])
		request.ResponseFormat = input["response_format"]
		request.ParallelToolCalls = input["parallel_tool_calls"]
		request.FrequencyPenalty = input["frequency_penalty"]
		request.PresencePenalty = input["presence_penalty"]
		request.Seed = input["seed"]

	case ProtocolResponses:
		request.MaxTokens = input["max_output_tokens"]
		request.Stop = input["stop"]
		request.Reasoning = firstAny(input["reasoning"], input["reasoning_effort"])
		instructions, err := decodeOpenAIBlocksChecked(input["instructions"])
		if err != nil {
			return request, fmt.Errorf("instructions: %w", err)
		}
		request.System = append(request.System, instructions...)
		switch value := input["input"].(type) {
		case string:
			request.Messages = append(request.Messages, bridgeMessage{Role: "user", Blocks: []bridgeBlock{{Kind: "text", Text: value}}})
		case []any:
			for i, raw := range value {
				item, ok := raw.(map[string]any)
				if !ok {
					return request, fmt.Errorf("input[%d] must be an object", i)
				}
				switch stringAt(item, "type") {
				case "reasoning":
					for _, block := range decodeResponsesReasoning(item) {
						appendBridgeBlock(&request.Messages, "assistant", block)
					}
				case "function_call":
					appendBridgeBlock(&request.Messages, "assistant", bridgeBlock{
						Kind:          "tool_call",
						ID:            firstString(stringAt(item, "call_id"), stringAt(item, "id")),
						Name:          stringAt(item, "name"),
						ArgumentsJSON: stringAt(item, "arguments"),
					})
				case "function_call_output":
					appendBridgeBlock(&request.Messages, "user", bridgeBlock{
						Kind:   "tool_result",
						CallID: stringAt(item, "call_id"),
						Result: item["output"],
					})
				case "message", "":
					role := stringAt(item, "role")
					blocks, err := decodeOpenAIBlocksChecked(item["content"])
					if err != nil {
						return request, fmt.Errorf("input[%d]: %w", i, err)
					}
					if role == "system" {
						request.System = append(request.System, blocks...)
					} else if role == "developer" {
						request.Developer = append(request.Developer, blocks...)
					} else if role == "user" || role == "assistant" {
						request.Messages = append(request.Messages, bridgeMessage{Role: role, Blocks: blocks})
					}
				default:
					return request, fmt.Errorf("input[%d] has unsupported Responses item type %q", i, stringAt(item, "type"))
				}
			}
		default:
			return request, fmt.Errorf("input must be a string or array")
		}
		for i, raw := range sliceAt(input, "tools") {
			tool, ok := raw.(map[string]any)
			if !ok || stringAt(tool, "type") != "function" {
				return request, fmt.Errorf("tools[%d] must be a function tool", i)
			}
			request.Tools = append(request.Tools, bridgeTool{
				Name:        stringAt(tool, "name"),
				Description: stringAt(tool, "description"),
				Schema:      tool["parameters"],
				Strict:      boolAt(tool, "strict"),
			})
		}
		request.ToolChoice = decodeResponsesToolChoice(input["tool_choice"])
		request.ResponseFormat = mapAt(input, "text", "format")
		request.ParallelToolCalls = input["parallel_tool_calls"]

	case ProtocolAnthropic:
		request.MaxTokens = input["max_tokens"]
		request.Stop = input["stop_sequences"]
		request.Reasoning = firstAny(input["thinking"], anyAt(input, "output_config", "effort"), input["effort"])
		blocks, err := decodeAnthropicBlocksChecked(input["system"])
		if err != nil {
			return request, fmt.Errorf("system: %w", err)
		}
		request.System = blocks
		for i, raw := range sliceAt(input, "messages") {
			message, ok := raw.(map[string]any)
			if !ok {
				return request, fmt.Errorf("messages[%d] must be an object", i)
			}
			role := stringAt(message, "role")
			// Some clients (e.g. Claude Code) inline system/developer messages
			// into the messages array instead of using the top-level system
			// field. Fold them into the system prompt rather than rejecting.
			if role == "system" {
				blocks, err := decodeAnthropicBlocksChecked(message["content"])
				if err != nil {
					return request, fmt.Errorf("messages[%d]: %w", i, err)
				}
				request.System = append(request.System, blocks...)
				continue
			}
			if role == "developer" {
				blocks, err := decodeAnthropicBlocksChecked(message["content"])
				if err != nil {
					return request, fmt.Errorf("messages[%d]: %w", i, err)
				}
				request.Developer = append(request.Developer, blocks...)
				continue
			}
			if role != "user" && role != "assistant" {
				return request, fmt.Errorf("messages[%d] has unsupported role %q", i, role)
			}
			blocks, err := decodeAnthropicBlocksChecked(message["content"])
			if err != nil {
				return request, fmt.Errorf("messages[%d]: %w", i, err)
			}
			request.Messages = append(request.Messages, bridgeMessage{Role: role, Blocks: blocks})
		}
		for i, raw := range sliceAt(input, "tools") {
			tool, ok := raw.(map[string]any)
			if !ok {
				return request, fmt.Errorf("tools[%d] must be an object", i)
			}
			request.Tools = append(request.Tools, bridgeTool{
				Name:        stringAt(tool, "name"),
				Description: stringAt(tool, "description"),
				Schema:      tool["input_schema"],
			})
		}
		request.ToolChoice = decodeAnthropicToolChoice(input["tool_choice"])

	default:
		return request, fmt.Errorf("unsupported input protocol %q", protocol)
	}
	return request, nil
}

func appendBridgeBlock(messages *[]bridgeMessage, role string, block bridgeBlock) {
	if len(*messages) > 0 {
		last := &(*messages)[len(*messages)-1]
		if last.Role == role && bridgeBlocksOnlyTools(last.Blocks) {
			last.Blocks = append(last.Blocks, block)
			return
		}
	}
	*messages = append(*messages, bridgeMessage{Role: role, Blocks: []bridgeBlock{block}})
}

func bridgeBlocksOnlyTools(blocks []bridgeBlock) bool {
	if len(blocks) == 0 {
		return false
	}
	for _, block := range blocks {
		if block.Kind != "tool_call" && block.Kind != "tool_result" {
			return false
		}
	}
	return true
}

func encodeBridgeRequest(protocol Protocol, request bridgeRequest) (map[string]any, error) {
	if err := validateBridgeRequest(protocol, request); err != nil {
		return nil, err
	}
	switch protocol {
	case ProtocolChat:
		return encodeChatRequest(request)
	case ProtocolResponses:
		return encodeResponsesRequest(request), nil
	case ProtocolAnthropic:
		return encodeAnthropicRequest(request), nil
	default:
		return nil, fmt.Errorf("unsupported output protocol %q", protocol)
	}
}

func validateBridgeRequest(protocol Protocol, request bridgeRequest) error {
	if protocol == ProtocolResponses {
		for _, block := range request.System {
			if block.Kind == "file" {
				return errors.New("file content is not valid inside Responses instructions; send it as a message input item")
			}
		}
	}
	blocks := make([]bridgeBlock, 0, len(request.System)+len(request.Developer))
	blocks = append(blocks, request.System...)
	blocks = append(blocks, request.Developer...)
	for _, message := range request.Messages {
		blocks = append(blocks, message.Blocks...)
	}
	for _, block := range blocks {
		if block.Kind == "file" && protocol == ProtocolAnthropic && block.FileID != "" && block.Data == "" && block.URL == "" {
			return fmt.Errorf("file %q cannot be represented by Anthropic Messages without file data or a URL", block.FileID)
		}
	}
	return nil
}

func encodeChatRequest(request bridgeRequest) (map[string]any, error) {
	output := map[string]any{"model": request.Model, "stream": request.Stream}
	put(output, "temperature", request.Temperature)
	put(output, "top_p", request.TopP)
	put(output, "max_tokens", request.MaxTokens)
	put(output, "stop", request.Stop)
	if effort := reasoningEffort(request.Reasoning); effort != nil {
		output["reasoning_effort"] = effort
	}
	put(output, "response_format", chatResponseFormat(request.ResponseFormat))
	put(output, "parallel_tool_calls", request.ParallelToolCalls)
	put(output, "frequency_penalty", request.FrequencyPenalty)
	put(output, "presence_penalty", request.PresencePenalty)
	put(output, "seed", request.Seed)
	if request.Stream {
		output["stream_options"] = map[string]any{"include_usage": true}
	}

	messages := make([]any, 0, len(request.Messages)+1)
	if len(request.System) > 0 {
		messages = append(messages, map[string]any{"role": "system", "content": encodeChatBlocks(request.System)})
	}
	if len(request.Developer) > 0 {
		messages = append(messages, map[string]any{"role": "developer", "content": encodeChatBlocks(request.Developer)})
	}
	pending := make(map[string]bool)
	pendingResults := make(map[string]bridgeBlock)
	var pendingOrder []string
	var deferred [][]bridgeBlock
	flushDeferred := func() {
		for _, blocks := range deferred {
			if len(blocks) > 0 {
				messages = append(messages, map[string]any{"role": "user", "content": encodeChatBlocks(blocks)})
			}
		}
		deferred = nil
	}

	for i, message := range request.Messages {
		var content, reasoning, calls, results []bridgeBlock
		for _, block := range message.Blocks {
			switch block.Kind {
			case "reasoning":
				reasoning = append(reasoning, block)
			case "tool_call":
				calls = append(calls, block)
			case "tool_result":
				results = append(results, block)
			default:
				content = append(content, block)
			}
		}
		if message.Role == "assistant" {
			if len(pending) > 0 {
				return nil, fmt.Errorf("messages[%d]: assistant message appears before tool results for %s", i, strings.Join(missingToolIDs(pendingOrder, pending), ", "))
			}
			if len(results) > 0 {
				return nil, fmt.Errorf("messages[%d]: assistant message contains tool results", i)
			}
			encoded := map[string]any{"role": "assistant", "content": nil}
			if value, ok := bridgeReasoningText(reasoning); ok {
				encoded["reasoning_content"] = value
			}
			if len(content) > 0 {
				encoded["content"] = encodeChatBlocks(content)
			}
			if len(calls) > 0 {
				toolCalls := make([]any, 0, len(calls))
				for _, call := range calls {
					if call.ID == "" || call.Name == "" {
						return nil, fmt.Errorf("messages[%d]: tool call must contain id and name", i)
					}
					if pending[call.ID] {
						return nil, fmt.Errorf("messages[%d]: duplicate tool call id %q", i, call.ID)
					}
					pending[call.ID] = true
					pendingOrder = append(pendingOrder, call.ID)
					toolCalls = append(toolCalls, map[string]any{
						"id":   call.ID,
						"type": "function",
						"function": map[string]any{
							"name":      call.Name,
							"arguments": bridgeArgumentsJSON(call),
						},
					})
				}
				encoded["tool_calls"] = toolCalls
			}
			if len(content) > 0 || len(reasoning) > 0 || len(calls) > 0 {
				messages = append(messages, encoded)
			}
			continue
		}

		if len(calls) > 0 {
			return nil, fmt.Errorf("messages[%d]: user message contains tool calls", i)
		}
		for _, result := range results {
			if _, duplicate := pendingResults[result.CallID]; duplicate {
				return nil, fmt.Errorf("messages[%d]: duplicate tool result %q", i, result.CallID)
			}
			if !pending[result.CallID] {
				return nil, fmt.Errorf("messages[%d]: tool result %q has no pending tool call", i, result.CallID)
			}
			pendingResults[result.CallID] = result
			delete(pending, result.CallID)
		}
		if len(content) > 0 {
			if len(pendingOrder) > 0 || len(deferred) > 0 {
				deferred = append(deferred, content)
			} else {
				messages = append(messages, map[string]any{"role": "user", "content": encodeChatBlocks(content)})
			}
		}
		if len(pending) == 0 && len(pendingOrder) > 0 {
			for _, id := range pendingOrder {
				result := pendingResults[id]
				messages = append(messages, map[string]any{
					"role":         "tool",
					"tool_call_id": id,
					"content":      bridgeToolResultContent(result.Result),
				})
			}
			flushDeferred()
			pendingOrder = nil
			clear(pendingResults)
		}
	}
	if len(pending) > 0 {
		return nil, fmt.Errorf("tool calls are missing results for %s", strings.Join(missingToolIDs(pendingOrder, pending), ", "))
	}
	flushDeferred()
	output["messages"] = messages

	if len(request.Tools) > 0 {
		tools := make([]any, 0, len(request.Tools))
		for _, tool := range request.Tools {
			function := map[string]any{
				"name":        tool.Name,
				"description": tool.Description,
				"parameters":  schemaOrDefault(tool.Schema),
			}
			if tool.Strict {
				function["strict"] = true
			}
			tools = append(tools, map[string]any{"type": "function", "function": function})
		}
		output["tools"] = tools
		if choice := encodeChatToolChoice(request.ToolChoice); choice != nil {
			output["tool_choice"] = choice
		}
	}
	return output, nil
}

func missingToolIDs(order []string, pending map[string]bool) []string {
	missing := make([]string, 0, len(pending))
	for _, id := range order {
		if pending[id] {
			missing = append(missing, id)
		}
	}
	if len(missing) == len(pending) {
		return missing
	}
	for id := range pending {
		found := false
		for _, current := range missing {
			if current == id {
				found = true
				break
			}
		}
		if !found {
			missing = append(missing, id)
		}
	}
	sort.Strings(missing)
	return missing
}

func encodeResponsesRequest(request bridgeRequest) map[string]any {
	output := map[string]any{"model": request.Model, "stream": request.Stream}
	put(output, "temperature", request.Temperature)
	put(output, "top_p", request.TopP)
	put(output, "max_output_tokens", request.MaxTokens)
	put(output, "stop", request.Stop)
	put(output, "metadata", request.Metadata)
	if len(request.System) > 0 {
		output["instructions"] = bridgeBlocksText(request.System)
	}
	if request.Reasoning != nil {
		switch value := request.Reasoning.(type) {
		case string:
			output["reasoning"] = map[string]any{"effort": value}
		default:
			output["reasoning"] = value
		}
	}
	if format := responsesTextFormat(request.ResponseFormat); format != nil {
		output["text"] = map[string]any{"format": format}
	}
	put(output, "parallel_tool_calls", request.ParallelToolCalls)

	items := make([]any, 0, len(request.Messages)+1)
	if len(request.Developer) > 0 {
		items = append(items, map[string]any{
			"type": "message", "role": "developer",
			"content": []any{map[string]any{"type": "input_text", "text": bridgeBlocksText(request.Developer)}},
		})
	}
	for _, message := range request.Messages {
		var content []any
		flushContent := func() {
			if len(content) == 0 {
				return
			}
			items = append(items, map[string]any{
				"type":    "message",
				"role":    normalizeResponsesRole(message.Role),
				"content": content,
			})
			content = nil
		}
		for _, block := range message.Blocks {
			switch block.Kind {
			case "reasoning":
				flushContent()
				// Only replay reasoning the upstream actually issued. A block
				// that arrived as plain text (Chat/Anthropic history) has no
				// server-side id, and synthesizing one guarantees a stale
				// reasoning-reference rejection, which previously forced every
				// such turn through the strip-and-retry path.
				if item, ok := encodeResponsesReasoningInput(block); ok {
					items = append(items, item)
				}
			case "text":
				kind := "input_text"
				if message.Role == "assistant" {
					kind = "output_text"
				}
				content = append(content, map[string]any{"type": kind, "text": block.Text})
			case "image":
				url := block.URL
				if url == "" && block.Data != "" {
					url = "data:" + block.MediaType + ";base64," + block.Data
				}
				content = append(content, map[string]any{"type": "input_image", "image_url": url})
			case "file":
				item := map[string]any{"type": "input_file"}
				put(item, "file_id", block.FileID)
				put(item, "file_data", block.Data)
				put(item, "filename", block.Filename)
				content = append(content, item)
			case "tool_call":
				flushContent()
				items = append(items, map[string]any{
					"type":      "function_call",
					"call_id":   block.ID,
					"name":      block.Name,
					"arguments": bridgeArgumentsJSON(block),
				})
			case "tool_result":
				flushContent()
				items = append(items, map[string]any{
					"type":    "function_call_output",
					"call_id": block.CallID,
					"output":  bridgeToolResultContent(block.Result),
				})
			}
		}
		flushContent()
	}
	output["input"] = items
	if len(request.Tools) > 0 {
		tools := make([]any, 0, len(request.Tools))
		for _, tool := range request.Tools {
			tools = append(tools, map[string]any{
				"type":        "function",
				"name":        tool.Name,
				"description": tool.Description,
				"parameters":  schemaOrDefault(tool.Schema),
				"strict":      tool.Strict,
			})
		}
		output["tools"] = tools
		if choice := encodeResponsesToolChoice(request.ToolChoice); choice != nil {
			output["tool_choice"] = choice
		}
	}
	return output
}

func encodeAnthropicRequest(request bridgeRequest) map[string]any {
	output := map[string]any{"model": request.Model, "stream": request.Stream}
	put(output, "temperature", request.Temperature)
	put(output, "top_p", request.TopP)
	put(output, "stop_sequences", request.Stop)
	if request.MaxTokens == nil {
		output["max_tokens"] = 4096
	} else {
		output["max_tokens"] = request.MaxTokens
	}
	if thinking, outputConfig := anthropicThinking(request.Reasoning); thinking != nil {
		output["thinking"] = thinking
		if outputConfig != nil {
			output["output_config"] = outputConfig
		}
	}
	if len(request.System) > 0 || len(request.Developer) > 0 {
		instructions := append([]bridgeBlock(nil), request.System...)
		instructions = append(instructions, request.Developer...)
		output["system"] = encodeAnthropicBlocks(instructions)
	}

	messages := make([]any, 0, len(request.Messages))
	for _, message := range request.Messages {
		content := encodeAnthropicBlocks(message.Blocks)
		if len(content) == 0 {
			continue
		}
		role := "user"
		if message.Role == "assistant" {
			role = "assistant"
		}
		if len(messages) > 0 {
			last, _ := messages[len(messages)-1].(map[string]any)
			if stringAt(last, "role") == role {
				last["content"] = append(asSlice(last["content"]), content...)
				continue
			}
		}
		messages = append(messages, map[string]any{"role": role, "content": content})
	}
	output["messages"] = messages
	// Anthropic has no tool_choice "none" value. Omitting the tools is the
	// protocol-compatible representation of an explicitly disabled tool set.
	if len(request.Tools) > 0 && request.ToolChoice.Mode != "none" {
		tools := make([]any, 0, len(request.Tools))
		for _, tool := range request.Tools {
			tools = append(tools, map[string]any{
				"name":         tool.Name,
				"description":  tool.Description,
				"input_schema": schemaOrDefault(tool.Schema),
			})
		}
		output["tools"] = tools
		if choice := encodeAnthropicToolChoice(request.ToolChoice); choice != nil {
			output["tool_choice"] = choice
		}
	}
	return output
}
