package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
)

const (
	openCodeCapabilitiesURL = "https://models.opencode.ai/api.json"
	openCodeZenDocsURL      = "https://raw.githubusercontent.com/anomalyco/opencode/dev/packages/web/src/content/docs/zen.mdx"
	openCodeGoDocsURL       = "https://raw.githubusercontent.com/anomalyco/opencode/dev/packages/web/src/content/docs/go.mdx"
)

var protocolDocEndpointPattern = regexp.MustCompile("\\|[^|]+\\|\\s*`?([^|`\\s]+)`?\\s*\\|\\s*`[^`]+/v1/(chat/completions|responses|messages)`")

type protocolCapabilities struct {
	Protocols   map[Tier]map[string]Protocol
	Unsupported map[Tier]map[string]bool
	Metadata    map[Tier]map[string]ModelMetadata
}

type capabilityProvider struct {
	ID     string                     `json:"id"`
	API    string                     `json:"api"`
	NPM    string                     `json:"npm"`
	Models map[string]capabilityModel `json:"models"`
}

type capabilityModel struct {
	ID               string                     `json:"id"`
	Provider         *capabilityModelProvider   `json:"provider"`
	Limit            *capabilityModelLimit      `json:"limit"`
	Reasoning        bool                       `json:"reasoning"`
	ToolCall         bool                       `json:"tool_call"`
	StructuredOutput bool                       `json:"structured_output"`
	Modalities       *capabilityModelModalities `json:"modalities"`
}

type capabilityModelModalities struct {
	Input  []string `json:"input"`
	Output []string `json:"output"`
}

type capabilityModelLimit struct {
	Context int `json:"context"`
	Input   int `json:"input"`
	Output  int `json:"output"`
}

// ModelMetadata carries the per-model capability fields surfaced through
// /v1/models so consumers (harnesses like Pi or jcode) get real context
// windows and feature flags from the catalog instead of guessing. It is
// purely additive: routing does not depend on any of these fields.
type ModelMetadata struct {
	ContextWindow    int      `json:"context_window,omitempty"`
	MaxInput         int      `json:"max_input,omitempty"`
	MaxOutput        int      `json:"max_output,omitempty"`
	Reasoning        bool     `json:"reasoning,omitempty"`
	ToolCall         bool     `json:"tool_call,omitempty"`
	StructuredOutput bool     `json:"structured_output,omitempty"`
	InputModalities  []string `json:"input_modalities,omitempty"`
	OutputModalities []string `json:"output_modalities,omitempty"`
}

func (m *capabilityModel) metadata() ModelMetadata {
	md := ModelMetadata{
		Reasoning:        m.Reasoning,
		ToolCall:         m.ToolCall,
		StructuredOutput: m.StructuredOutput,
	}
	if m.Modalities != nil {
		md.InputModalities = m.Modalities.Input
		md.OutputModalities = m.Modalities.Output
	}
	if m.Limit != nil {
		md.ContextWindow = m.Limit.Context
		md.MaxInput = m.Limit.Input
		md.MaxOutput = m.Limit.Output
	}
	return md
}

type capabilityModelProvider struct {
	NPM string `json:"npm"`
}

// fetchProtocolCapabilities reads OpenCode's machine-readable provider
// catalog. Unlike /v1/models, this source includes the SDK selected for each
// model, which is the upstream's protocol declaration. No model IDs are kept
// in this project.
func fetchProtocolCapabilities(ctx context.Context, client *http.Client, endpoint string) (protocolCapabilities, error) {
	if client == nil {
		client = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return protocolCapabilities{}, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", opencodeUserAgent())
	resp, err := client.Do(req)
	if err != nil {
		return protocolCapabilities{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return protocolCapabilities{}, fmt.Errorf("OpenCode capability endpoint returned HTTP %d", resp.StatusCode)
	}
	var providers map[string]capabilityProvider
	dec := json.NewDecoder(io.LimitReader(resp.Body, 64<<20))
	if err := dec.Decode(&providers); err != nil {
		return protocolCapabilities{}, err
	}
	result := protocolCapabilities{
		Protocols:   map[Tier]map[string]Protocol{TierZen: {}, TierGo: {}},
		Unsupported: map[Tier]map[string]bool{TierZen: {}, TierGo: {}},
		Metadata:    map[Tier]map[string]ModelMetadata{TierZen: {}, TierGo: {}},
	}
	for providerID, provider := range providers {
		tier, ok := capabilityTier(providerID, provider.API)
		if !ok {
			continue
		}
		for modelID, model := range provider.Models {
			if model.ID != "" {
				modelID = model.ID
			}
			npm := provider.NPM
			if model.Provider != nil && model.Provider.NPM != "" {
				npm = model.Provider.NPM
			}
			if protocol, ok := protocolForSDK(npm); ok {
				result.Protocols[tier][modelID] = protocol
			} else {
				result.Unsupported[tier][modelID] = true
			}
			result.Metadata[tier][modelID] = model.metadata()
		}
	}
	// The machine catalog is the primary source. The upstream endpoint tables
	// are a supplemental source for models whose provider inherits a default SDK
	// but whose published endpoint is more specific (for example a Messages
	// route). This remains data-driven: no model IDs are embedded here.
	for _, doc := range []struct {
		tier Tier
		url  string
	}{
		{TierZen, openCodeZenDocsURL},
		{TierGo, openCodeGoDocsURL},
	} {
		protocols, err := fetchProtocolDocs(ctx, client, doc.url)
		if err != nil {
			continue
		}
		for modelID, protocol := range protocols {
			result.Protocols[doc.tier][modelID] = protocol
			delete(result.Unsupported[doc.tier], modelID)
		}
	}
	if len(result.Protocols[TierZen]) == 0 && len(result.Protocols[TierGo]) == 0 && len(result.Unsupported[TierZen]) == 0 && len(result.Unsupported[TierGo]) == 0 {
		return protocolCapabilities{}, errors.New("OpenCode capability endpoint returned no Zen or Go models")
	}
	return result, nil
}

func fetchProtocolDocs(ctx context.Context, client *http.Client, endpoint string) (map[string]Protocol, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "text/plain, text/markdown, */*")
	req.Header.Set("User-Agent", opencodeUserAgent())
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("OpenCode endpoint documentation returned HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, err
	}
	result := make(map[string]Protocol)
	for _, line := range strings.Split(string(body), "\n") {
		match := protocolDocEndpointPattern.FindStringSubmatch(line)
		if len(match) != 3 {
			continue
		}
		modelID := strings.TrimSpace(match[1])
		if modelID == "" || strings.ContainsAny(modelID, " `|") {
			continue
		}
		var protocol Protocol
		switch match[2] {
		case "chat/completions":
			protocol = ProtocolChat
		case "responses":
			protocol = ProtocolResponses
		case "messages":
			protocol = ProtocolAnthropic
		}
		if protocol != "" {
			result[modelID] = protocol
		}
	}
	if len(result) == 0 {
		return nil, errors.New("OpenCode endpoint documentation returned no protocol rows")
	}
	return result, nil
}

func capabilityTier(providerID, api string) (Tier, bool) {
	value := strings.ToLower(strings.TrimSpace(providerID + " " + api))
	if strings.Contains(value, "opencode-go") || strings.Contains(value, "/go/") {
		return TierGo, true
	}
	if strings.Contains(value, "opencode") || strings.Contains(value, "/zen/") {
		return TierZen, true
	}
	return "", false
}

func protocolForSDK(npm string) (Protocol, bool) {
	value := strings.ToLower(strings.TrimSpace(npm))
	switch {
	case strings.Contains(value, "anthropic"):
		return ProtocolAnthropic, true
	case value == "@ai-sdk/openai" || strings.HasSuffix(value, "/openai"):
		return ProtocolResponses, true
	case strings.Contains(value, "openai-compatible"):
		return ProtocolChat, true
	default:
		return "", false
	}
}

type modelsResponse struct {
	Data []struct {
		ID string `json:"id"`
	} `json:"data"`
}

func fetchModels(ctx context.Context, client *http.Client, baseURL, key string) ([]string, int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(baseURL, "/")+"/v1/models", nil)
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("User-Agent", opencodeUserAgent())
	req.Header.Set("x-opencode-client", "cli")
	resp, err := client.Do(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, resp.StatusCode, fmt.Errorf("models endpoint returned HTTP %d", resp.StatusCode)
	}
	var payload modelsResponse
	dec := json.NewDecoder(io.LimitReader(resp.Body, 8<<20))
	if err := dec.Decode(&payload); err != nil {
		return nil, resp.StatusCode, err
	}
	models := make([]string, 0, len(payload.Data))
	for _, item := range payload.Data {
		if item.ID != "" {
			models = append(models, item.ID)
		}
	}
	if len(models) == 0 {
		return nil, resp.StatusCode, errors.New("models endpoint returned an empty list")
	}
	return models, resp.StatusCode, nil
}
