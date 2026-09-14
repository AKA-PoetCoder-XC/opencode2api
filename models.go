package main

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

type Protocol string

const (
	ProtocolChat      Protocol = "chat"
	ProtocolResponses Protocol = "responses"
	ProtocolAnthropic Protocol = "anthropic"
)

func validProtocol(p Protocol) bool {
	return p == ProtocolChat || p == ProtocolResponses || p == ProtocolAnthropic
}

type Tier string

const (
	TierZen Tier = "zen"
	TierGo  Tier = "go"
)

type modelRoute struct {
	ID       string
	Tier     Tier
	Protocol Protocol
	// Protocols is the native protocol for each possible upstream tier. Zen
	// and Go intentionally do not share one global protocol: OpenCode's
	// catalog currently exposes, for example, MiniMax through Chat on Zen and
	// Messages on Go. The request must therefore be re-encoded when a retry
	// crosses tiers.
	Protocols map[Tier]Protocol
	Anonymous bool
	// KeyTiers is the ordered authenticated fallback plan. Anonymous requests
	// always start on Zen, then enter this list when the public credential does
	// not succeed.
	KeyTiers []Tier
}

type ModelRouteDiagnostic struct {
	Model                string            `json:"model"`
	RequestedProtocol    Protocol          `json:"requested_protocol,omitempty"`
	NativeProtocol       Protocol          `json:"native_protocol"`
	NativeProtocols      map[Tier]Protocol `json:"native_protocols,omitempty"`
	ProtocolSource       string            `json:"protocol_source"`
	AvailableZen         bool              `json:"available_zen"`
	AvailableGo          bool              `json:"available_go"`
	Tier                 Tier              `json:"tier,omitempty"`
	Anonymous            bool              `json:"anonymous"`
	KeyID                string            `json:"key_id,omitempty"`
	Channel              string            `json:"channel,omitempty"`
	Attempts             int               `json:"attempts,omitempty"`
	KeyTiers             []Tier            `json:"key_tiers,omitempty"`
	AnonymousEligibility AnonymousDecision `json:"anonymous_eligibility"`
	RouteError           string            `json:"route_error,omitempty"`
}

type modelCatalog struct {
	mu        sync.RWMutex
	zen       map[string]bool
	goModels  map[string]bool
	protocols map[string]Protocol
	// nativeProtocols is populated from OpenCode's public model capability
	// catalog. protocols remains the user-configured override map.
	nativeProtocols map[Tier]map[string]Protocol
	unsupported     map[Tier]map[string]bool
	modelMeta       map[Tier]map[string]ModelMetadata
	updatedAt       time.Time
	prefer          Tier
	metadata        *modelMetadataStore
	cachePath       string
	cacheSource     string
	stale           bool
	refreshAfter    time.Duration
}

type modelCatalogSnapshot struct {
	Zen         int       `json:"zen"`
	Go          int       `json:"go"`
	Total       int       `json:"total"`
	Exposed     int       `json:"exposed"`
	UpdatedAt   time.Time `json:"updated_at,omitempty"`
	CacheSource string    `json:"cache_source,omitempty"`
	Stale       bool      `json:"stale"`
}

func newModelCatalog(prefer Tier, overrides map[string]string) *modelCatalog {
	protocols := make(map[string]Protocol, len(overrides))
	for model, protocol := range overrides {
		protocols[model] = Protocol(protocol)
	}
	return &modelCatalog{
		zen: map[string]bool{}, goModels: map[string]bool{}, protocols: protocols,
		nativeProtocols: map[Tier]map[string]Protocol{TierZen: {}, TierGo: {}},
		unsupported:     map[Tier]map[string]bool{TierZen: {}, TierGo: {}}, prefer: prefer,
		cacheSource: "none",
	}
}

func (c *modelCatalog) SetCachePath(path string) {
	c.mu.Lock()
	c.cachePath = path
	c.mu.Unlock()
}

func (c *modelCatalog) SetRefreshInterval(interval time.Duration) {
	c.mu.Lock()
	c.refreshAfter = interval
	c.mu.Unlock()
}

func (c *modelCatalog) Replace(zen, goModels []string) {
	c.ReplaceWithCapabilities(zen, goModels, nil, nil, nil)
}

func (c *modelCatalog) ReplaceWithCapabilities(zen, goModels []string, native map[Tier]map[string]Protocol, unsupported map[Tier]map[string]bool, metadata map[Tier]map[string]ModelMetadata) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if zen != nil {
		c.zen = toSet(zen)
	}
	if goModels != nil {
		c.goModels = toSet(goModels)
	}
	if native != nil {
		for _, tier := range []Tier{TierZen, TierGo} {
			if protocols, ok := native[tier]; ok {
				c.nativeProtocols[tier] = cloneProtocols(protocols)
			}
		}
	}
	if unsupported != nil {
		for _, tier := range []Tier{TierZen, TierGo} {
			if models, ok := unsupported[tier]; ok {
				c.unsupported[tier] = cloneBools(models)
			}
		}
	}
	if metadata != nil {
		c.modelMeta = cloneModelMeta(metadata)
	}
	c.updatedAt = time.Now().UTC()
	c.cacheSource = "live"
	c.stale = false
}

func (c *modelCatalog) CopyState(source *modelCatalog) {
	if source == nil {
		return
	}
	source.mu.RLock()
	zen := make(map[string]bool, len(source.zen))
	goModels := make(map[string]bool, len(source.goModels))
	for model, available := range source.zen {
		zen[model] = available
	}
	for model, available := range source.goModels {
		goModels[model] = available
	}
	native := map[Tier]map[string]Protocol{TierZen: {}, TierGo: {}}
	unsupported := map[Tier]map[string]bool{TierZen: {}, TierGo: {}}
	for _, tier := range []Tier{TierZen, TierGo} {
		for model, protocol := range source.nativeProtocols[tier] {
			native[tier][model] = protocol
		}
		for model, value := range source.unsupported[tier] {
			unsupported[tier][model] = value
		}
	}
	meta := cloneModelMeta(source.modelMeta)
	updatedAt := source.updatedAt
	cacheSource := source.cacheSource
	stale := source.stale
	source.mu.RUnlock()
	c.mu.Lock()
	c.zen, c.goModels, c.nativeProtocols, c.unsupported, c.updatedAt = zen, goModels, native, unsupported, updatedAt
	c.modelMeta = meta
	c.cacheSource, c.stale = cacheSource, stale
	c.mu.Unlock()
}

func (c *modelCatalog) Route(model string, hasZenKeys, hasGoKeys, hasAnonymous bool) (modelRoute, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.routeLocked(model, hasZenKeys, hasGoKeys, hasAnonymous)
}

func (c *modelCatalog) routeLocked(model string, hasZenKeys, hasGoKeys, hasAnonymous bool) (modelRoute, error) {
	keyTiers := c.keyTierOrderLocked(model, hasZenKeys, hasGoKeys)
	// OpenCode's public credential is a Zen-only lane. Every free model starts
	// there, even if the current catalog only advertises it on Go: an upstream
	// rejection will move the request into the authenticated fallback plan.
	decision := c.anonymousDecision(model)
	if hasAnonymous && decision.Allowed && (c.protocols[model] != "" || !c.unsupported[TierZen][model]) &&
		(len(c.zen) == 0 && len(c.goModels) == 0 || c.zen[model] || c.goModels[model]) {
		protocols := c.protocolsForLocked(model, keyTiers, true)
		return modelRoute{ID: model, Tier: TierZen, Protocol: protocols[TierZen], Protocols: protocols, Anonymous: true, KeyTiers: keyTiers}, nil
	}
	if len(keyTiers) > 0 {
		protocols := c.protocolsForLocked(model, keyTiers, false)
		return modelRoute{ID: model, Tier: keyTiers[0], Protocol: protocols[keyTiers[0]], Protocols: protocols, KeyTiers: keyTiers}, nil
	}
	return modelRoute{}, fmt.Errorf("model %q is not available in the configured Zen or Go pools", model)
}

func (r modelRoute) ProtocolFor(tier Tier) Protocol {
	if protocol := r.Protocols[tier]; protocol != "" {
		return protocol
	}
	return r.Protocol
}

func (c *modelCatalog) protocolsForLocked(model string, keyTiers []Tier, includeZen bool) map[Tier]Protocol {
	protocols := make(map[Tier]Protocol, len(keyTiers)+1)
	if includeZen {
		protocols[TierZen] = c.protocolForLocked(model, TierZen)
	}
	for _, tier := range keyTiers {
		protocols[tier] = c.protocolForLocked(model, tier)
	}
	return protocols
}

func (c *modelCatalog) protocolForLocked(model string, tier Tier) Protocol {
	if protocol := c.protocols[model]; protocol != "" {
		return protocol
	}
	if protocol := c.nativeProtocols[tier][model]; protocol != "" {
		return protocol
	}
	// The OpenCode capability catalog is authoritative when available. Chat is
	// the only safe protocol-neutral fallback for an ID that has just appeared
	// in /v1/models but is not present in the capability snapshot yet.
	return ProtocolChat
}

// keyTierOrderLocked builds an authenticated route in prefer order. A tier is
// included only when it has a key and advertises the model. Before the first
// successful catalog refresh, configured key pools remain usable so temporary
// discovery failures do not take the gateway offline.
func (c *modelCatalog) keyTierOrderLocked(model string, hasZenKeys, hasGoKeys bool) []Tier {
	catalogPending := len(c.zen) == 0 && len(c.goModels) == 0
	available := func(tier Tier) bool {
		switch tier {
		case TierZen:
			return hasZenKeys && (catalogPending || c.zen[model]) && c.tierSupportedLocked(model, TierZen)
		case TierGo:
			return hasGoKeys && (catalogPending || c.goModels[model]) && c.tierSupportedLocked(model, TierGo)
		default:
			return false
		}
	}
	order := []Tier{TierZen, TierGo}
	if c.prefer == TierGo {
		order[0], order[1] = order[1], order[0]
	}
	result := make([]Tier, 0, len(order))
	for _, tier := range order {
		if available(tier) {
			result = append(result, tier)
		}
	}
	return result
}

// RouteForTier builds a route pinned to one tier, used by the Playground's
// per-key diagnostics. It never selects the anonymous lane and never adds a
// fallback tier: the operator asked to exercise one configured key, so the
// result describes that key and nothing else.
func (c *modelCatalog) RouteForTier(model string, tier Tier, hasZenKeys, hasGoKeys bool) (modelRoute, error) {
	if tier != TierZen && tier != TierGo {
		return modelRoute{}, errors.New("selected key tier must be zen or go")
	}
	hasKeys := hasZenKeys
	if tier == TierGo {
		hasKeys = hasGoKeys
	}
	if !hasKeys {
		return modelRoute{}, fmt.Errorf("no %s key is configured", tier)
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	catalogPending := len(c.zen) == 0 && len(c.goModels) == 0
	advertised := c.zen[model]
	if tier == TierGo {
		advertised = c.goModels[model]
	}
	if !catalogPending && !advertised {
		return modelRoute{}, fmt.Errorf("model %q is not available in the selected %s key tier", model, tier)
	}
	if !c.tierSupportedLocked(model, tier) {
		return modelRoute{}, fmt.Errorf("model %q uses an upstream protocol that is not available on the selected %s key tier", model, tier)
	}
	protocol := c.protocolForLocked(model, tier)
	return modelRoute{
		ID: model, Tier: tier, Protocol: protocol,
		Protocols: map[Tier]Protocol{tier: protocol},
		KeyTiers:  []Tier{tier},
	}, nil
}

func (c *modelCatalog) anonymousDecision(model string) AnonymousDecision {
	if c.metadata != nil {
		return c.metadata.Decide(model)
	}
	return AnonymousDecision{Allowed: isFreeModel(model), Source: "name_fallback_metadata_pending"}
}

func (c *modelCatalog) Diagnostic(model string, requested Protocol, hasZenKeys, hasGoKeys, hasAnonymous bool) ModelRouteDiagnostic {
	c.mu.RLock()
	configured, explicit := c.protocols[model]
	zen, goModel := c.zen[model], c.goModels[model]
	nativeProtocols := map[Tier]Protocol{
		TierZen: c.protocolForLocked(model, TierZen),
		TierGo:  c.protocolForLocked(model, TierGo),
	}
	_, zenKnown := c.nativeProtocols[TierZen][model]
	_, goKnown := c.nativeProtocols[TierGo][model]
	c.mu.RUnlock()
	source := "configured"
	if !explicit {
		source = "default"
		if zenKnown || goKnown {
			source = "upstream"
		}
	}
	protocol := configured
	if protocol == "" {
		// Route() below selects the preferred available tier. This is only the
		// fallback shown when no route can currently be built.
		protocol = nativeProtocols[TierZen]
		if c.prefer == TierGo {
			protocol = nativeProtocols[TierGo]
		}
	}
	diagnostic := ModelRouteDiagnostic{
		Model: model, RequestedProtocol: requested, NativeProtocol: protocol, NativeProtocols: nativeProtocols, ProtocolSource: source,
		AvailableZen: zen, AvailableGo: goModel, AnonymousEligibility: c.anonymousDecision(model),
	}
	route, err := c.Route(model, hasZenKeys, hasGoKeys, hasAnonymous)
	if err != nil {
		diagnostic.RouteError = err.Error()
		return diagnostic
	}
	diagnostic.NativeProtocol = route.Protocol
	diagnostic.NativeProtocols = route.Protocols
	diagnostic.Tier, diagnostic.Anonymous = route.Tier, route.Anonymous
	diagnostic.KeyTiers = append([]Tier(nil), route.KeyTiers...)
	return diagnostic
}

func isFreeModel(model string) bool {
	return strings.Contains(strings.ToLower(model), "free")
}

func (c *modelCatalog) List() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	ids := c.modelIDsLocked()
	models := ids[:0]
	for _, model := range ids {
		if c.supportedLocked(model) {
			models = append(models, model)
		}
	}
	return models
}

func (c *modelCatalog) modelIDsLocked() []string {
	seen := make(map[string]bool, len(c.zen)+len(c.goModels))
	for model := range c.zen {
		seen[model] = true
	}
	for model := range c.goModels {
		seen[model] = true
	}
	return sortedSetKeys(seen)
}

func (c *modelCatalog) Snapshot() modelCatalogSnapshot {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.snapshotLocked(c.modelIDsLocked())
}

// AvailableModels provides discovery and readiness with the same route
// filtering, including configured key tiers and anonymous eligibility.
func (c *modelCatalog) AvailableModels(hasZenKeys, hasGoKeys, hasAnonymous bool) ([]modelRoute, modelCatalogSnapshot) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	ids := c.modelIDsLocked()
	routes := make([]modelRoute, 0, len(ids))
	for _, model := range ids {
		if !c.supportedLocked(model) {
			continue
		}
		if route, err := c.routeLocked(model, hasZenKeys, hasGoKeys, hasAnonymous); err == nil {
			routes = append(routes, route)
		}
	}
	snapshot := c.snapshotLocked(ids)
	snapshot.Exposed = len(routes)
	return routes, snapshot
}

func (c *modelCatalog) snapshotLocked(ids []string) modelCatalogSnapshot {
	exposed := 0
	for _, model := range ids {
		if c.supportedLocked(model) {
			exposed++
		}
	}
	stale := c.stale
	if !c.updatedAt.IsZero() && c.refreshAfter > 0 {
		stale = stale || time.Since(c.updatedAt) > max(2*c.refreshAfter, time.Minute)
	}
	return modelCatalogSnapshot{
		Zen: len(c.zen), Go: len(c.goModels), Total: len(ids), Exposed: exposed,
		UpdatedAt: c.updatedAt, CacheSource: c.cacheSource, Stale: stale,
	}
}

func (c *modelCatalog) Supported(model string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.supportedLocked(model)
}

// MetadataForTier returns the rich per-model metadata (context window,
// reasoning, tool call, modalities) captured from the opencode catalog for
// the tier that will actually serve the request. Same-named models can carry
// different limits per tier, so callers must pass route.Tier — never a
// tier-blind lookup. Anonymous routes always resolve to TierZen, which keeps
// the keyless path on Zen metadata. The zero value is returned for models
// the catalog does not describe (or before the first capability refresh).
func (c *modelCatalog) MetadataForTier(model string, tier Tier) ModelMetadata {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.modelMeta[tier][model]
}

func (c *modelCatalog) supportedLocked(model string) bool {
	if len(c.zen) == 0 && len(c.goModels) == 0 {
		return true
	}
	if c.zen[model] && c.tierSupportedLocked(model, TierZen) {
		return true
	}
	if c.goModels[model] && c.tierSupportedLocked(model, TierGo) {
		return true
	}
	return false
}

func (c *modelCatalog) tierSupportedLocked(model string, tier Tier) bool {
	if c.protocols[model] != "" {
		return true
	}
	if c.unsupported[tier][model] {
		return false
	}
	if c.nativeProtocols[tier][model] != "" {
		return true
	}
	// A pending catalog has no upstream capability snapshot to contradict a
	// configured key, so retain the pre-refresh compatibility behavior.
	return len(c.zen) == 0 && len(c.goModels) == 0
}

func toSet(items []string) map[string]bool {
	out := make(map[string]bool, len(items))
	for _, item := range items {
		out[item] = true
	}
	return out
}

func cloneProtocols(source map[string]Protocol) map[string]Protocol {
	result := make(map[string]Protocol, len(source))
	for model, protocol := range source {
		result[model] = protocol
	}
	return result
}

func cloneBools(source map[string]bool) map[string]bool {
	result := make(map[string]bool, len(source))
	for model, value := range source {
		result[model] = value
	}
	return result
}

func cloneModelMeta(source map[Tier]map[string]ModelMetadata) map[Tier]map[string]ModelMetadata {
	result := map[Tier]map[string]ModelMetadata{TierZen: {}, TierGo: {}}
	for _, tier := range []Tier{TierZen, TierGo} {
		for id, md := range source[tier] {
			result[tier][id] = md
		}
	}
	return result
}

func sortedSetKeys(source map[string]bool) []string {
	result := make([]string, 0, len(source))
	for model, available := range source {
		if available {
			result = append(result, model)
		}
	}
	sort.Strings(result)
	return result
}

func cloneTierProtocols(source map[Tier]map[string]Protocol) map[Tier]map[string]Protocol {
	result := map[Tier]map[string]Protocol{TierZen: {}, TierGo: {}}
	for _, tier := range []Tier{TierZen, TierGo} {
		if protocols, ok := source[tier]; ok {
			result[tier] = cloneProtocols(protocols)
		}
	}
	return result
}

func cloneTierBools(source map[Tier]map[string]bool) map[Tier]map[string]bool {
	result := map[Tier]map[string]bool{TierZen: {}, TierGo: {}}
	for _, tier := range []Tier{TierZen, TierGo} {
		if models, ok := source[tier]; ok {
			result[tier] = cloneBools(models)
		}
	}
	return result
}
