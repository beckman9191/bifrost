package streamusage

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/bytedance/sonic"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	"github.com/maximhq/bifrost/core/schemas"
)

// Accumulator is an optional tracer capability. A gateway reuses its existing
// accumulator; the standalone SDK uses the bounded-lifetime fallback below.
// The extra reference is released by Close, including on startup failure.
type Accumulator interface {
	BeginUsageAccumulation(*schemas.BifrostContext) bool
	CaptureUsageChunk(*schemas.BifrostContext, *schemas.BifrostResponse)
	EndUsageAccumulation(*schemas.BifrostContext)
}

// Manager owns per-attempt state outside BifrostContext. Entries are removed by
// the provider's once-only stream finalizer; retries each acquire a fresh entry.
// Retained content is bounded by the received output of active streams, and is
// not duplicated when the tracer supplies an accumulator.
type Manager struct {
	next    atomic.Uint64
	streams sync.Map
}

type Stream struct {
	tokenizers        map[string]bool
	manager           *Manager
	id                uint64
	mu                sync.Mutex
	closed            bool
	request           schemas.BifrostRequest
	model             string
	provider          schemas.ModelProvider
	ctx               *schemas.BifrostContext
	backend           Accumulator
	chunks            int
	finished          bool
	reported          *schemas.BifrostLLMUsage
	completionUpdates int
	settled           bool
	allChoices        bool
	fallback          outputBuffer
}

func (m *Manager) Open(ctx *schemas.BifrostContext, request *schemas.BifrostRequest, provider schemas.ModelProvider, model string, tokenizers ...string) *Stream {
	switch request.RequestType {
	case schemas.ChatCompletionStreamRequest, schemas.TextCompletionStreamRequest, schemas.ResponsesStreamRequest:
	default:
		return nil
	}
	ctx.SetValue(schemas.BifrostContextKeyStreamAccumulatedUsage, nil)
	ctx.SetValue(schemas.BifrostContextKeyStreamUsageUpdates, 0)
	s := &Stream{manager: m, id: m.next.Add(1), ctx: ctx, request: *request, provider: provider, model: model}
	if len(tokenizers) > 0 {
		s.tokenizers = make(map[string]bool, len(tokenizers))
		for _, kind := range tokenizers {
			s.tokenizers[kind] = true
		}
	}
	if request.ChatRequest != nil {
		copy := *request.ChatRequest
		s.request.ChatRequest = &copy
	}
	if request.ResponsesRequest != nil {
		copy := *request.ResponsesRequest
		s.request.ResponsesRequest = &copy
	}
	if request.TextCompletionRequest != nil {
		copy := *request.TextCompletionRequest
		s.request.TextCompletionRequest = &copy
	}
	if tracer, ok := ctx.Value(schemas.BifrostContextKeyTracer).(Accumulator); ok && tracer.BeginUsageAccumulation(ctx) {
		s.backend = tracer
	}
	m.streams.Store(s.id, s)
	return s
}

func (s *Stream) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.closed = true
	if s.backend != nil {
		s.backend.EndUsageAccumulation(s.ctx)
	}
	s.request = schemas.BifrostRequest{}
	s.fallback = outputBuffer{}
	s.reported = nil
	s.manager.streams.Delete(s.id)
}

// Observe runs before accounting plugins, so every consumer sees the same usage.
// The success path never tokenizes or builds an accumulated response snapshot.
func (s *Stream) Observe(ctx *schemas.BifrostContext, response *schemas.BifrostResponse, failure *schemas.BifrostError) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.settled {
		return
	}
	if response != nil {
		s.chunks++
		if s.backend != nil && !s.allChoices && response.ChatResponse != nil {
			for _, choice := range response.ChatResponse.Choices {
				if choice.Index != 0 {
					s.allChoices = true
					if snapshot := ctx.GetAccumulatedResponse(); snapshot != nil {
						s.fallback.seed(snapshot)
					}
					break
				}
			}
		}
		if s.backend == nil || s.allChoices {
			s.fallback.add(response)
		}
		usage := response.NormalizedUsage()
		if response.ResponsesStreamResponse != nil && response.ResponsesStreamResponse.Response != nil {
			usage = response.ResponsesStreamResponse.Response.Usage.ToBifrostLLMUsage()
		}
		if usage != nil {
			if usage.CompletionTokens > 0 {
				s.completionUpdates++
			}
			if s.reported == nil {
				s.reported = usage.DeepCopy()
			} else {
				// LiteLLM uses the last positive count, preserving explicit zero
				// only until a positive count arrives (not a maximum).
				previous := s.reported
				s.reported = usage.DeepCopy()
				if usage.PromptTokens == 0 {
					s.reported.PromptTokens = previous.PromptTokens
				}
				if usage.CompletionTokens == 0 {
					s.reported.CompletionTokens = previous.CompletionTokens
				}
				if s.reported.PromptTokensDetails == nil {
					s.reported.PromptTokensDetails = previous.PromptTokensDetails
				}
				if s.reported.CompletionTokensDetails == nil {
					s.reported.CompletionTokensDetails = previous.CompletionTokensDetails
				}
			}
		}
		if response.ChatResponse != nil {
			for _, c := range response.ChatResponse.Choices {
				if c.FinishReason != nil {
					s.finished = true
				}
			}
		}
		if response.TextCompletionResponse != nil {
			for _, c := range response.TextCompletionResponse.Choices {
				if c.FinishReason != nil {
					s.finished = true
				}
			}
		}
		if r := response.ResponsesStreamResponse; r != nil && (r.Type == schemas.ResponsesStreamResponseTypeCompleted || r.Type == schemas.ResponsesStreamResponseTypeIncomplete) {
			s.finished = true
		}
	}
	if failure == nil {
		if final, _ := ctx.Value(schemas.BifrostContextKeyStreamEndIndicator).(bool); final {
			s.settled = true
		}
		return
	}
	if failure.Error == nil || failure.Error.Type == nil || (*failure.Error.Type != schemas.RequestCancelled && *failure.Error.Type != schemas.RequestTimedOut) {
		return
	}
	s.settled = true
	if s.chunks == 0 {
		return
	} // A failure before any output is not estimated as spend.
	reported := failure.ExtraFields.BilledUsage
	if reported == nil {
		reported = s.reported
	}
	estimatePrompt, estimateCompletion := reported == nil, reported == nil
	// Native Anthropic's accumulating handle includes a provisional message_start
	// count. A message_delta/finish event makes it authoritative, even if it is 1.
	updates, _ := ctx.Value(schemas.BifrostContextKeyStreamUsageUpdates).(int)
	if s.provider == schemas.Anthropic && !s.finished && updates < 2 && s.completionUpdates < 2 {
		estimateCompletion = true
	}
	if !estimatePrompt && !estimateCompletion {
		if failure.ExtraFields.BilledUsage == nil {
			failure.ExtraFields.BilledUsage = reported.DeepCopy()
		}
		return
	}
	output, reasoning := s.fallback.text()
	if s.backend != nil && !s.allChoices {
		snapshot := ctx.GetAccumulatedResponse()
		if snapshot == nil {
			return
		}
		output, reasoning = responseText(snapshot)
	}
	c, err := selectConfiguredCounter(s.model, s.tokenizers)
	if err != nil {
		return
	}
	usage := reported.DeepCopy()
	if usage == nil {
		usage = &schemas.BifrostLLMUsage{}
	}
	if estimatePrompt {
		n, err := promptCount(c, &s.request)
		if err != nil {
			estimatePrompt = false
		} else {
			usage.PromptTokens = n
		}
	}
	if estimateCompletion {
		n, err := c.count(output)
		if err != nil {
			return
		}
		// The reference counts reasoning without model=, i.e. cl100k_base.
		rc, err := selectCounter("")
		if err != nil {
			return
		}
		rn := 0
		for _, part := range reasoning {
			n, err := rc.count(part)
			if err != nil {
				return
			}
			rn += n
		}
		usage.CompletionTokens = n + rn
		if len(reasoning) > 0 && (usage.CompletionTokensDetails == nil || usage.CompletionTokensDetails.ReasoningTokens == 0) {
			if usage.CompletionTokensDetails == nil {
				usage.CompletionTokensDetails = &schemas.ChatCompletionTokensDetails{}
			}
			usage.CompletionTokensDetails.ReasoningTokens = rn
		}
	}
	usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	usage.Estimation = &schemas.TokenUsageEstimation{Prompt: estimatePrompt, Completion: estimateCompletion, Tokenizer: c.name, Reference: ReferenceCommit}
	failure.ExtraFields.BilledUsage = usage
}

type toolBuffer struct {
	id, name, kind string
	arguments      strings.Builder
}
type choiceBuffer struct {
	content, reasoning strings.Builder
	tools              map[int]*toolBuffer
}
type outputBuffer struct{ choices map[int]*choiceBuffer }

func (o *outputBuffer) choice(index int) *choiceBuffer {
	if o.choices == nil {
		o.choices = make(map[int]*choiceBuffer)
	}
	if o.choices[index] == nil {
		o.choices[index] = &choiceBuffer{}
	}
	return o.choices[index]
}
func (o *outputBuffer) add(response *schemas.BifrostResponse) {
	if r := response.ChatResponse; r != nil {
		for _, choice := range r.Choices {
			if choice.ChatStreamResponseChoice == nil || choice.Delta == nil {
				continue
			}
			c, d := o.choice(choice.Index), choice.Delta
			if d.Content != nil {
				c.content.WriteString(*d.Content)
			}
			if d.Reasoning != nil {
				c.reasoning.WriteString(*d.Reasoning)
			}
			for _, t := range d.ToolCalls {
				if c.tools == nil {
					c.tools = make(map[int]*toolBuffer)
				}
				b := c.tools[int(t.Index)]
				if b == nil {
					b = &toolBuffer{kind: "function"}
					c.tools[int(t.Index)] = b
				}
				if t.ID != nil {
					b.id = *t.ID
				}
				if t.Type != nil {
					b.kind = *t.Type
				}
				if t.Function.Name != nil {
					b.name += *t.Function.Name
				}
				b.arguments.WriteString(t.Function.Arguments)
			}
		}
	}
	if r := response.TextCompletionResponse; r != nil {
		for _, c := range r.Choices {
			if c.TextCompletionResponseChoice != nil && c.Text != nil {
				o.choice(c.Index).content.WriteString(*c.Text)
			}
		}
	}
	if r := response.ResponsesStreamResponse; r != nil {
		index := 0
		if r.OutputIndex != nil {
			index = *r.OutputIndex
		}
		c := o.choice(index)
		if r.Delta != nil {
			switch r.Type {
			case schemas.ResponsesStreamResponseTypeOutputTextDelta:
				c.content.WriteString(*r.Delta)
			case schemas.ResponsesStreamResponseTypeReasoningSummaryTextDelta:
				c.reasoning.WriteString(*r.Delta)
			case schemas.ResponsesStreamResponseTypeFunctionCallArgumentsDelta:
				if c.tools == nil {
					c.tools = map[int]*toolBuffer{0: {kind: "function"}}
				}
				c.tools[0].arguments.WriteString(*r.Delta)
			}
		}
		if r.Item != nil && r.Item.ResponsesToolMessage != nil {
			t := r.Item.ResponsesToolMessage
			if c.tools == nil {
				c.tools = map[int]*toolBuffer{0: {kind: "function"}}
			}
			b := c.tools[0]
			if t.Name != nil {
				b.name = *t.Name
			}
			if t.CallID != nil {
				b.id = *t.CallID
			}
		}
	}
}
func (o *outputBuffer) text() (string, []string) {
	keys := make([]int, 0, len(o.choices))
	for k := range o.choices {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	var output strings.Builder
	var reasoning []string
	for _, k := range keys {
		c := o.choices[k]
		output.WriteString(c.content.String())
		if c.reasoning.Len() > 0 {
			reasoning = append(reasoning, c.reasoning.String())
		}
		tools := make([]int, 0, len(c.tools))
		for k := range c.tools {
			tools = append(tools, k)
		}
		sort.Ints(tools)
		for _, k := range tools {
			b := c.tools[k]
			output.WriteString(toolJSON(b.id, b.name, b.kind, b.arguments.String()))
		}
	}
	return output.String(), reasoning
}
func toolJSON(id, name, kind, arguments string) string {
	// Same field order as LiteLLM's ChatCompletionMessageToolCall.model_dump_json.
	value := struct {
		Function struct {
			Arguments string `json:"arguments"`
			Name      string `json:"name"`
		} `json:"function"`
		ID   string `json:"id"`
		Type string `json:"type"`
	}{ID: id, Type: kind}
	value.Function.Name, value.Function.Arguments = name, arguments
	b, _ := providerUtils.MarshalSorted(value)
	return string(b)
}
func responseText(response *schemas.BifrostResponse) (string, []string) {
	if response == nil {
		return "", nil
	}
	if response.ResponsesResponse != nil {
		response = &schemas.BifrostResponse{ChatResponse: response.ResponsesResponse.ToBifrostChatResponse()}
	}
	var text strings.Builder
	var reasoning []string
	if response.TextCompletionResponse != nil {
		for _, c := range response.TextCompletionResponse.Choices {
			if c.TextCompletionResponseChoice != nil && c.Text != nil {
				text.WriteString(*c.Text)
			}
		}
	}
	if response.ChatResponse != nil {
		for _, c := range response.ChatResponse.Choices {
			if c.ChatNonStreamResponseChoice == nil || c.Message == nil {
				continue
			}
			m := c.Message
			if m.Content != nil && m.Content.ContentStr != nil {
				text.WriteString(*m.Content.ContentStr)
			}
			if m.ChatAssistantMessage != nil {
				if m.Reasoning != nil {
					if *m.Reasoning != "" {
						reasoning = append(reasoning, *m.Reasoning)
					}
				}
				for _, t := range m.ToolCalls {
					id, name, kind := "", "", "function"
					if t.ID != nil {
						id = *t.ID
					}
					if t.Function.Name != nil {
						name = *t.Function.Name
					}
					if t.Type != nil {
						kind = *t.Type
					}
					text.WriteString(toolJSON(id, name, kind, t.Function.Arguments))
				}
			}
		}
	}
	return text.String(), reasoning
}

func promptCount(c namedCounter, request *schemas.BifrostRequest) (int, error) {
	chat := request.ChatRequest
	if chat == nil && request.ResponsesRequest != nil {
		chat = request.ResponsesRequest.ToChatRequest()
	}
	if request.TextCompletionRequest != nil && request.TextCompletionRequest.Input != nil {
		input := request.TextCompletionRequest.Input
		if input.PromptStr != nil {
			return c.count(*input.PromptStr)
		}
		return 0, fmt.Errorf("non-text prompt")
	}
	if chat == nil {
		return 0, fmt.Errorf("request unavailable")
	}
	// Full-document decode is intentional: LiteLLM counts every string-valued
	// message field, plus content blocks and tool arguments.
	encoded, err := providerUtils.MarshalSorted(chat.Input)
	if err != nil {
		return 0, err
	}
	var messages []map[string]interface{}
	if err := sonic.Unmarshal(encoded, &messages); err != nil {
		return 0, err
	}
	total := 3
	for _, message := range messages {
		overhead, nameOverhead := 3, 1
		if chat.Model == "gpt-3.5-turbo-0301" {
			overhead, nameOverhead = 4, -1
		}
		total += overhead
		for key, value := range message {
			switch v := value.(type) {
			case string:
				// Bifrost emits both reasoning spellings; the neutral source is one field.
				if key == "reasoning_content" {
					continue
				}
				n, err := c.count(v)
				if err != nil {
					return 0, err
				}
				total += n
				if key == "name" {
					total += nameOverhead
				}
			case []interface{}:
				for _, item := range v {
					block, ok := item.(map[string]interface{})
					if !ok {
						continue
					}
					var text string
					if key == "tool_calls" {
						if fn, ok := block["function"].(map[string]interface{}); ok {
							text, _ = fn["arguments"].(string)
						}
					} else if key == "content" {
						text, _ = block["text"].(string)
						if text == "" {
							if typ, _ := block["type"].(string); typ != "text" && typ != "input_text" && typ != "output_text" {
								return 0, fmt.Errorf("unsupported prompt content %s", typ)
							}
						}
					}
					n, err := c.count(text)
					if err != nil {
						return 0, err
					}
					total += n
				}
			}
		}
	}
	return total, nil
}

// Capture runs after post-hooks on normal chunks. Accounting of a cancellation
// runs before hooks against the chunks already captured by previous invocations.
func (s *Stream) Capture(ctx *schemas.BifrostContext, response *schemas.BifrostResponse) {
	if s == nil || response == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.closed && s.backend != nil {
		s.backend.CaptureUsageChunk(ctx, response)
	}
}

func (o *outputBuffer) seed(response *schemas.BifrostResponse) {
	if response.ChatResponse == nil {
		return
	}
	for _, choice := range response.ChatResponse.Choices {
		if choice.ChatNonStreamResponseChoice == nil || choice.Message == nil {
			continue
		}
		m, c := choice.Message, o.choice(choice.Index)
		if m.Content != nil && m.Content.ContentStr != nil {
			c.content.WriteString(*m.Content.ContentStr)
		}
		if m.ChatAssistantMessage != nil {
			if m.Reasoning != nil {
				c.reasoning.WriteString(*m.Reasoning)
			}
			for _, t := range m.ToolCalls {
				if c.tools == nil {
					c.tools = make(map[int]*toolBuffer)
				}
				b := &toolBuffer{kind: "function"}
				if t.ID != nil {
					b.id = *t.ID
				}
				if t.Function.Name != nil {
					b.name = *t.Function.Name
				}
				if t.Type != nil {
					b.kind = *t.Type
				}
				b.arguments.WriteString(t.Function.Arguments)
				c.tools[int(t.Index)] = b
			}
		}
	}
}
