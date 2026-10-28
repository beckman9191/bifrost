package streamusage

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/maximhq/bifrost/core/schemas"
)

func testStream(t *testing.T, provider schemas.ModelProvider) (*Stream, *schemas.BifrostContext) {
	t.Helper()
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	m := &Manager{}
	s := m.Open(ctx, &schemas.BifrostRequest{RequestType: schemas.ChatCompletionStreamRequest, ChatRequest: &schemas.BifrostChatRequest{Model: "unknown", Input: []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")}}}}}, provider, "unknown")
	t.Cleanup(func() {
		s.Close()
		s.Close()
		m.streams.Range(func(_, _ any) bool { t.Error("stream entry leaked"); return true })
	})
	return s, ctx
}
func textChunk(text string) *schemas.BifrostResponse {
	return &schemas.BifrostResponse{ChatResponse: &schemas.BifrostChatResponse{Choices: []schemas.BifrostResponseChoice{{Index: 0, ChatStreamResponseChoice: &schemas.ChatStreamResponseChoice{Delta: &schemas.ChatStreamResponseChoiceDelta{Content: schemas.Ptr(text)}}}}}}
}
func cancellation() *schemas.BifrostError {
	return &schemas.BifrostError{Error: &schemas.ErrorField{Type: schemas.Ptr(schemas.RequestCancelled)}}
}

func TestInterruptedUsageReconciliation(t *testing.T) {
	for _, tc := range []struct {
		name      string
		provider  schemas.ModelProvider
		reported  *schemas.BifrostLLMUsage
		updates   int
		finished  bool
		want      int
		estimated bool
	}{
		{name: "missing usage", provider: schemas.OpenAI, want: 2, estimated: true},
		{name: "explicit zero", provider: schemas.OpenAI, reported: &schemas.BifrostLLMUsage{PromptTokens: 10}, want: 0},
		{name: "reported count", provider: schemas.Gemini, reported: &schemas.BifrostLLMUsage{PromptTokens: 10, CompletionTokens: 9, TotalTokens: 19}, want: 9},
		{name: "anthropic provisional", provider: schemas.Anthropic, reported: &schemas.BifrostLLMUsage{PromptTokens: 26, CompletionTokens: 4, TotalTokens: 30}, updates: 1, want: 2, estimated: true},
		{name: "anthropic later usage", provider: schemas.Anthropic, reported: &schemas.BifrostLLMUsage{PromptTokens: 26, CompletionTokens: 1, TotalTokens: 27}, updates: 2, want: 1},
		{name: "anthropic finish", provider: schemas.Anthropic, reported: &schemas.BifrostLLMUsage{PromptTokens: 26, CompletionTokens: 1, TotalTokens: 27}, updates: 1, finished: true, want: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, ctx := testStream(t, tc.provider)
			s.Observe(ctx, textChunk("hello "), nil)
			s.Observe(ctx, textChunk("world"), nil)
			s.finished = tc.finished
			ctx.SetValue(schemas.BifrostContextKeyStreamUsageUpdates, tc.updates)
			err := cancellation()
			err.ExtraFields.BilledUsage = tc.reported
			s.Observe(ctx, nil, err)
			got := err.ExtraFields.BilledUsage
			if got == nil || got.CompletionTokens != tc.want {
				t.Fatalf("usage=%+v want completion=%d", got, tc.want)
			}
			if (got.Estimation != nil) != tc.estimated {
				t.Fatalf("estimation=%+v", got.Estimation)
			}
			if tc.reported != nil && got.PromptTokens != tc.reported.PromptTokens {
				t.Fatal("reported prompt changed")
			}
			if got.TotalTokens != got.PromptTokens+got.CompletionTokens && tc.estimated {
				t.Fatal("inconsistent total")
			}
			before := got.DeepCopy()
			s.Observe(ctx, nil, err)
			if err.ExtraFields.BilledUsage.CompletionTokens != before.CompletionTokens {
				t.Fatal("settled twice")
			}
		})
	}
}
func TestCancellationBeforeAnyChunkDoesNotInventUsage(t *testing.T) {
	s, ctx := testStream(t, schemas.OpenAI)
	err := cancellation()
	s.Observe(ctx, nil, err)
	if err.ExtraFields.BilledUsage != nil {
		t.Fatal("billed before output")
	}
}
func TestFinalUsageWinsCancellationRace(t *testing.T) {
	s, ctx := testStream(t, schemas.OpenAI)
	chunk := textChunk("hello world")
	chunk.ChatResponse.Usage = &schemas.BifrostLLMUsage{PromptTokens: 8, CompletionTokens: 2, TotalTokens: 10}
	ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
	s.Observe(ctx, chunk, nil)
	err := cancellation()
	s.Observe(ctx, nil, err)
	if err.ExtraFields.BilledUsage != nil {
		t.Fatal("a completed stream was billed twice")
	}
}
func TestReasoningAndToolsAreCountedOnce(t *testing.T) {
	s, ctx := testStream(t, schemas.OpenAI)
	a := textChunk("hello ")
	a.ChatResponse.Choices[0].Delta.Reasoning = schemas.Ptr("think ")
	a.ChatResponse.Choices[0].Delta.ToolCalls = []schemas.ChatAssistantMessageToolCall{{ID: schemas.Ptr("call_1"), Type: schemas.Ptr("function"), Function: schemas.ChatAssistantMessageToolCallFunction{Name: schemas.Ptr("weather"), Arguments: `{"city":`}}}
	b := textChunk("world")
	b.ChatResponse.Choices[0].Delta.Reasoning = schemas.Ptr("carefully")
	b.ChatResponse.Choices[0].Delta.ToolCalls = []schemas.ChatAssistantMessageToolCall{{Function: schemas.ChatAssistantMessageToolCallFunction{Arguments: `"Paris"}`}}}
	s.Observe(ctx, a, nil)
	s.Observe(ctx, b, nil)
	err := cancellation()
	s.Observe(ctx, nil, err)
	c, _ := selectCounter("")
	out, _ := c.count("hello world" + toolJSON("call_1", "weather", "function", `{"city":"Paris"}`))
	rn, _ := c.count("think carefully")
	got := err.ExtraFields.BilledUsage
	if got == nil || got.CompletionTokens != out+rn || got.CompletionTokensDetails.ReasoningTokens != rn {
		t.Fatalf("usage=%+v", got)
	}
}
func TestChunkBoundariesDoNotChangeCount(t *testing.T) {
	text := "hello world 👩‍💻 café 你好"
	s, ctx := testStream(t, schemas.OpenAI)
	for _, r := range text {
		s.Observe(ctx, textChunk(string(r)), nil)
	}
	err := cancellation()
	s.Observe(ctx, nil, err)
	c, _ := selectCounter("")
	want, _ := c.count(text)
	if err.ExtraFields.BilledUsage.CompletionTokens != want {
		t.Fatal("count depends on stream chunk boundaries")
	}
}
func BenchmarkObserveStreamChunk(b *testing.B) {
	for _, active := range []bool{false, true} {
		name := "inactive"
		if active {
			name = "active"
		}
		b.Run(name, func(b *testing.B) {
			chunk := textChunk(strings.Repeat("hello ", 8))
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			var s *Stream
			if active {
				s = (&Manager{}).Open(ctx, &schemas.BifrostRequest{RequestType: schemas.ChatCompletionStreamRequest}, schemas.OpenAI, "unknown")
				defer s.Close()
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				s.Observe(ctx, chunk, nil)
				if s != nil && i%128 == 127 {
					s.fallback = outputBuffer{}
				}
			}
		})
	}
}
func BenchmarkCancelledUsage(b *testing.B) {
	c, _ := selectCounter("")
	text := strings.Repeat("hello world ", 1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = c.count(text)
	}
}

func TestPythonTokenizerParity(t *testing.T) {
	data, err := os.ReadFile("testdata/tokenizers.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixtures []struct {
		Tokenizer string
		Text      string
		Tokens    int
	}
	if err = json.Unmarshal(data, &fixtures); err != nil {
		t.Fatal(err)
	}
	models := map[string]string{"cl100k_base": "unknown", "o200k_base": "gpt-4o", "anthropic": "claude-2", "deepseek": "deepseek-chat", "llama2": "llama-2", "llama3": "llama-3"}
	for _, f := range fixtures {
		t.Run(f.Tokenizer+"/"+f.Text[:min(len(f.Text), 20)], func(t *testing.T) {
			c, err := selectCounter(models[f.Tokenizer])
			if !strings.HasSuffix(f.Tokenizer, "_base") {
				Warm("anthropic", "deepseek", "llama2", "llama3")
				value, _ := counters.Load(f.Tokenizer)
				entry := value.(*counterEntry)
				c, err = entry.value, entry.err
			}
			if err != nil {
				t.Fatal(err)
			}
			if c.name != f.Tokenizer {
				t.Fatalf("expected %s, selected fallback %s", f.Tokenizer, c.name)
			}
			got, err := c.count(f.Text)
			if err != nil {
				t.Fatal(err)
			}
			if got != f.Tokens {
				t.Fatalf("got=%d Python=%d", got, f.Tokens)
			}
		})
	}
}

func TestResponsesCancelledUsage(t *testing.T) {
	s, ctx := testStream(t, schemas.OpenAI)
	for i, text := range []string{"hello ", "world"} {
		s.Observe(ctx, &schemas.BifrostResponse{ResponsesStreamResponse: &schemas.BifrostResponsesStreamResponse{Type: schemas.ResponsesStreamResponseTypeOutputTextDelta, SequenceNumber: i, Delta: schemas.Ptr(text)}}, nil)
	}
	err := cancellation()
	s.Observe(ctx, nil, err)
	if got := err.ExtraFields.BilledUsage; got == nil || got.CompletionTokens != 2 {
		t.Fatalf("usage=%+v", got)
	}
}
func TestInterruptedProviderCoverage(t *testing.T) {
	for _, provider := range []schemas.ModelProvider{schemas.OpenAI, schemas.Anthropic, schemas.Bedrock, schemas.Gemini, schemas.Vertex, schemas.Azure, schemas.Groq, schemas.Ollama, schemas.OpenRouter, schemas.Cohere, schemas.Mistral} {
		t.Run(string(provider), func(t *testing.T) {
			s, ctx := testStream(t, provider)
			s.Observe(ctx, textChunk("hello world"), nil)
			err := cancellation()
			s.Observe(ctx, nil, err)
			if got := err.ExtraFields.BilledUsage; got == nil || got.CompletionTokens != 2 || got.PromptTokens != 8 {
				t.Fatalf("usage=%+v", got)
			}
		})
	}
}
func TestMultipleChoicesAndLastPositiveUsage(t *testing.T) {
	s, ctx := testStream(t, schemas.OpenAI)
	a := textChunk("hello ")
	b := textChunk("world")
	b.ChatResponse.Choices[0].Index = 1
	s.Observe(ctx, a, nil)
	s.Observe(ctx, b, nil)
	err := cancellation()
	s.Observe(ctx, nil, err)
	if got := err.ExtraFields.BilledUsage; got == nil || got.CompletionTokens != 2 {
		t.Fatalf("usage=%+v", got)
	}
	s, ctx = testStream(t, schemas.OpenAI)
	for _, n := range []int{10, 8, 0} {
		chunk := textChunk("")
		chunk.ChatResponse.Usage = &schemas.BifrostLLMUsage{PromptTokens: 12, CompletionTokens: n}
		s.Observe(ctx, chunk, nil)
	}
	err = cancellation()
	s.Observe(ctx, nil, err)
	if got := err.ExtraFields.BilledUsage; got.CompletionTokens != 8 {
		t.Fatalf("expected last positive count, not maximum: %+v", got)
	}
}

// Run once in a fresh test process with -benchtime=1x to measure startup and
// retained memory for all four embedded vocabularies.
func BenchmarkTokenizerStartup(b *testing.B) {
	if b.N != 1 {
		b.Skip("run with -benchtime=1x in a fresh process")
	}
	rss := func() uint64 {
		data, err := os.ReadFile("/proc/self/statm")
		if err != nil {
			return 0
		}
		fields := strings.Fields(string(data))
		if len(fields) < 2 {
			return 0
		}
		n, _ := strconv.ParseUint(fields[1], 10, 64)
		return n * uint64(os.Getpagesize())
	}
	measure := func(prefix string, warm func()) {
		runtime.GC()
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		r0, cpu, start := rss(), processCPU(), time.Now()
		warm()
		elapsed, used := time.Since(start), processCPU()-cpu
		runtime.GC()
		runtime.ReadMemStats(&after)
		r1 := rss()
		b.ReportMetric(float64(elapsed.Nanoseconds()), prefix+"-wall-ns")
		b.ReportMetric(float64(used.Nanoseconds()), prefix+"-cpu-ns")
		if after.HeapAlloc > before.HeapAlloc {
			b.ReportMetric(float64(after.HeapAlloc-before.HeapAlloc), prefix+"-retained-Go-B")
		}
		if r1 > r0 {
			b.ReportMetric(float64(r1-r0), prefix+"-retained-RSS-B")
		}
	}
	b.ResetTimer()
	// The generic encodings load for every deployment; specialized is the extra.
	measure("generic", func() { Warm() })
	measure("specialized", func() { Warm("anthropic", "deepseek", "llama2", "llama3") })
	b.StopTimer()
}

func TestTokenizerSelectionIsOptIn(t *testing.T) {
	// Even a loaded tokenizer belonging to another gateway must remain unselected.
	Warm("deepseek")
	c, err := selectConfiguredCounter("deepseek-chat", nil)
	if err != nil || c.name != "cl100k_base" {
		t.Fatalf("generic selection = %s, %v", c.name, err)
	}
	c, err = selectConfiguredCounter("command-r", map[string]bool{"cohere": true})
	if err != nil || c.name != "cl100k_base" {
		t.Fatalf("Cohere must use generic fallback: %s, %v", c.name, err)
	}
	c, err = selectConfiguredCounter("deepseek-chat", map[string]bool{"deepseek": true})
	if err != nil || c.name != "deepseek" {
		t.Fatalf("explicit selection = %s, %v", c.name, err)
	}
}

// Every embedded vocabulary must be the exact upstream file the manifest pins.
func TestEmbeddedVocabulariesMatchManifest(t *testing.T) {
	data, err := os.ReadFile("assets/manifest.json")
	if err != nil {
		t.Fatal(err)
	}
	var manifest struct {
		Assets []struct{ Kind, SHA256 string }
	}
	if err = json.Unmarshal(data, &manifest); err != nil {
		t.Fatal(err)
	}
	embedded, err := vocabularies.ReadDir("assets")
	if err != nil {
		t.Fatal(err)
	}
	if len(embedded) != len(manifest.Assets) {
		t.Fatalf("embedded %d vocabularies, manifest lists %d", len(embedded), len(manifest.Assets))
	}
	for _, asset := range manifest.Assets {
		compressed, err := vocabularies.ReadFile("assets/" + asset.Kind + ".json.gz")
		if err != nil {
			t.Fatal(err)
		}
		raw, err := gunzip(compressed)
		if err != nil {
			t.Fatal(err)
		}
		if sum := sha256.Sum256(raw); hex.EncodeToString(sum[:]) != asset.SHA256 {
			t.Fatalf("%s vocabulary does not match manifest checksum", asset.Kind)
		}
	}
}

// A vocabulary using components the counter does not implement must fail to
// load, so selection falls back to tiktoken instead of miscounting.
func TestUnsupportedVocabularyComponentsRejected(t *testing.T) {
	base := `{"added_tokens":[],"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":["a b"]}`
	if _, err := parseBPE([]byte(base + `}`)); err != nil {
		t.Fatalf("minimal vocabulary: %v", err)
	}
	for name, extra := range map[string]string{
		"normalizer":     `,"normalizer":{"type":"Lowercase"}`,
		"pre-tokenizer":  `,"pre_tokenizer":{"type":"Whitespace"}`,
		"split behavior": `,"pre_tokenizer":{"type":"Split","pattern":{"Regex":"a"},"behavior":"Removed","invert":false}`,
		"post-processor": `,"post_processor":{"type":"RobertaProcessing"}`,
	} {
		if _, err := parseBPE([]byte(base + extra + `}`)); err == nil {
			t.Errorf("%s: expected rejection", name)
		}
	}
	if _, err := parseBPE([]byte(`{"model":{"type":"WordPiece","vocab":{}}}`)); err == nil {
		t.Error("WordPiece model: expected rejection")
	}
}

// Words longer than any merge chain exercise the heap-ordered merge loop.
func TestBPEMergeOrder(t *testing.T) {
	c, err := parseBPE([]byte(`{"model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2,"abab":3,"ba":4},"merges":["a b","ab ab","b a"]}}`))
	if err != nil {
		t.Fatal(err)
	}
	// "ab" outranks "ba", so "abab" becomes one token rather than a+ba+b.
	for text, want := range map[string]int{"": 0, "a": 1, "ab": 1, "aba": 2, "abab": 1, "ababab": 2, "bab": 2} {
		if got, err := c.Count(text); err != nil || got != want {
			t.Errorf("Count(%q) = %d, %v; want %d", text, got, err, want)
		}
	}
}

// specializedBenchmarks pairs each optional tokenizer family with a model that
// selects it. "generic" runs the same model with no families enabled, which is
// the default cl100k_base path every deployment already pays.
var specializedBenchmarks = []struct{ kind, model string }{
	{"anthropic", "claude-opus-4-5"},
	{"deepseek", "deepseek-chat"},
	{"llama2", "llama-2"},
	{"llama3", "llama-3"},
}

// benchmarkOutput builds deterministic mixed output (prose, code, JSON, CJK) of
// roughly size bytes, resembling what a cancelled stream has received.
func benchmarkOutput(size int) string {
	parts := []string{
		"The model streamed a detailed answer about distributed tracing and token budgets. ",
		"func handle(ctx context.Context) error {\n\treturn nil\n}\n",
		`{"city":"Paris","temperature":21.5,"tags":["a","b"]} `,
		"日本語のテキストと中文测试。 ",
		"It's 12345 o'clock; we'll retry 3 times.\n",
	}
	var b strings.Builder
	for i := 0; b.Len() < size; i++ {
		b.WriteString(parts[i%len(parts)])
	}
	return b.String()[:size]
}

// processCPU returns user+system CPU time consumed by this process, including
// GC workers, from /proc. It returns 0 where /proc is unavailable.
func processCPU() time.Duration {
	data, err := os.ReadFile("/proc/self/stat")
	if err != nil {
		return 0
	}
	fields := strings.Fields(string(data[strings.LastIndexByte(string(data), ')')+1:]))
	if len(fields) < 13 {
		return 0
	}
	utime, _ := strconv.ParseInt(fields[11], 10, 64)
	stime, _ := strconv.ParseInt(fields[12], 10, 64)
	return time.Duration(utime+stime) * 10 * time.Millisecond // USER_HZ is 100 on Linux
}

func reportCPU(b *testing.B, start time.Duration) {
	if cpu := processCPU() - start; cpu > 0 && b.N > 0 {
		b.ReportMetric(float64(cpu.Nanoseconds())/float64(b.N), "cpu-ns/op")
	}
}

func benchmarkCancelledStream(b *testing.B, model string, output string, tokenizers ...string) {
	request := &schemas.BifrostRequest{RequestType: schemas.ChatCompletionStreamRequest, ChatRequest: &schemas.BifrostChatRequest{Model: model, Input: []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr(benchmarkOutput(2048))}}}}}
	const chunk = 64
	chunks := make([]*schemas.BifrostResponse, 0, len(output)/chunk+1)
	for i := 0; i < len(output); i += chunk {
		chunks = append(chunks, textChunk(output[i:min(i+chunk, len(output))]))
	}
	m := &Manager{}
	ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
	run := func() *schemas.BifrostLLMUsage {
		s := m.Open(ctx, request, schemas.OpenAI, model, tokenizers...)
		for _, c := range chunks {
			s.Observe(ctx, c, nil)
		}
		failure := cancellation()
		s.Observe(ctx, nil, failure)
		s.Close()
		return failure.ExtraFields.BilledUsage
	}
	want := "cl100k_base"
	if len(tokenizers) > 0 {
		want = tokenizers[0]
	}
	if usage := run(); usage == nil || usage.Estimation == nil || usage.Estimation.Tokenizer != want {
		b.Fatalf("estimate did not use %s: %+v", want, usage)
	}
	cpu := processCPU()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		run()
	}
	b.StopTimer()
	reportCPU(b, cpu)
}

// BenchmarkCancelledStreamEstimate is the full per-cancellation cost (buffering
// every chunk, then prompt and completion counting) with each specialized
// tokenizer enabled against the generic default for the same model and output.
func BenchmarkCancelledStreamEstimate(b *testing.B) {
	Warm("anthropic", "deepseek", "llama2", "llama3")
	for _, size := range []int{1 << 10, 16 << 10, 128 << 10} {
		output := benchmarkOutput(size)
		for _, f := range specializedBenchmarks {
			b.Run(fmt.Sprintf("%dKB/%s/generic", size>>10, f.kind), func(b *testing.B) { benchmarkCancelledStream(b, f.model, output) })
			b.Run(fmt.Sprintf("%dKB/%s/specialized", size>>10, f.kind), func(b *testing.B) { benchmarkCancelledStream(b, f.model, output, f.kind) })
		}
	}
}

// BenchmarkCancelledStreamEstimateParallel shows throughput and CPU when many
// streams are cancelled at once and share one loaded tokenizer.
func BenchmarkCancelledStreamEstimateParallel(b *testing.B) {
	Warm("anthropic", "deepseek", "llama2", "llama3")
	output := benchmarkOutput(16 << 10)
	request := &schemas.BifrostRequest{RequestType: schemas.ChatCompletionStreamRequest, ChatRequest: &schemas.BifrostChatRequest{Input: []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr(benchmarkOutput(2048))}}}}}
	for _, f := range specializedBenchmarks {
		for _, mode := range []string{"generic", "specialized"} {
			var tokenizers []string
			if mode == "specialized" {
				tokenizers = []string{f.kind}
			}
			b.Run(f.kind+"/"+mode, func(b *testing.B) {
				m := &Manager{}
				cpu := processCPU()
				b.ReportAllocs()
				b.ResetTimer()
				b.RunParallel(func(pb *testing.PB) {
					ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
					chunk := textChunk(output)
					for pb.Next() {
						s := m.Open(ctx, request, schemas.OpenAI, f.model, tokenizers...)
						s.Observe(ctx, chunk, nil)
						s.Observe(ctx, nil, cancellation())
						s.Close()
					}
				})
				b.StopTimer()
				reportCPU(b, cpu)
			})
		}
	}
}

// BenchmarkStreamSuccessPath measures what enabling a tokenizer adds to a
// stream that is not cancelled: Open, 64 chunks and a normal Close. Selection
// only happens on interruption, so both modes should match.
func BenchmarkStreamSuccessPath(b *testing.B) {
	request := &schemas.BifrostRequest{RequestType: schemas.ChatCompletionStreamRequest, ChatRequest: &schemas.BifrostChatRequest{Input: []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hi")}}}}}
	chunk := textChunk("hello world ")
	for _, mode := range []string{"generic", "specialized"} {
		var tokenizers []string
		if mode == "specialized" {
			tokenizers = []string{"anthropic", "deepseek", "llama2", "llama3"}
		}
		b.Run(mode, func(b *testing.B) {
			m := &Manager{}
			ctx := schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				s := m.Open(ctx, request, schemas.OpenAI, "claude-opus-4-5", tokenizers...)
				for j := 0; j < 64; j++ {
					s.Observe(ctx, chunk, nil)
				}
				s.Close()
			}
		})
	}
}

// BenchmarkVocabularyLoad is the one-time cost of enabling a family: time, CPU,
// bytes allocated while loading, and Go heap retained by the loaded counter.
func BenchmarkVocabularyLoad(b *testing.B) {
	for _, f := range specializedBenchmarks {
		b.Run(f.kind, func(b *testing.B) {
			runtime.GC()
			var before, after runtime.MemStats
			runtime.ReadMemStats(&before)
			kept, err := loadVocabularyCounter(f.kind)
			if err != nil {
				b.Fatal(err)
			}
			runtime.GC()
			runtime.ReadMemStats(&after)
			runtime.KeepAlive(kept)
			cpu := processCPU()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := loadVocabularyCounter(f.kind); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			reportCPU(b, cpu)
			b.ReportMetric(float64(after.HeapAlloc-before.HeapAlloc), "retained-heap-B")
		})
	}
}
