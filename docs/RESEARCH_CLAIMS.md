# Research Claims Tracker

| Phase | Claim | Current evidence | Status |
|---|---|---|---|
| 3 | Green prompt engineering reduces energy/carbon through shorter, equally useful real-model outputs. | Round 1 used n=1 per arm on a broad exhaustive prompt; both hit the 512-token cap. Round 2's corrected tally: how-to optimized completed at 768 tokens/429 words while baseline was truncated at >=1024 tokens/>=575 words (at least 25% fewer tokens and words); focused explanation was truncated in both arms; paragraph outputs were an unchanged-prompt A/A pair (~5.5% word-count noise), excluded from the optimizer tally. One qualifying shape is below the 2-shape threshold. V2 and an 8-request, quota-gated replication are prepared but not run. Energy uses a placeholder coefficient identical to lca-demo, so no causal energy claim follows. September estimates remain superseded. | Round 2 directional; replication pending |

## Phase 3 validation state

The optimizer, A/B routing, audit logging, token accounting, and telemetry
plumbing are implemented and covered by tests. Gemini mode captures nonzero
provider `usage.prompt_tokens` and `usage.completion_tokens`; absent or zero
counts fall back to tagged `tiktoken` estimates. The provider's Gemini usage
example indicates that `completion_tokens` excludes hidden thinking, so
parametric decode energy uses the larger of visible completion and
`total_tokens - prompt_tokens`, logging visible and hidden counts separately.
Round 1 had n=1 per arm; both outputs were truncated at 512 on a prompt that
asks for extensive coverage while the optimizer asks for brevity. The
length instruction did not shorten output on this prompt at this cap; the
result is inconclusive rather than evidence that the model ignores the
optimizer. In round 2, one how-to pair had at least 25% fewer tokens and words
in the complete optimized response (768 tokens/429 words) than the truncated
baseline (at least 1024 tokens/575 words). The focused pair was truncated in
both arms. The paragraph pair used identical effective prompts, so its
182-vs-172-word difference is A/A noise (~5.5%) and is excluded. The corrected
tally is one qualifying shape, one A/A, and one truncated in both; the
2-of-3 threshold was not met. The variations instruction increased the
focused response in this single-shot test; any benefit from avoiding
follow-up prompts was not tested. The how-to's five top-level bullets and 18
total bullet lines met the instruction's literal cap but not its intent.

Optimizer v2 is implemented offline: it adds a configurable word cap instead
of a five-bullet instruction and combines variations only when explicitly
enabled. An 8-request replication is pre-registered at 2048 output tokens,
with two counterbalanced pairs for each of the focused explanation and
how-to. Per-shape noise is the word-count spread between the two baseline
responses; the paragraph A/A request is dropped. A shape counts only if all
four outputs stop, both optimized prompts changed, optimized words beat the
baseline mean by more than baseline spread, and mean optimized coverage is
not lower. A shorter result that loses coverage is reported as "shorter but
less complete." A larger run is planned only if both shapes count. Run only
after a fresh quota check confirms at least 10 RPD remaining. The committed
rubric uses blind randomized response labels and per-element 0/1 scoring.
No requests were sent while preparing this plan.

The local vLLM service is disabled because the container has no GPU. Phase 3
should not close on mock-backend evidence alone. Gemini connectivity and
provider-reported usage are confirmed; thinking-budget consumption is
supported by the first controlled probe, and the explicit 512-token limit
explains the two longer truncations. The Gemini energy coefficients are an
explicit placeholder copied from `lca-demo`, not a model-specific
measurement. Real-model energy/carbon effects remain unvalidated.
