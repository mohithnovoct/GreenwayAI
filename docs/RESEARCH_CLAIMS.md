# Research Claims Tracker

| Phase | Claim | Current evidence | Status |
|---|---|---|---|
| 3 | Green prompt engineering reduces energy/carbon through shorter, equally useful real-model outputs. | Round 1 used n=1 per arm on a broad exhaustive prompt; both hit the 512-token cap. Round 2 completed all three counterbalanced prompt shapes at n=1 per arm. Only the one-paragraph shape had both arms complete and optimized text shorter (172 vs. 182 words), but the effective prompts were identical there. One qualifying shape is below the 2-of-3 threshold; no larger run is planned. Energy uses a placeholder coefficient identical to lca-demo, so no causal energy claim follows. September estimates remain superseded. | Directional round 2; no optimizer effect established |

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
optimizer. Round 2 completed n=1 per arm across three prompt shapes. Only one
shape produced shorter optimized text with both responses complete; in that
shape the effective optimized prompt was unchanged. This misses the
pre-registered 2-of-3 threshold and establishes no optimizer effect. A
task-scoped quantified word cap is proposed as a separately labeled optimizer
v2 experiment, not implemented.

The local vLLM service is disabled because the container has no GPU. Phase 3
should not close on mock-backend evidence alone. Gemini connectivity and
provider-reported usage are confirmed; thinking-budget consumption is
supported by the first controlled probe, and the explicit 512-token limit
explains the two longer truncations. The Gemini energy coefficients are an
explicit placeholder copied from `lca-demo`, not a model-specific
measurement. Real-model energy/carbon effects remain unvalidated.
