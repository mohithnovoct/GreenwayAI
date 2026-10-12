# Research Claims Tracker

| Phase | Claim | Current evidence | Status |
|---|---|---|---|
| 3 | Green prompt engineering reduces energy/carbon through shorter, equally useful real-model outputs. | Round 1 used one sample per arm on a broad prompt demanding exhaustive coverage. Both baseline and optimized outputs reached the 512-token cap (`finish_reason=length`). The gateway log confirms the optimized prompt included the persona rule (`role-added`). Result: the length instruction did not shorten output on this prompt at this cap; this is not evidence that the model generally ignores the optimizer. Round 2 was aborted after an upstream 503 and a one-token diagnostic response; no matched samples were collected and its decision rule is unevaluated. A separately labeled v2 proposal is recorded in BENCHMARK.md. September estimates `0.000102263335 kWh` and `0.038811930061 gCO2e` remain superseded; the Gemini coefficient row is a placeholder identical to `lca-demo`. | Round 1 inconclusive; round 2 aborted |

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
optimizer. Round 2's saved three-shape test was not completed: an upstream
503 halted the runner, and the available request budget was preserved rather
than continuing with an incomplete set. A task-scoped quantified word cap is
proposed as a separately labeled optimizer v2 experiment, not implemented.

The local vLLM service is disabled because the container has no GPU. Phase 3
should not close on mock-backend evidence alone. Gemini connectivity and
provider-reported usage are confirmed; thinking-budget consumption is
supported by the first controlled probe, and the explicit 512-token limit
explains the two longer truncations. The Gemini energy coefficients are an
explicit placeholder copied from `lca-demo`, not a model-specific
measurement. Real-model energy/carbon effects remain unvalidated.
