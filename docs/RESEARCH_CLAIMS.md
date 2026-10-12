# Research Claims Tracker

| Phase | Claim | Current evidence | Status |
|---|---|---|---|
| 3 | Green prompt engineering reduces energy/carbon through shorter, equally useful real-model outputs. | Three Gemini 2.5 Flash probes ran on 2026-10-12. The 256-token default-thinking call reported prompt=155, completion=9, total=407 and `finish_reason=length`, consistent with thinking tokens consuming most of the decode budget. Both 512-token `reasoning_effort=none` calls consumed all 512 completion tokens and ended at the limit. The optimized response failed the gate: no explicit expert persona and 14 bullet lines versus the five-bullet limit. No A/B was run. September estimates `0.000102263335 kWh` and `0.038811930061 gCO2e` remain superseded due to old NVML-tier accounting. | Probe completed; A/B blocked by compliance failure |

## Phase 3 validation state

The optimizer, A/B routing, audit logging, token accounting, and telemetry
plumbing are implemented and covered by tests. Gemini mode captures nonzero
provider `usage.prompt_tokens` and `usage.completion_tokens`; absent or zero
counts fall back to tagged `tiktoken` estimates. Three live probes establish
that the 256-token default-thinking request used nearly the full generation
budget while showing only nine completion tokens, and that the no-reasoning
512-token outputs still reached their explicit token ceiling. The optimized
response did not meet the persona/five-bullet compliance gate, so no
counterbalanced A/B was run and the causal energy/carbon claim remains
unvalidated.

The local vLLM service is disabled because the container has no GPU. Phase 3
should not close on mock-backend evidence alone. Gemini connectivity and
provider-reported usage are confirmed; thinking-budget consumption is
supported by the first controlled probe, and the explicit 512-token limit
explains the two longer truncations. The optimizer failed the compliance gate,
so its real-model energy/carbon effect remains unvalidated and the A/B must not
run until the prompt behavior is corrected or separately justified.
