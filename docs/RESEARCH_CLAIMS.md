# Research Claims Tracker

| Phase | Claim | Current evidence | Status |
|---|---|---|---|
| 3 | Green prompt engineering reduces energy/carbon through shorter, equally useful real-model outputs. | Gemini returned real text and API usage fields, but the partial sample was truncated and the project dashboard shows Gemini 2.5 Flash at 21/20 RPD and 8/5 RPM. Compliance-gated 256-token A/B evidence could not be collected. | Partially validated; quota-blocked |

## Phase 3 validation state

The optimizer, A/B routing, audit logging, token accounting, and telemetry
plumbing are implemented and covered by tests. Gemini mode captures the API's
own `usage.prompt_tokens` and `usage.completion_tokens` rather than using the
local tokenizer estimate when those fields are present. A live counterbalanced
run with a compliance check is still required before making the causal claim.

This environment has no configured hosted-model credential, and the local vLLM
service is disabled because the container has no GPU. Phase 3 should not close
on mock-backend evidence alone. The current finding is that real Gemini calls
work and usage is captured, but the optimizer's persona/format compliance and
energy effect remain unvalidated because the actual project quota was
exhausted.
