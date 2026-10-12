# Research Claims Tracker

| Phase | Claim | Current evidence | Status |
|---|---|---|---|
| 3 | Green prompt engineering reduces energy/carbon through shorter, equally useful real-model outputs. | September Gemini responses ended with `finish_reason=length`; the cause was not isolated, and no compliance-gated A/B was completed. The September estimates `0.000102263335 kWh` and `0.038811930061 gCO2e` are superseded because they used the old NVML-tier accounting behavior. On 2026-10-12 AI Studio showed Gemini 2.5 Flash at 0/20 RPD (28-day view; usage may lag up to 15 minutes), but this workspace has no configured API key, so no probe requests were sent. | Unvalidated; awaiting credentials and controlled probe |

## Phase 3 validation state

The optimizer, A/B routing, audit logging, token accounting, and telemetry
plumbing are implemented and covered by tests. Gemini mode captures nonzero
provider `usage.prompt_tokens` and `usage.completion_tokens`; absent or zero
counts fall back to tagged `tiktoken` estimates. A live controlled truncation
probe and a compliance-gated counterbalanced run are still required before
making the causal claim.

This environment has no configured hosted-model credential, and the local vLLM
service is disabled because the container has no GPU. Phase 3 should not close
on mock-backend evidence alone. Earlier Gemini calls established connectivity
but not the cause of truncation, optimizer compliance, or energy effect. The
current AI Studio quota display has headroom, but no new calls can be made until
a key is configured in this workspace.
