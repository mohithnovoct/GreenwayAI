# Research Claims Tracker

| Phase | Claim | Current evidence | Status |
|---|---|---|---|
| 3 | Green prompt engineering reduces energy/carbon through shorter, equally useful real-model outputs. | Pipeline and carbon formula are verified; the adaptive mock's shorter-output result is scripted simulation only. Real-model output length, quality, and avoided re-prompts remain untested. | Partially validated; real-model validation blocked |

## Phase 3 validation state

The optimizer, A/B routing, audit logging, token accounting, and telemetry
plumbing are implemented and covered by tests. The corrected adaptive-mock run
used realistic prompts and counterbalanced 1000-request arms, but its 96-to-27
output-token reduction is hardcoded by `mock_backend.py` when an optimizer
constraint phrase is present. It must not be presented as evidence for the
paper's causal claim.

This environment has no configured hosted-model credential, and the local vLLM
service is disabled because the container has no GPU. A real-model test remains
open pending access to a small hosted endpoint or a GPU-backed local deployment.
Phase 3 should not close on mock-backend evidence alone.
