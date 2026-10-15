# Research Claims Tracker

| Phase | Claim | Current evidence | Status |
|---|---|---|---|
| 3 | Green prompt engineering reduces energy/carbon through shorter, equally useful real-model outputs. | Round 3 (two pairs per shape) reduced mean provider completion tokens by 78.02% for focused explanation (1055.5 to 232) and 82.21% for the short how-to (1816 to 323). Resolved mean coverage fell from 5/5 to 4/5 and 7/7 to 6/7, respectively; both shapes are "shorter but less complete", so the preregistered two-shape qualification rule was not met. Energy deltas (-74.58%, -80.04%) are placeholder-coefficient proxy estimates shared with lca-demo, not measured or Google-published values. September estimates remain superseded. | Shorter outputs observed; equal usefulness and energy/carbon savings not established |

## Phase 3 validation state

The optimizer, A/B routing, audit logging, token accounting, and telemetry
plumbing are implemented and covered by tests. Gemini mode captures nonzero
provider `usage.prompt_tokens` and `usage.completion_tokens`; absent or zero
counts fall back to tagged `tiktoken` estimates. The provider's Gemini usage
example indicates that `completion_tokens` excludes hidden thinking, so
parametric decode energy uses the larger of visible completion and
`total_tokens - prompt_tokens`, logging visible and hidden counts separately.
Round 1 had n=1 per arm; both outputs were truncated at 512 on a prompt that
asks for extensive coverage while the optimizer asks for brevity. The length
instruction did not shorten output on that prompt at this cap; this did not
establish that the model ignored the optimizer. Round 2 had one qualifying
how-to pair, one unchanged-prompt A/A pair, and one shape truncated in both
arms; the corrected two-of-three threshold was not met. The variations
instruction increased the focused response in that single-shot test; any
benefit from avoiding follow-up prompts was not tested. The how-to's five
top-level bullets and 18 total bullet lines met the literal cap but not its
intent.

Round 3 subsequently ran eight requests at `max_tokens=2048`, with two
counterbalanced pairs for each of the focused explanation and how-to. Both
shapes had all four responses stop, optimizer changes on both optimized
prompts, and optimized mean word counts below the baseline mean minus the
baseline spread. After resolving independent scoring disagreements against the
frozen rubric, mean coverage was 5/5 baseline versus 4/5 optimized for focused
explanation, and 7/7 versus 6/7 for the how-to. Both shapes are **shorter but
less complete** and neither qualifies under the preregistered rule. The
focused-explanation coverage prediction did not hold; the how-to coverage
prediction did. Because both shapes must qualify, no larger run was triggered.

Pass 1 and independent pass 2 agreed on 43/48 decisions overall (89.58%):
18/20 (90%) for focused explanation and 25/28 (89.29%) for the how-to. The
scoring is rubric-bound, not blind, because response length and phrasing can
reveal the arm. Both score files, exact disagreement adjudications, quoted
evidence, item agreement, and resolved coverage totals are recorded in
[`verify/gemini_probe_round3_adjudications.json`](../verify/gemini_probe_round3_adjudications.json).

The round-3 energy deltas were -74.58% for focused explanation and -80.04% for
the how-to, but they are **placeholder-coefficient proxy estimates** using a
row copied from `lca-demo`, not model-specific or Google-published
measurements. They do not establish real-model energy/carbon savings; outputs
were also less complete on both shapes. The September Gemini figures
`0.000102263335 kWh` and `0.038811930061 gCO2e` remain superseded because they
were computed under the old NVML-tier accounting behavior.
