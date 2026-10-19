# Ingress gateway tax

The `backend` column measures a direct HTTP request to `--backend`; the
`gateway` column measures the same request through ingress, including
normalization, hashing, tokenization, connection-pool reuse, and proxying.
Samples measure backend first and gateway second under the same concurrency
slot. `gateway tax` is the paired gateway-minus-backend latency distribution,
not a subtraction of unrelated percentiles.

The documented path uses the default Compose bridge (`greenwayai_default`),
with the gateway reaching the independent mock backend at
`mock-backend:8001`. This avoids the earlier host-published path mismatch.
The standard command is:

```bash
uv run python ingress/load_test.py --gateway http://localhost:8000 --backend http://localhost:8001 -n 1000 -c 50
```

Because `uv` was unavailable in the reproduction environment, the equivalent
benchmark client ran in the gateway image on the Compose network:

```bash
docker run --rm --network greenwayai_default greenwayai-gateway-test python ingress/load_test.py --gateway http://gateway:8000 --backend http://mock-backend:8001 -n 1000 -c 50
```

Hardware-backed NVML validation requires an NVIDIA deployment target. This
dev/CI environment is GPU-less, so figures collected here use clearly labeled
`mock_nvml` data and are synthetic development baselines, not GPU measurements.

The gateway now supports `GREENWAY_BACKEND_MODE=gemini`. It sends requests to
`https://generativelanguage.googleapis.com/v1beta/openai/chat/completions`,
uses `gemini-2.5-flash`, authenticates with `GEMINI_API_KEY`, and prefers the
provider's `usage.prompt_tokens` and `usage.completion_tokens` over the local
`tiktoken` estimate. The key must be obtained manually from [Google AI
Studio](https://aistudio.google.com/apikey) and placed only in the ignored
`.env` file.

If usage is absent or both provider token counts are zero (as with the mock
backend), the gateway estimates input and output tokens with `tiktoken`.
`X-Greenway-Token-Source` and the structured log's `token_count_source`
identify these counts as `tiktoken_estimated`; nonzero provider usage remains
tagged `provider`. In-process mock responses report null energy and carbon
with energy tier `unavailable`, rather than attributing mock sampler power to
the request.

Compose provides two A/B gateway instances with matching application settings
except for `PROMPT_OPT_ENABLED`: baseline is disabled at port 8000, optimized
is enabled at port 8002, and port 8001 remains the mock backend.

The current mock backend normally echoes the final input message, so its output
is not a generated answer. Its response length therefore follows the echoed
input text, but it has no quality-dependent generation or re-prompt behavior.

**Important: the adaptive mock result is scripted, not measured model
behavior.** When `GREENWAY_MOCK_ADAPTIVE_RESPONSES=true`, `mock_backend.py`
checks whether the final message contains the literal phrase `no more than 5
bullets` or `return only valid json`. If either phrase is present, it hardcodes
three canned bullets; otherwise it hardcodes eight canned bullets. This
deliberately assumes that an explicit constraint produces a shorter answer. It
does not detect answer quality, accuracy, or avoided re-prompts, and the
resulting 96-to-27 output-token drop is not evidence that a real model would
behave that way.

The original 100-request test used an 8-token prompt and expanded it to 29
tokens, so it could only measure added input overhead. The adaptive mode is a
controlled mechanism proxy, not a real-model experiment.

### Accounting and telemetry corrections

NVML attribution now divides each sampled power interval among all requests
active during that interval and prunes completed request/sample state. Remote
backends, including Gemini, automatically use parametric accounting with a
dedicated LCA-tier proxy row for `gemini-2.5-flash`; they do not use local NVML.
Provider-specific grid identifiers are translated from `GREENWAY_REGION`, and
CarbonEngine reuses one HTTP client for its cached provider refreshes. Full
prompt bodies are omitted from structured logs unless
`GREENWAY_LOG_PROMPTS=true`.

The NVML path now treats hardware-integrated joules as authoritative. It no
longer adds the token formula on top of measured energy; the token formula is
reserved for parametric/remote accounting. In the controlled regression case
with 100 input tokens, 50 output tokens, and a measured `0.000123 kWh`, the
previous stacked calculation would have been `0.000158 kWh` (`0.000035` token
estimate plus hardware energy). The corrected result is exactly:

```text
hardware_only=0.000123
stacked_before=0.000158
```

Backend URL classification now examines the host. `localhost`, loopback/private
addresses, and Docker service names remain NVML-eligible; dotted external
hosts force parametric accounting. Gemini remains explicitly remote.

The vLLM service remains intentionally disabled because this container has no
GPU. Hosted Gemini validation uses provider-reported tokens and proxy
coefficients, not local GPU telemetry.

## Benchmark History

| Date | Conditions | p50 tax | p95 tax | Outcome |
|---|---|---:|---:|---|
| 2026-08-24 | Warmed two-CPU mock baseline, 200 requests at concurrency 20 | 17.48 ms | 157.92 ms | Meets target |
| 2026-08-24 | Warmed two-CPU mock baseline, 1000 requests at concurrency 50 | 36.13 ms | 179.36 ms | Meets target |
| 2026-08-31 | Same 1000/50 workload; host contention, four-worker mock backend, mock/parametric telemetry | 265.35 ms | 1027.62 ms | Does not meet target; contention-sensitive |

The 2026-08-24 1000/50 baseline was measured after warm-up with an `httpx`
pool of 100 connections/keepalives, four mock-backend workers, and a 50-worker
gateway executor. The container reported two CPUs, so the explicit executor
avoids the six-worker default cap. Its complete paired result was:

```text
path       p50_ms   p95_ms   p99_ms
gateway     231.55   345.78   518.59
backend     194.93   282.10   343.32
gateway tax  36.13   179.36   334.55
```

The corresponding gateway component measurements were normalization
`3.617/19.095 ms`, pool acquire `0.000/0.000 ms`, HTTP round trip
`10.727/94.851 ms`, response read `1.560/6.803 ms`, and proxy total
`12.869/102.745 ms` at p50/p95. Pool acquisition was not the bottleneck.
The current implementation uses one lifespan-scoped pooled client, worker
thread offload for normalization/tokenization, eager fallback-tokenizer setup,
and the 50-worker executor; these changes removed the earlier multi-second
queueing and host/network artifact.

## Known Limitation: Benchmark Reproducibility

Absolute latency in this container varies with shared-host load. Do not compare
results across sessions without checking host state and re-establishing a
baseline under comparable conditions. The 2026-08-31 run demonstrates that
materially slower tax can occur without a code change, while pool wait remains
near zero.

For the Phase 2 Task 8 verification loop, add a relative criterion such as a
bounded gateway/backend latency ratio or maximum tax percentage. Both paths
share host pressure, so this may be more robust than an absolute threshold.
This is a proposed follow-up, not an implemented replacement for the current
acceptance metric.

## Phase target

Microsecond overhead is reasonable for compiled Go/Rust proxies but not for
this Python/FastAPI implementation. Accept a warmed p50 gateway tax below
50 ms and p95 below 200 ms on the documented two-CPU mock baseline, with no
multi-second event-loop queueing. Further optimization should be judged against
a real vLLM backend rather than a compiled-proxy target.

### Conclusion

The 2026-08-24 measurements met the gateway-tax target at both tested loads:
p50/p95 tax was `17.48/157.92 ms` at 200/20 and `36.13/179.36 ms` at 1000/50.
These are legitimate paired measurements under the documented low-contention
conditions. The 2026-08-31 reproduction failed at `265.35/1027.62 ms` under
host contention; matching behavior in mock-NVML and parametric modes attributes
that result to resource contention rather than a code regression. The target
has not been shown to pass consistently and must be re-measured under controlled
low-contention conditions before being cited as a general phase pass.

## Carbon Path Cost Analysis (Task 4)

Task 4 computes `carbon_gco2e` during request handling. The accepted evidence
is the 2026-09-03 counterbalanced comparison: 20 alternating rounds of 50
requests at concurrency 10, with enabled first on odd rounds and disabled first
on even rounds. Each arm received 1000 requests under the same host conditions.

| Arm | p50 tax (ms) | p95 tax (ms) | p99 tax (ms) |
|---|---:|---:|---:|
| enabled | 22.47 | 49.15 | 64.15 |
| disabled | 24.67 | 68.66 | 106.41 |

**Difference at p50: -2.20 ms** (enabled was slightly faster). Enabled p50
ranged from 19.48-29.95 ms and disabled from 20.42-34.01 ms; enabled was
slower in 5 of 20 paired rounds. The arms therefore overlap substantially and
the run does not support a measurable carbon-path penalty at this sample size.

### Conclusion

The counterbalanced run does **not** reproduce the earlier apparent carbon
cost. Under static-fallback MOER, mock NVML, and shared-container conditions,
enabled was 2.20 ms faster at p50, while disabled was slower at p95 and p99.
The difference is ordinary request variance and is not distinguishable from
noise without a larger controlled study. The earlier uncounterbalanced run
suggested a roughly 17 ms cost; the properly counterbalanced re-run found no
measurable cost and supersedes it.

The measured request-path call is
`request.app.state.carbon_engine.get_moer_with_tier()`. Background cache
refresh avoids blocking on every request, though startup misses and lock
contention can still add latency. No immediate optimization is required; the
path is A/B-disableable with `GREENWAY_CARBON_DISABLED=true` and should be
re-measured with live provider credentials, a real vLLM backend, and GPU
telemetry before production claims.

## Prompt optimization A/B measurement

Set `PROMPT_OPT_ENABLED=true` and choose the optimized traffic share with
`PROMPT_OPT_AB_PERCENT` (0-100). The same request path reports arm-specific
energy and carbon headers. To measure the delta on a sample, run:

```bash
PROMPT_OPT_ENABLED=true PROMPT_OPT_AB_PERCENT=50 GREENWAY_ENERGY_MODE=parametric \
uv run python ingress/load_test.py --gateway http://localhost:8000 --backend http://localhost:8001 \
-n 1000 -c 50 --ab-sample 1000
```

The command reports mean energy and carbon per request for each observed arm
and `optimized - baseline` deltas. A local 100-request run on 2026-09-03 used
the mock backend, parametric telemetry, and the 50% split (53 optimized / 47
baseline requests). It measured:

```text
metric       optimized          baseline            delta
energy_kwh   0.0000083           0.000002           +0.0000063
carbon_gco2e 0.003150093024      0.00075905856      +0.002391034464
```

This is measured simulated-backend evidence, not a real-model result or
production estimate. Re-run with a real model, controlled host conditions, and
hardware telemetry before using the delta as a deployment decision.

### Corrected realistic counterbalanced run

The 2026-09-03 follow-up used a 159-token sustainability-review prompt, the
same adaptive mock backend for both arms, parametric telemetry, and 20
counterbalanced rounds of 50 requests at concurrency 10 (1000 requests per
arm). The optimizer added role and length instructions, producing 180 input
tokens; the adaptive backend returned 96 output tokens for baseline and 27 for
optimized requests.

```text
arm          p50 tax   p95 tax   p99 tax   mean energy_kwh   mean carbon_gco2e
optimized     37.88     362.29     416.98       0.0000234       0.008880985152
baseline      24.49     354.94     411.07       0.0000351       0.013321477728
delta        +13.39      +7.35      +5.91      -0.0000117      -0.004440492576
```

The optimized arm used less modeled energy by `1.17e-05 kWh/request` and less
modeled carbon by `0.004440492576 gCO2e/request`, while p50 gateway tax was
`13.39 ms` higher. This demonstrates only the scripted shorter-output
assumption in the adaptive mock. It does not establish real-model answer
quality, accuracy, or avoided re-prompts. Phase 3 measurement remains open
pending a real model.

## Real-model evidence: Gemini 2.5 Flash

**Status: rounds 1-3 completed and analyzed.** Round-1 evidence, the round-2
corrected interpretation, all six round-2 responses, the round-3
pre-registration, and the round-3 adjudication are recorded below.

#### Round-2 protocol and quota record

The initial round-2 rule text required a shorter, complete optimized output on
at least two of three shapes. The retrospective correction treats a complete
optimized output as qualifying against a truncated baseline only when the
baseline's lower bound still establishes at least a 25% reduction. An
unchanged-prompt pair is A/A noise and is excluded. This yields one qualifying
shape, one A/A comparison, and one shape truncated in both arms; the
two-shape threshold is not met. Round 3 uses the stricter pre-registered rule
that both arms in both replicate pairs must finish with `stop`.
Round 2 uses `reasoning_effort=none`, `max_tokens=1024`, one baseline and one
optimized call per shape, counterbalanced order, and at least 16 seconds
between request starts. HTTP 429/500/503 responses are retried at the same
sample/arm position after 30 seconds, at most twice; every attempt and response
body is persisted before retrying. Set `--max-api-attempts` to the current
displayed RPD remaining minus the two-request reserve.
Per-attempt retry logs are written to
`verify/gemini_probe_round2_retry_attempts.json`; full successful samples are
written to `verify/gemini_probe_round2_results.json`.

Prompts are saved in
[`verify/gemini_probe_round2_prompts.json`](../verify/gemini_probe_round2_prompts.json)
and were measured at 96, 93, and 82 `tiktoken` tokens, respectively. Shapes:
focused explanation, short how-to, and one-paragraph analysis. The executable
runner is [`verify/gemini_probe_round2.py`](../verify/gemini_probe_round2.py);
it writes complete raw responses and per-sample metrics to
`verify/gemini_probe_round2_results.json`.

#### Round-2 quota and attempt record (2026-10-12)

Immediately before round 2, AI Studio displayed Gemini 2.5 Flash at `5 / 20`
RPD, i.e. 15 remaining. The runner's attempt budget was 13, preserving the
two-request reserve. All six planned requests succeeded on their first
attempt; no retry was needed. Their starts were sequential and more than 15
seconds apart (intervals approximately 22.5, 22.3, 21.2, 22.3, and 18.7
seconds). After the run, a refreshed dashboard displayed `11 / 20` RPD
(9 remaining); the dashboard therefore reflected all six additional
successful calls. Full response bodies and provider usage are saved in
[`verify/gemini_probe_round2_results.json`](../verify/gemini_probe_round2_results.json);
per-attempt status/body records are in
[`verify/gemini_probe_round2_retry_attempts.json`](../verify/gemini_probe_round2_retry_attempts.json).
The round-2 runner saved its scheduled arm and locally derived effective
prompt, but did not save the gateway's `prompt_optimization_arm` response
header or structured change list. The historical change lists are therefore
reconstructed from those saved prompts and the optimizer implementation at
commit `e2ef232`, not copied from raw gateway log records. The round-3 runner now persists the response arm and optimizer-change headers,
and compares the gateway's effective-prompt content hash with its local
preview before sending the next request.

All six samples report `reasoning_effort=none`, `max_tokens=1024`,
`token_count_source=provider`, `telemetry_source=parametric_estimated`,
`energy_source_tier=lca`, and zero hidden tokens:

| Sample | Shape | Arm | Usage (prompt/completion/total) | Finish | Hidden | Words | Top bullets | All bullet lines | Headings | Text count | Energy kWh | Carbon gCO2e |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| 1 | Focused explanation | Baseline | 92 / 1024 / 1116 | length | 0 | 639 | 19 | 19 | 4 | lower bound | 0.0007444 | 0.282521596032 |
| 2 | Focused explanation | Optimized | 110 / 1024 / 1134 | length | 0 | 665 | 2 | 8 | 3 | lower bound | 0.0007498 | 0.284571054144 |
| 3 | Short how-to | Optimized | 116 / 768 / 884 | stop | 0 | 429 | 5 | 18 | 0 | complete | 0.0005724 | 0.217242559872 |
| 4 | Short how-to | Baseline | 95 / 1024 / 1119 | length | 0 | 575 | 13 | 28 | 6 | lower bound | 0.0007453 | 0.282863172384 |
| 5 | One-paragraph analysis | Baseline | 86 / 220 / 306 | stop | 0 | 182 | 0 | 0 | 0 | complete | 0.0001798 | 0.068239364544 |
| 6 | One-paragraph analysis | Optimized | 86 / 209 / 295 | stop | 0 | 172 | 0 | 0 | 0 | complete | 0.0001721 | 0.065316989088 |

Across the three samples per arm, mean provider completion tokens were 756
(baseline) and 667 (optimized); mean words were 465.3 and 422.0. Mean proxy
energy was `0.0005565` vs. `0.0004981 kWh` (optimized minus baseline:
`-0.0000584 kWh`), and mean proxy carbon was `0.211208044320` vs.
`0.189043534368 gCO2e`. These aggregate figures include truncated responses
and are not a valid causal savings estimate.

The corrected tally is **one qualifying shape, one A/A comparison, and one
shape truncated in both arms**:

- **Focused explanation:** both responses hit the 1024-token cap, so this
  shape is inconclusive.
- **Short how-to:** optimized completed at 768 tokens and 429 words;
  baseline was truncated at at least 1024 tokens and at least 575 words.
  Optimized therefore used at least 25% fewer completion tokens and at least
  25% fewer words. It qualifies as the one shorter-output result under the
  corrected comparison rule. The optimized answer had 5 top-level bullets
  and 18 total bullet lines: it met the literal five-top-level-bullet
  instruction, but not its intent of a concise checklist.
- **One-paragraph analysis:** both effective prompts were unchanged and both
  responses completed (182 vs. 172 words). This is A/A noise, about 5.5%
  relative to the baseline, and is excluded from the optimizer-effect tally.

The variations rule increased output in the focused-explanation single-shot
response. Its intended benefit—avoiding follow-up prompts for separately
requested variants—was not tested here.

The effective focused-explanation prompt gained a persona instruction and an
unintended "provide all requested variations" instruction; the latter was
triggered by "one brief example." The optimized how-to prompt gained persona
and five-bullet instructions. The optimized one-paragraph prompt had no
changes because "as a short one-paragraph assessment" matched the optimizer's
existing role/length heuristics; it was not an optimizer treatment.

The decision threshold was at least two qualifying shapes; only one qualifies,
so the threshold was not met. Round 3 was then run as a separately
quota-gated experiment under the preregistration below.

All energy/carbon numbers above are proxy-coefficient estimates, not
Google-published measurements. The Gemini row is a placeholder shared with
`lca-demo`; observed energy deltas are not evidence of a causal energy saving.

#### Round-3 pre-registration (committed before collection)

Optimizer v2 replaces the default five-bullet instruction with a configurable
word cap (`PROMPT_OPT_WORD_CAP`, default `150`; the optimized instance will use
150). It adds `Answer in under N words.` when a numeric word cap is not already
present, even if the input only says "brief" or "short". Combining variations
is now opt-in through `PROMPT_OPT_COMBINE_VARIATIONS`; the default is off.
Compose pins `PROMPT_OPT_AB_PERCENT=100` on both gateway services, with prompt
optimization disabled on baseline and enabled on the optimized instance.

The coverage rubric is frozen in
[`verify/gemini_probe_round3_rubric.json`](../verify/gemini_probe_round3_rubric.json)
before any round-3 requests. The focused explanation has five binary items:
(1) explicitly defines the key as a unique request identifier for the same
logical operation across retries; (2) says the server stores the key associated
with the original result/response; (3) names limited retention, a duration, or
expiry; (4) says the same key reuses the original result without re-execution
and contrasts this with an unprotected duplicate side effect; and (5) gives a
concrete retry scenario that says the operation executes twice without an
idempotency key and explicitly says reusing the same key prevents the second
effect. A lost response is not required. The how-to has seven binary items:
(1) gives a selection rule/value for
both connect and read timeouts; (2) retries transient failures and excludes
ordinary client failures; (3) names exponential backoff and randomized jitter;
(4) names `Retry-After` and says to honor its value; (5) gives a finite attempt
count or overall time limit; (6) prohibits retrying a non-idempotent operation
unless the same idempotency key is used; and (7) names a measurable retry
signal. Each check is YES/NO (stored as 1/0); the exact criteria in the JSON
are controlling.

The quota-gated run has **eight** requests, all with
`reasoning_effort=none` and `max_tokens=2048`: two counterbalanced
baseline/optimized pairs for focused explanation and two for short how-to.
The paragraph A/A request has been dropped. For each shape, the baseline
noise is the absolute word-count spread between its two baseline responses.
The former 28-day dashboard reading is a peak-over-range value, not today's
usage, so it is not used as the quota gate. Before launch, select **1 Day** in
AI Studio and cross-check today's Pacific-date usage against the saved attempt
logs. Proceed only when this cross-check leaves at least **10 requests**
(eight planned requests plus two reserved). If timestamps are missing, count
those attempts against today conservatively.

Quota-gate amendment committed before collection (2026-10-12): at 03:01 PDT
on October 12, the AI Studio **1 Day** view showed Gemini 2.5 Flash RPD
`0 / 20`. The saved attempt logs contain seven timestamped requests on
2026-10-11 Pacific time and one 503 planned request with no timestamp. The six
round-two result records duplicate six entries in the retry-attempt log and
are not additional requests. Conservatively assigning the untimestamped 503
to October 12 gives at most one request used today and at least 19 remaining;
the timestamped log count for October 12 is zero. No round-three requests had
been sent at this point. This gate amendment is permissible because no
round-three responses existed.

Pre-registered decision rule: a shape counts only if all four responses finish
with `finish_reason=stop`, both optimized prompts have non-empty change
records, mean optimized word count is below the mean baseline count by more
than that shape's baseline spread, and mean optimized coverage score is not
below mean baseline coverage. If the first three conditions hold but coverage
is lower, report **"shorter but less complete"**; that is a task-scaled-cap
finding, not a success. Plan a larger run only if both shapes count.

Each response was scored twice against only the committed rubric: agent pass 1
and an independent pass 2 outside the repository using the single saved
response export, identified by sample number and containing no arm field.
Response length and wording can reveal an arm; therefore scoring is
**rubric-bound, not blind**. Raw per-item agreement over the four responses
for that shape/item is reported; Cohen's kappa is not used. Any pass
disagreement is resolved by quoting the response text for that item, without
consulting the arm assignment. The pass files and the exact resolution record
are saved separately in `verify/gemini_probe_round3_pass1.json`,
`verify/gemini_probe_round3_pass2.json`, and
[`verify/gemini_probe_round3_adjudications.json`](../verify/gemini_probe_round3_adjudications.json).

Predictions, recorded before the run: (a) focused explanation will pass the
coverage comparison at 150 words; (b) how-to will be shorter but fail coverage
at 150 words because seven distinct required elements compete for the cap.
If focused passes and how-to is shorter but less complete, design v3 with a
word cap that scales with the number of required elements. If both shapes
pass, plan a larger run with more pairs. If either shape is not complete,
changed, or shorter beyond baseline spread, retain the null/inconclusive
finding and revise the experiment only as a separately preregistered version.

`verify/gemini_probe_round3.py` saved the full responses, response headers,
results file, and sample-number-only response export; it did not score
responses during collection. The export has one record per response with only
`sample_number`, `prompt`, and `response_text`; it has no arm field. The run
results, attempt log, export, and both scoring passes are saved under
`verify/`. The run-start snapshot records the pinned rubric commit, HEAD, and
clean-tree state. Before collection, the runner verified that the rubric,
prompts, and optimizer were byte-identical to their pinned commit, and its
dry-run printed all eight assignments/settings/effective prompts without
contacting Gemini.

#### Round-3 results and score adjudication

The eight planned requests all completed with `finish_reason=stop`. The two
optimized prompts for each shape had non-empty change records. The primary
result is the provider completion-token delta:

| Shape | Baseline mean completion tokens | Optimized mean completion tokens | Delta | Baseline mean words (spread) | Optimized mean words |
|---|---:|---:|---:|---:|---:|
| Focused explanation | 1055.5 | 232 | -823.5 (-78.02%) | 722 (34) | 142.5 |
| Short how-to | 1816 | 323 | -1493 (-82.21%) | 1056 (168) | 134 |

The length-only rule passed for both shapes: optimized mean words were below
the baseline mean by more than the two-baseline word-count spread. Coverage did
not hold at the same cap:

| Shape | Baseline mean coverage | Optimized mean coverage | Preregistered outcome |
|---|---:|---:|---|
| Focused explanation (5 items) | 5/5 | 4/5 | shorter but less complete |
| Short how-to (7 items) | 7/7 | 6/7 | shorter but less complete |

Sensitivity, holding the final resolutions above fixed: under the original
`example` criterion requiring an explicit successful operation followed by a
lost/unreceived response, optimized sample 2 remains 3/5. Its `definition` and
`retry_vs_duplicate` items are NO; its example says the payment processor
confirms success before a network timeout, then describes the duplicate charge
without a key and reuse of the key preventing it. Optimized sample 3 becomes
4/5: its other four items are YES, but its example says the network times out
without explicitly stating that the first operation succeeded and its response
was not received. The sensitivity mean is therefore 3.5/5 (3/5 and 4/5),
not 3/5 for both samples. If instead only sample 8 `monitoring_signal` were
scored NO, the how-to means would be 6.5/7 baseline and 6/7 optimized. These
are counterfactual checks; the final scores and rule application above
continue to use the relaxed, preregistered rubric and the adjudicated YES for
sample 8.

The predictions were mixed: focused explanation was predicted to preserve
coverage, but it scored lower; the how-to was predicted to lose coverage, and
did. Neither shape meets the coverage condition, so neither qualifies and the
preregistered larger-run trigger (both shapes qualify) was not met. Both
results show a substantial length reduction at this 150-word instruction, but
not equally complete outputs.

Pass 1 and independent pass 2 agreed on 43/48 decisions (89.58%): 18/20
(90.00%) for focused explanation and 25/28 (89.29%) for the how-to. Per-item
agreement was focused: definition 3/4, stored data 4/4, retention 4/4,
retry-vs-duplicate 4/4, example 3/4; how-to: connect/read timeouts 4/4,
transient failures 2/4, backoff/jitter 4/4, Retry-After 4/4, attempt bound
4/4, non-idempotent safety 4/4, monitoring signal 3/4. All five
disagreements were adjudicated against the exact rubric and response text; the
evidence and final decisions are in the adjudication JSON.

The energy comparison is secondary and is a **placeholder-coefficient proxy
estimate**, not a Google-published figure or measured Gemini energy. The
focused-explanation means were 0.00076645 kWh baseline and 0.0001948 kWh
optimized (delta -0.00057165 kWh, -74.58%). The how-to means were 0.0012997
kWh and 0.0002594 kWh (delta -0.0010403 kWh, -80.04%). The coefficient row
is shared with `lca-demo`; these values do not establish a causal energy
saving. With two pairs per shape, this is directional evidence only and does
not establish statistical significance.

#### Superseded September telemetry

The September Gemini figures `0.000102263335 kWh` and
`0.038811930061 gCO2e` are **superseded and must not be used**. They were
computed under the old NVML-tier accounting behavior and do not represent the
current Gemini LCA-tier parametric coefficient estimate. The configured
`gemini-2.5-flash` row uses the same `3e-7` prefill and `7e-7` decode
coefficients as `lca-demo`; it is a placeholder, not a Gemini-specific
measurement or Google-published figure.

#### Task 3 reasoning-token design note

For Gemini 2.5 Flash, the observed `usage.completion_tokens` count excludes
hidden thinking tokens: in the 256-token probe, `total_tokens - prompt_tokens`
was 252 while `completion_tokens` was 9. For parametric energy accounting,
the gateway therefore uses
`decode_tokens = max(completion_tokens, total_tokens - prompt_tokens)`.
It keeps `output_tokens_visible` and `output_tokens_hidden` distinct in logs
while preserving `token_count_source=provider`.

#### Truncation hypothesis and round-1 result

Earlier partial Gemini samples ended with `finish_reason=length`; their short
visible text alone does not establish why. The previous explanation that
Gemini's default output limit caused the short responses was unsupported:
thinking-token consumption, output limits, and other causes were not isolated.
The controlled probe supports the thinking-token explanation for the
256-token response: provider totals imply 252 generated tokens although only
9 are visible as completion tokens. The raw usage object does not expose a
separate thought-token detail. Raising the limit to 512 with
`reasoning_effort="none"` still ends at the 512-token output ceiling, so those
longer responses are explained by that configured limit.

#### Controlled Gemini probe (2026-10-12)

Immediately before the probe, the selected free-tier Gemini 2.5 Flash row in
AI Studio showed `0 / 5` RPM, `0 / 250K` TPM, and `0 / 20` RPD (28-day view).
Exactly three requests were sent, sequentially, with at least 15 seconds
between calls. The local `.env` key was used without printing it. Both gateway
instances were healthy; their effective environments differed only in
`PROMPT_OPT_ENABLED`.

The same 159-token user prompt was submitted for all three calls:

```text
I am preparing a quarterly sustainability review for a cloud-based language model service. Analyze the following operational scenario and produce practical recommendations for the engineering and finance teams. The service receives customer support questions, incident reports, and documentation requests from several regions. We need to understand which changes could reduce unnecessary model work without reducing answer quality or omitting important caveats. Compare response strategies, identify where repeated requests or ambiguous terminology may create avoidable work, and explain how an engineering team could validate improvements using production-like traffic. Include assumptions, risks, and measurable success criteria. Discuss how the recommendations affect latency, energy use, carbon reporting, and user satisfaction. The audience understands basic APIs and observability but may not know specialized model-serving terminology, so define abbreviations the first time you use them.
```

For the optimized arm, the gateway's optimizer added the role and concise
five-bullet instruction (logged changes: `role-added`, `length-constrained`).
The effective optimized user prompt was:

```text
Act as a precise domain expert. I am preparing a quarterly sustainability review for a cloud-based language model service. Analyze the following operational scenario and produce practical recommendations for the engineering and finance teams. The service receives customer support questions, incident reports, and documentation requests from several regions. We need to understand which changes could reduce unnecessary model work without reducing answer quality or omitting important caveats. Compare response strategies, identify where repeated requests or ambiguous terminology may create avoidable work, and explain how an engineering team could validate improvements using production-like traffic. Include assumptions, risks, and measurable success criteria. Discuss how the recommendations affect latency, energy use, carbon reporting, and user satisfaction. The audience understands basic APIs and observability but may not know specialized model-serving terminology, so define abbreviations the first time you use them. Keep the response concise and organized into no more than 5 bullets.
```

**Sample 1 — baseline, default thinking, `max_tokens=256`**

Raw response text:

```text
Okay, this is a critical exercise for any
```

Raw usage: `{"completion_tokens":9,"prompt_tokens":155,"total_tokens":407}`

`finish_reason`: `length`. The total usage minus prompt and visible completion
is `407 - 155 - 9 = 243` tokens. Prompt plus all generated tokens is
`155 + 9 + 243 = 407`; the generated count is 252, just below the 256 limit.
This strongly supports the hypothesis that thinking tokens consumed most of
the generation budget without appearing in `completion_tokens`. Google did
not provide a separate thought-token field in this response.

**Sample 2 — baseline, `reasoning_effort=none`, `max_tokens=512`**

Raw response text:

```text
## Quarterly Sustainability Review: Recommendations for Cloud-based Language Model Service

This document analyzes operational scenarios for our cloud-based language model service to provide practical recommendations for engineering and finance teams. The goal is to reduce unnecessary model work without sacrificing answer quality or omitting important caveats, focusing on customer support questions, incident reports, and documentation requests across various regions.

### Operational Scenario Analysis: Unnecessary Model Work Drivers

Our language model (LLM) service processes a high volume of requests daily. While the LLM is crucial for providing accurate and timely responses, certain patterns indicate avoidable work. These include:

*   **Repeated Requests:** Users often submit the same or very similar queries multiple times due to impatience, network issues, or a lack of immediate confirmation.
*   **Ambiguous Terminology/Lack of Specificity:** Vague or overly broad questions require the LLM to spend more computational resources trying to infer intent or explore a wider range of possible answers, which it might then discard.
*   **Sequential Problem-Solving:** Users sometimes break down a single complex problem into multiple simpler questions, submitting them one after another, leading to redundant context processing by the LLM.
*   **Pre-existing Answers:** Questions for which answers already exist in readily accessible static documentation (e.g., FAQs, knowledge base articles) without requiring LLM generation.
*   **Out-of-Scope Requests:** Questions unrelated to the service's domain or requiring human intervention (e.g., account-specific issues, direct access requests).

### Recommendations for Engineering and Finance Teams

To address these drivers, we propose a multi-pronged approach involving strategic pre-processing, intelligent routing, and feedback mechanisms.

#### Proposed Response Strategies & Engineering Interventions

**1. Smart Request Pre-processing and Semantic Deduplication (Engineering Lead)**

*   **Strategy:** Implement a lightweight pre-processing layer before passing requests to the main LLM. This layer would
    *   **Normalize Text:** Convert text to lowercase, remove common stop words (e.g., "a," "an," "the"), correct minor typos, and standardize abbreviations.
    *   **Semantic Hashing/Embedding:** Generate a compact numerical representation (embedding) of the user query. This can be done using a smaller, more efficient embedding model than the main generative LLM.
    *   **Deduplication Cache (Cache-Aside Pattern):** Store recent queries and their generated responses in a temporary cache (e.g.,
```

Raw usage: `{"completion_tokens":512,"prompt_tokens":155,"total_tokens":667}`

`finish_reason`: `length`. The response used all 512 allowed completion tokens
and is truncated mid-sentence.

**Sample 3 — optimized, `reasoning_effort=none`, `max_tokens=512`**

Raw response text:

```text
Here's an analysis and recommendations for reducing unnecessary model work in your cloud-based language model (LLM) service, balancing efficiency with quality:

*   **Implement a Tiered Triage and Intent Classification System:**
    *   **Recommendation:** For your engineering team, integrate a lightweight, pre-processing layer *before* routing all requests to the primary, more resource-intensive LLM. This layer would use a small, specialized, and highly efficient **Natural Language Processing (NLP)** model or rule-based system for initial triage and intent classification (e.g., "password reset," "billing inquiry," "bug report," "documentation").
    *   **Impact/Details:**
        *   **Latency:** Marginally increases initial processing time (negligible for a small model), but significantly reduces overall processing time for many requests by avoiding the main LLM.
        *   **Energy Use/Carbon Reporting:** Substantially reduces energy consumption and associated carbon emissions by offloading simpler, frequently asked questions (FAQs) or easily categorized requests from the main LLM.
        *   **User Satisfaction:** Improves by routing specific queries to dedicated, optimized workflows (e.g., direct link to password reset, canned responses for known issues), potentially providing faster, more targeted initial responses without invoking the full model unnecessarily. High-quality answers are preserved for complex queries.
        *   **Assumptions:** A significant portion of incoming requests are repetitive, low-complexity, or fall into predictable categories.
        *   **Risks:** Initial misclassification could delay resolution or provide an irrelevant response for a small percentage of requests, requiring careful tuning.
        *   **Validation (Engineering):** Use A/B testing or shadow deployment with production-like traffic. Measure the percentage of requests successfully handled by the triage layer without invoking the main LLM, and the downstream error rate. Compare the latency of classified vs. unclassified requests.
        *   **Success Criteria:**
            *   **Direct-to-Answer Rate:** ≥ 30% of requests handled without full LLM invocation.
            *   **Classification Accuracy:** ≥ 95% intent classification accuracy.
            *   **Main LLM Latency Reduction:** Average latency for main LLM invocations decreases by ≥ 10%.
    *   **Finance Integration:** This allows for a more granular understanding of model usage costs per request type, highlighting cost savings from reducing full LLM invocations. Resource allocation can
```

Raw usage: `{"completion_tokens":512,"prompt_tokens":176,"total_tokens":688}`

`finish_reason`: `length`. The optimized response also used its full 512-token
completion limit. The gateway log shows the optimizer applied its persona rule
(`prompt_optimization_changes` includes `role-added`); the saved effective
prompt begins "Act as a precise domain expert." The response body itself is
not used to infer whether the persona instruction was included.

#### Offline round-1 response metrics

Counts are from the saved response text. Both 512-token responses were
truncated, so their word, bullet, and heading counts are lower bounds. The
effective optimized prompt contains the role instruction as verified by the
gateway's `role-added` optimization log; persona wording in generated prose is
not scored.

| Sample | Arm | finish_reason | completion tokens | Words | Top-level bullets | All bullet lines | Headings | Counts |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 2 | baseline | length | 512 | 351 | 6 | 9 | 4 | lower bounds |
| 3 | optimized | length | 512 | 325 | 1 | 14 | 0 | lower bounds |

The single sample per arm does not establish an output-length effect. Both
responses were truncated at 512, and the prompt demands broad, exhaustive
coverage while the optimizer adds a brevity constraint. Conclusion: **the
length instruction did not shorten output on this prompt at this cap**. This
does not show that the model generally ignores the optimizer.

The following gateway records were captured before the hidden-token accounting
fix. In sample 1, the raw log's `5.28e-05 kWh` and `0.020039145984 gCO2e`
under-count decode work because they used only the 9 visible completion
tokens. Recalculating from the same provider usage with 252 decode tokens gives
`0.0002229 kWh` and `0.084597076512 gCO2e` at the same coefficients and MOER.
Samples 2 and 3 have no inferred hidden tokens, so their logged energy values
remain unchanged. The regression test now checks both hand calculations.

All three gateway records reported `token_count_source=provider`,
`telemetry_source=parametric_estimated`, and `energy_source_tier=lca` (not
`nvml` or `mock_nvml`). Energy and carbon figures are proxy-coefficient
estimates, not Google-published figures. No A/B aggregates or statistical
claims are reported.

Raw gateway log records:

```json
{"request_id": "542019fe-5f9c-4147-9aff-8a29e381c4ce", "content_hash": "2abee62e76c0cc8d422cc680b29a7c2dcca7a1aa93371d1667e905f72488a41a", "input_tokens": 155, "prompt_optimization_arm": "baseline", "token_count_source": "provider", "original_input_tokens": 159, "optimized_input_tokens": 159, "prompt_optimization_changes": [], "output_tokens": 9, "energy_kwh": 5.28e-05, "carbon_gco2e": 0.020039145984, "moer_kg_per_kwh": 0.31, "carbon_source_tier": "static_fallback", "normalization_ms": 14.201, "proxy_ms": 2726.282, "pool_acquire_ms": 1.483, "round_trip_ms": 2722.351, "response_read_ms": 2.008, "latency_ms": 2742.48, "telemetry_source": "parametric_estimated", "energy_source_tier": "lca"}
{"request_id": "1d89267d-b57d-46d1-84d3-b064231edac1", "content_hash": "2abee62e76c0cc8d422cc680b29a7c2dcca7a1aa93371d1667e905f72488a41a", "input_tokens": 155, "prompt_optimization_arm": "baseline", "token_count_source": "provider", "original_input_tokens": 159, "optimized_input_tokens": 159, "prompt_optimization_changes": [], "output_tokens": 512, "energy_kwh": 0.0004049, "carbon_gco2e": 0.153671405472, "moer_kg_per_kwh": 0.31, "carbon_source_tier": "static_fallback", "normalization_ms": 0.717, "proxy_ms": 3715.093, "pool_acquire_ms": 1.426, "round_trip_ms": 3711.578, "response_read_ms": 1.129, "latency_ms": 3716.838, "telemetry_source": "parametric_estimated", "energy_source_tier": "lca"}
{"request_id": "60fc5d14-b12c-4bc8-b6f8-671bc802e581", "content_hash": "52277fb8ea91e32cf560e79d838bad0618e754ca6f4928d25855dc9a40c5ed31", "input_tokens": 176, "prompt_optimization_arm": "optimized", "token_count_source": "provider", "original_input_tokens": 159, "optimized_input_tokens": 180, "prompt_optimization_changes": ["role-added", "length-constrained"], "output_tokens": 512, "energy_kwh": 0.0004112, "carbon_gco2e": 0.156062439936, "moer_kg_per_kwh": 0.31, "carbon_source_tier": "static_fallback", "normalization_ms": 10.571, "proxy_ms": 3841.224, "pool_acquire_ms": 0.785, "round_trip_ms": 3837.514, "response_read_ms": 1.317, "latency_ms": 3853.082, "telemetry_source": "parametric_estimated", "energy_source_tier": "lca"}
```

## Correctness and failure behavior

The carbon result uses the auditable formula:

```text
carbon_gco2e = energy_kwh * raw_moer_kg_per_kwh * 1.04 * 1.08 * provider_pue * 1000
```

The multiplier chain is covered by
`tests/test_carbon_engine.py::test_apply_infrastructure_multipliers_chain_matches_exact_value`
and the end-to-end hand calculations in the same test module. Provider
selection and static fallback are also covered there. Fail-open behavior is
covered by
`tests/test_telemetry_baseline.py::test_gateway_fails_open_when_all_carbon_providers_raise`
and
`tests/test_telemetry_baseline.py::test_gateway_fails_open_when_nvml_raises_mid_request`:
the request remains HTTP 200 and unavailable energy/carbon is emitted as
`null` with an `unavailable` source rather than failing the inference request.

The counterbalanced raw evidence below is retained because it is the current,
valid evidence for Task 4's performance question.

## Appendix: Counterbalanced raw per-round evidence (2026-09-03)

Each round contained one 50-request batch per arm at concurrency 10. Odd rounds
ran enabled -> disabled; even rounds ran disabled -> enabled. Resource readings
were captured immediately before each batch.

| Round | Arm order | Resource state before enabled | Enabled p50 (ms) | Resource state before disabled | Disabled p50 (ms) |
|---:|---|---|---:|---|---:|
| 1 | enabled -> disabled | load=1.40, mem=12731.0 MiB | 29.95 | load=1.40, mem=12728.7 MiB | 34.01 |
| 2 | disabled -> enabled | load=1.40, mem=12724.2 MiB | 21.15 | load=1.40, mem=12724.0 MiB | 26.37 |
| 3 | enabled -> disabled | load=1.40, mem=12724.1 MiB | 27.35 | load=1.40, mem=12721.4 MiB | 24.73 |
| 4 | disabled -> enabled | load=1.52, mem=12722.7 MiB | 20.60 | load=1.40, mem=12721.5 MiB | 23.84 |
| 5 | enabled -> disabled | load=1.52, mem=12722.7 MiB | 22.53 | load=1.52, mem=12722.4 MiB | 23.17 |
| 6 | disabled -> enabled | load=1.52, mem=12732.5 MiB | 22.37 | load=1.52, mem=12722.4 MiB | 24.11 |
| 7 | enabled -> disabled | load=1.52, mem=12731.2 MiB | 24.60 | load=1.52, mem=12731.7 MiB | 27.87 |
| 8 | disabled -> enabled | load=1.52, mem=12717.1 MiB | 22.06 | load=1.52, mem=12717.5 MiB | 24.11 |
| 9 | enabled -> disabled | load=1.52, mem=12717.3 MiB | 22.17 | load=1.52, mem=12717.3 MiB | 25.53 |
| 10 | disabled -> enabled | load=1.52, mem=12713.7 MiB | 20.68 | load=1.52, mem=12714.8 MiB | 29.10 |
| 11 | enabled -> disabled | load=1.52, mem=12713.5 MiB | 26.13 | load=1.52, mem=12731.8 MiB | 20.65 |
| 12 | disabled -> enabled | load=1.52, mem=12734.9 MiB | 22.96 | load=1.52, mem=12733.0 MiB | 23.67 |
| 13 | enabled -> disabled | load=1.88, mem=12734.7 MiB | 19.53 | load=1.88, mem=12736.8 MiB | 20.86 |
| 14 | disabled -> enabled | load=1.88, mem=12741.0 MiB | 22.06 | load=1.88, mem=12736.8 MiB | 30.60 |
| 15 | enabled -> disabled | load=1.88, mem=12743.0 MiB | 20.12 | load=1.88, mem=12743.1 MiB | 23.52 |
| 16 | disabled -> enabled | load=1.88, mem=12745.2 MiB | 23.69 | load=1.88, mem=12745.2 MiB | 20.42 |
| 17 | enabled -> disabled | load=1.88, mem=12747.4 MiB | 22.42 | load=1.88, mem=12747.4 MiB | 32.64 |
| 18 | disabled -> enabled | load=1.88, mem=12749.4 MiB | 19.48 | load=1.88, mem=12748.9 MiB | 20.73 |
| 19 | enabled -> disabled | load=1.88, mem=12757.2 MiB | 24.73 | load=1.88, mem=12757.6 MiB | 21.50 |
| 20 | disabled -> enabled | load=1.88, mem=12759.8 MiB | 23.30 | load=1.88, mem=12757.6 MiB | 22.28 |

## Phase 4 pre-flight: embedding latency

Before measuring, the preregistered workload is one cold model load followed
by 5 warm-up embeddings and 100 sequential measured prompts. The exact prompt
template and runner are in
[`verify/benchmark_semantic_embedding.py`](../verify/benchmark_semantic_embedding.py).
Each generated prompt must be 90-110 tokens according to the selected model's
tokenizer; otherwise the runner refuses to produce latency results. Each
measured embedding call is timed end-to-end through `asyncio.to_thread`.
Report all 100 durations and nearest-rank p50/p95; there is no pass/fail
latency threshold because this is a cost characterization, not a cache
correctness decision. The benchmark container is limited to 2 CPUs and
PyTorch intra-op/inter-op threads are each set to 2. Capture host load
immediately before this batch.

The first workload-template check reported 139-140 model tokens; the runner
refused to enter warm-up or timing. The template was shortened before any
latency sample was collected. The committed template now validates at 104-105
tokens per prompt.

The selected model is `sentence-transformers/all-MiniLM-L6-v2`, pinned to
revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Its Hugging Face model
card lists Apache-2.0; the `model.safetensors` artifact is 90,868,376 bytes
(approximately 86.6 MiB), excluding the small tokenizer/config files. The
gateway loads the model during startup only when `CACHE_ENABLED=true`, and
logs the model ID, revision, dimensions, and CPU device before serving
requests. The RediSearch module bundled in
`redis/redis-stack-server:7.4.0-v2` is distributed under Redis's pre-Redis-8
source-available terms, RSALv2 or SSPLv1 (see the
[RediSearch license](https://github.com/RediSearch/RediSearch/blob/master/LICENSE.txt)).
Docker Hub reports that this image tag was last pushed on 2025-01-08, about
649 days before this 2026-10-19 check
([tag metadata](https://hub.docker.com/v2/repositories/redis/redis-stack-server/tags/7.4.0-v2)).
Its age and source-available licensing make a newer image or alternative
worth evaluating in a later review; this pre-flight does not switch images.
The stack image selected for vector search is pinned to
`redis/redis-stack-server:7.4.0-v2`.

### Task 1 pre-flight results (2026-10-19)

The original Compose image, `redis:7-alpine`, did not include vector search.
Raw command replies:

```text
FT._LIST:
ERR unknown command 'FT._LIST', with args beginning with:
FT.CREATE gw_phase4_control ON HASH PREFIX 1 gw-phase4-control: SCHEMA embedding VECTOR HNSW 6 TYPE FLOAT32 DIM 384 DISTANCE_METRIC COSINE:
ERR unknown command 'FT.CREATE', with args beginning with: 'gw_phase4_control' 'ON' 'HASH' 'PREFIX' '1' 'gw-phase4-control:' 'SCHEMA' 'embedding' 'VECTOR' 'HNSW' '6' 'TYPE' 'FLOAT32' 'DIM'
```

Compose now pins `redis/redis-stack-server:7.4.0-v2`
(`redis_version:7.4.2`,
`redis/redis-stack-server@sha256:df62866294f7fab407fa1b88b9c1eceb53d2b99c571196ec2de621b166107858`).
The throwaway control then passed and was removed:

```text
FT._LIST before:
(empty)
FT.CREATE:
OK
FT._LIST after:
gw_phase4_control
FT.DROPINDEX:
OK
```

The selected model card identifies the license as Apache-2.0. Its default
`model.safetensors` artifact is 90,868,376 bytes (86.6 MiB); tokenizer and
configuration files add a small amount. The model card is
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
and the model artifact list is available from the
[Hugging Face model API](https://huggingface.co/api/models/sentence-transformers/all-MiniLM-L6-v2/tree/main?recursive=true&expand=true).

Actual gateway startup with `CACHE_ENABLED=true` logged:

```text
semantic embedding model loading model_id=sentence-transformers/all-MiniLM-L6-v2 revision=1110a243fdf4706b3f48f1d95db1a4f5529b4d41 device=cpu
semantic embedding model loaded model_id=sentence-transformers/all-MiniLM-L6-v2 revision=1110a243fdf4706b3f48f1d95db1a4f5529b4d41 dimensions=384 device=cpu
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

`GET /health` returned HTTP 200 after startup. The latency run used
`--cpus=2`, with two PyTorch intra-op and inter-op threads. Immediately before
the batch, Windows host CPU total was 15.95% at
`2026-10-19T10:41:02.9807046+05:30`; there were five warm-up calls followed by
100 measured prompts of 104-105 model tokens (mean 104.97), each executed via
`asyncio.to_thread`. Raw per-call timings and run metadata are committed in
[`verify/phase4_embedding_benchmark_results.json`](../verify/phase4_embedding_benchmark_results.json).
Nearest-rank p50 was **29.764186 ms** and p95 was **34.756024 ms**. This is
the CPU embedding cost only; it excludes Redis search and is not cache hit
latency.

The full suite at this gate reported `67 passed, 1 warning`. The warning was
the existing Starlette/httpx `BlockingPortal` deprecation.

### Phase 4 follow-up design constraints

Task 2 exact-tier policy: use
`greenwayai:cache:v1:<namespace-sha256>:exact:<request-sha256>`. The request
digest covers sorted canonical JSON for the full effective request after
backend model selection plus the backend endpoint identity, so messages,
model, endpoint, and every supplied generation/output parameter participate;
request IDs and transport metadata do not. The namespace is
deployment-configured and hashed in the key; caller
headers are not treated as tenant identity. The initial namespace is
`greenwayai-single-tenant`, explicitly assuming one trusted deployment tenant.
The default TTL is 3,600 seconds. Only non-streaming, single-choice,
tool/function-free requests with `temperature=0` are lookup-eligible. Store
only HTTP 200 JSON responses with one assistant choice ending in
`finish_reason=stop` and no tool call or refusal. Redis errors fail open to the
backend. `X-Greenway-Cache-Bypass: true` skips both lookup and storage.
`X-Greenway-Cache-Tier` and the `cache_tier` telemetry attribute report one of
`exact`, `semantic`, `miss`, `bypass`, or `unavailable`. Lookup order is exact,
then semantic, then backend. Semantic hits log their cosine similarity.
The cache is shared among every caller of one gateway deployment. This is
unsafe for multi-tenant deployments because one caller could receive another
caller's cached content. A multi-tenant deployment must authenticate callers
and have trusted authentication middleware set
`request.state.authenticated_tenant_id`; the optional tenant key component is
empty unless that state value is present. The gateway never derives tenant
identity from a request header. Cache keys include the optimizer arm and hash
the effective request after prompt optimization, preventing optimized and
baseline entries from sharing a key. Cached response bodies are plaintext user
content in Redis for up to 3,600 seconds; protect the Redis network and storage
accordingly.
The gateway now performs exact lookup before backend inference and stores only
responses passing that policy. Exact hits return the saved response with a new
gateway request ID; the response body remains provider-originated. A hit does
not receive a remote-inference energy estimate, so its energy and carbon
headers/log values are null rather than billed as another provider generation.

Semantic caching is implemented but remains disabled by default with
`SEMANTIC_CACHE_ENABLED=false`. Enabling it requires an explicit
`SEMANTIC_CACHE_SIMILARITY_THRESHOLD`; no production threshold is selected
until Task 4 validates one. Embeddings use only the final user message's text.
This isolates semantic similarity to the latest user question while the
system prompt, every prior turn, final-message metadata, model, endpoint,
optimizer arm, authenticated tenant scope, and all other request parameters
must match exactly. Their SHA-256 tags are applied inside the RediSearch KNN
query, before nearest-neighbor selection, never as a Python post-filter.
Embedding timeouts default to 100 ms and are configurable with
`SEMANTIC_CACHE_EMBED_TIMEOUT_MS`; timeouts, embedding failures, and Redis
errors fail open to the backend and label the request `unavailable`. Embedding
uses a dedicated two-worker executor, separate from the gateway's
50-worker executor, with PyTorch intra-op and inter-op thread counts set to 2.
The index is created idempotently at startup, uses the loaded model's
dimension, and is versioned by the embedding model revision. Task 5 must
measure embedding latency at concurrency 10 and 50 and report p50/p95 for each;
the sequential Task 1 measurement is not a substitute for those concurrent
measurements. Compose exposes Redis only on the private service network
(`expose: 6379`, no host `ports` mapping), with `maxmemory=256mb` and
`maxmemory-policy=allkeys-lru`. Authentication is disabled by default for this
isolated development network; operators must configure `REDIS_PASSWORD`, add
the corresponding `--requirepass` to `REDIS_ARGS`, and set credentials in
`REDIS_URL` before using a shared or externally reachable Redis service.
