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

No real-model path is configured in this environment: `.env.example` has no
provider key, the vLLM service is intentionally disabled, and this container
has no GPU for a local vLLM deployment. A hosted small-model check could be
run once a provider endpoint and credential are supplied; until then, real-model
validation is a documented Phase 3 blocker, analogous to GPU-backed NVML.

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

**Status: quota-blocked after partial real evidence.** The focused gateway test
uses a fake response only to verify plumbing: a response usage object of
`prompt_tokens=41` and `completion_tokens=7` produces those Greenway headers
and energy inputs. It is not Gemini evidence.

The planned live run was 41 Gemini requests: one end-to-end validation request,
then 20 counterbalanced rounds with one baseline and one optimized request per
round. It retains the 159-token sustainability-review prompt and six optimizer
rules. Google’s current rate-limits page (updated 2026-09-02) says active RPM,
input TPM, and RPD limits are project/model-specific and must be checked in AI
Studio before running; low concurrency or sequential requests avoid RPM bursts.
The pricing page lists Gemini 2.5 Flash input and output tokens as free. No paid
grounding tools are used.

Once a key is available, this section must be replaced with each request's
exact prompts, raw Gemini response, API-reported token counts, energy, carbon,
and gateway tax, plus the aggregate optimized-minus-baseline deltas and the
actual answer-length finding.

### Compliance gate and quota status (2026-09-03)

The authenticated AI Studio dashboard for project
`gen-lang-client-0592888602` reports Gemini 2.5 Flash limits of **5 RPM**,
**250K TPM**, and **20 RPD**. Its selected 28-day peak display currently reads
`8/5 RPM`, `2.09K/250K TPM`, and `21/20 RPD`, and the page says the rate limit
has been reached. The prior partial run also received HTTP 429. Therefore no
additional requests were sent after this dashboard check.

The required pre-run compliance sample (full baseline and optimized responses,
persona signal, and <=5-bullet check) could not be completed after the quota
was exhausted. Consequently there is no defensible full A/B result, aggregate
real energy/carbon delta, or gateway-tax estimate for the 256-token run. The
earlier successful samples remain valid real-model observations but all ended
with `finish_reason=length`; they show neither reliable instruction compliance
nor a valid output-saving mechanism.

The gateway was also hardened so an upstream 429 is returned rather than being
turned into a secondary 500. Resume only after the dashboard shows available
RPD and RPM headroom. Use the compliance gate first, then a small sequential
run sized below the displayed quota; do not use duplicate direct Gemini calls
for tax measurement unless the quota budget explicitly permits them.

### Quota-limited partial run (2026-09-03)

After the key was supplied, one baseline validation request returned HTTP 200
with raw Gemini text `## Quarterly Sustainability Review: Optimizing LLM Work`,
`prompt_tokens=155`, and `completion_tokens=10`. Its Greenway accounting was
`0.000102263335 kWh` and `0.038811930061 gCO2e`; the response used Gemini's
reported usage fields, not the tokenizer estimate.

A subsequent counterbalanced attempt completed 9 experimental requests (5
baseline, 4 optimized) before Gemini returned HTTP 429 on the next direct
comparison request. Representative raw responses were:

```text
baseline, round 1: "Okay, let's analyze the operational"; input=155 output=8
optimized, round 1: "Here's an analysis and practical recommendations"; input=176 output=8
baseline, round 3: "This sustainability review provides practical recommendations for your cloud-"; input=155 output=10
optimized, round 3: "Here are practical recommendations for your engineering and finance teams"; input=176 output=10
```

All successful samples ended with `finish_reason=length` under Gemini's
default output limit, so this partial run cannot establish usefulness or a
reliable output-length effect. The run also cannot provide a defensible gateway
tax aggregate after the 429. The benchmark harness now sets `max_tokens=256`
for the next run so completions can finish naturally. The 429 demonstrates that
the active project quota, rather than the documented free pricing, controls the
usable sample rate; check the AI Studio rate-limit page before resuming.

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
