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

**Status: real-model probe pending.** A focused gateway test uses a fake
response to verify that nonzero provider usage (`prompt_tokens=41`,
`completion_tokens=7`) is preferred over local token estimates. It is not
Gemini evidence.

#### Superseded September telemetry

The September Gemini figures `0.000102263335 kWh` and
`0.038811930061 gCO2e` are **superseded and must not be used**. They were
computed under the old NVML-tier accounting behavior and do not represent the
current Gemini LCA-tier parametric coefficient estimate.

#### Truncation hypothesis and probe plan

Earlier partial Gemini samples ended with `finish_reason=length`; their short
visible text alone does not establish why. The previous explanation that
Gemini's default output limit caused the short responses was unsupported:
thinking-token consumption, output limits, and other causes were not isolated.
The controlled probe below tests the theory. Usage arithmetic supports the
thinking-token explanation for the original 256-token response, but the raw
usage object does not expose a separate thought-token detail. Raising the
limit to 512 with `reasoning_effort="none"` still ends at the 512-token output
ceiling, so those longer responses are directly explained by that configured
limit.

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
completion limit. It did not express an expert persona and contained 14
Markdown bullet lines (including nested bullets); the baseline 512-token
response contained 9 bullet lines and no explicit expert-persona signal.
Therefore the compliance gate **failed**. Do not proceed to the sequential
A/B; these samples cannot support a claim about comparative energy, carbon, or
answer quality.

All three gateway records reported `token_count_source=provider`,
`telemetry_source=parametric_estimated`, and `energy_source_tier=lca` (not
`nvml` or `mock_nvml`). The observed energy and carbon values are proxy
coefficient estimates, not Google-published figures. No A/B aggregates or
statistical claims are reported.

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
