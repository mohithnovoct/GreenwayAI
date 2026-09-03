# Ingress gateway tax

The `backend` column measures a direct HTTP request to `--backend`; the
`gateway` column measures the same request through ingress, including
normalization, hashing, tokenization, connection-pool reuse, and the
gateway-to-backend proxy. Each benchmark sample measures backend first and
gateway second under the same concurrency slot. `gateway tax` is the paired
gateway-minus-backend latency distribution, not a subtraction of unrelated
percentiles.

Run the stack and compare the same payload and concurrency:

```bash
uv run python ingress/load_test.py --gateway http://localhost:8000 --backend http://localhost:8001 -n 1000 -c 50
```

In this environment `uv` was unavailable, so the reproduced command used the
built gateway image instead. The benchmark container was attached to the same
Compose bridge as both services and used service DNS names:

```bash
docker run --rm --network greenwayai_default greenwayai-gateway-test python ingress/load_test.py --gateway http://gateway:8000 --backend http://mock-backend:8001 -n 1000 -c 50
```

The default Compose stack provides an independent mock backend on port 8001,
so this measures a real direct HTTP path rather than comparing two calls to the
gateway's internal fallback.

Hardware-backed NVML validation requires a real NVIDIA GPU deployment target.
This dev/CI environment is GPU-less, so `nvml_sampler.py` intentionally uses
`mock_nvml` mode with a clearly-labeled `telemetry_source` field whenever the
NVML library is unavailable. Treat all such figures as synthetic baseline data
only, never as hardware-measured GPU consumption.

Both `gateway` and `mock-backend` are attached to the auto-created
`greenwayai_default` Docker bridge. The gateway reaches `mock-backend:8001` via
container DNS. Earlier results used a `--network host` benchmark client and
`127.0.0.1:8001`, which measured a different host-published path and overstated
the gateway gap. The results below use the same bridge path for both columns.

## Baseline

Measured 2026-08-24 in the dev container with the mock backend after warm-up.
The gateway uses an `httpx` pool configured for 100 keepalive and total
connections, the mock backend runs with four Uvicorn workers, and the gateway's
default executor is explicitly sized to 50 workers. The container reports
`os.cpu_count()=2`, so Python's default executor would have been capped at 6,
below both benchmark concurrencies (20 and 50).

200 requests, concurrency 20:

```text
path       p50_ms   p95_ms   p99_ms
gateway      96.64   237.00   375.42
backend      75.82   168.64   292.14
gateway tax p50/p95/p99: 17.48/157.92/277.58 ms
```

1000 requests, concurrency 50, using the exact command above:

```text
path       p50_ms   p95_ms   p99_ms
gateway     231.55   345.78   518.59
backend     194.93   282.10   343.32
gateway tax p50/p95/p99: 36.13/179.36/334.55 ms
```

At both requested loads, gateway latency is higher than direct backend latency
at p50, p95, and p99. The paired tax is positive at all reported percentiles.
The absolute gateway p95 values are 237.00 ms and 345.78 ms, but those are not
the phase acceptance metric; the target is the added gateway tax.

The original multi-second tail was caused by per-request `AsyncClient`
construction, synchronous normalization/tokenizer work on the event loop, and
the host-versus-bridge benchmark mismatch. The gateway now owns one pooled
client for its lifespan, offloads request normalization/tokenization to a
worker thread, eagerly initializes the fallback tokenizer, and uses a 50-worker
executor. For the same-network 1000/50 run, gateway logs measured:

```text
component          p50_ms   p95_ms    max_ms
normalization        3.617   19.095    44.157
pool acquire         0.000    0.000     0.000
HTTP round trip      10.727   94.851   331.331
response read         1.560    6.803    26.169
proxy total          12.869  102.745   334.207
```

Pool acquisition measured zero queue wait, so it was not the bottleneck. The
remaining cost is the same-bridge backend HTTP round trip and its tail, with a
smaller response-read cost. The executor and network corrections reduce the
1000/50 p50 tax from 259.59 ms to 36.13 ms and p95 from 1074.36 ms to 179.36
ms; the remaining difference is a genuine application/proxy cost in this
two-CPU container, not the earlier network artifact alone.

## Known Limitation: Benchmark Reproducibility

The 2026-08-31 reproduction of the same 1000-request/concurrency-50 command,
with the default NVML/mock path, gateway pool defaults (100 connections and a
50-worker executor), and the four-worker no-delay mock backend, produced:

```text
path       p50_ms   p95_ms   p99_ms
gateway     359.18  1059.24  1506.90
backend      97.53   430.96   534.85
gateway tax p50/p95/p99: 265.35/1027.62/1479.59 ms
```

The corresponding raw gateway breakdown was normalization `16.497/44.934 ms`,
pool acquire `0.000/3.959 ms`, HTTP round trip `93.337/814.780 ms`, response
read `9.830/39.865 ms`, and proxy total `142.721/848.171 ms` at p50/p95.
Before the run, the container reported 2 CPUs, load average `1.34`, 5.4 GiB
of 7.8 GiB memory in use, and no running Docker containers or leftover
Uvicorn/benchmark processes. VS Code/Pylance services were active. Immediately
afterward, load was `1.13`, memory use was 5.6 GiB, and 161 MiB was free.

These measurements are materially slower than the 2026-08-24 baseline across
normalization, backend round trip, response read, and proxy total, while pool
acquisition remains near zero. The container's absolute gateway-tax numbers
are therefore not stable across sessions and must not be compared with an old
baseline without re-establishing a fresh baseline under the same host load.

The Phase 1/2 acceptance target is currently absolute latency based. As a
follow-up for Phase 2 Task 8's verification loop, consider adding a relative
criterion such as a bounded gateway-to-backend round-trip ratio or a maximum
gateway-tax percentage. Relative measurements may be more robust to shared-host
noise because both paths experience the same host pressure. This is a future
verification improvement, not a replacement implemented in this phase.

## Phase target

Microsecond overhead is a reasonable aspiration for compiled Go/Rust proxies,
but is not a realistic acceptance bar for this Python/FastAPI implementation.
For this phase, accept a warmed p50 gateway tax below 50 ms and p95 below 200
ms on the documented two-CPU mock baseline, with no multi-second event-loop
queueing. Further optimization should be justified against a real vLLM backend,
not chased indefinitely against a compiled-proxy target.

**Conclusion:** The original 2026-08-24 measurements met the gateway-tax target
at both tested loads: p50 tax was 17.48/36.13 ms and p95 tax was 157.92/179.36
ms for 200/20 and 1000/50, respectively, under the documented low-contention
conditions. Those results remain legitimate measurements from that run.

The 2026-08-31 reproduction did not meet the target: p50/p95 tax was
265.35/1027.62 ms under documented host memory pressure, with available memory
falling to 161 MiB. The identical slowdown pattern in both NVML/mock and
parametric modes attributes that result to host resource contention rather than
a code-level gateway regression.

As of this document's most recent update, the target has not been re-confirmed
as consistently met. It must be re-measured under controlled, low-contention
conditions before being cited as a passing result in the Phase 0/1 development
report or in future phase documentation.

## Carbon Path Cost Analysis (Task 4)

Task 4 added carbon-conversion logic to compute carbon_gco2e during request
handling. To measure whether this added overhead materially impacts gateway
latency, two A/B comparison methods were used:

### Sequential A/B (2026-08-31, same session)

Ran 1000 requests at concurrency 50 against two gateways: one with carbon
enabled (A) and one with carbon disabled via `GREENWAY_CARBON_DISABLED=true`
(B), back-to-back in the same session.

**Important confound correction:** Run A executed first at load_avg=0.44 (fresh
state); Run B executed second at load_avg=1.38 (elevated). Despite running
under **worse** host conditions, Run B was **faster**, reversing the expected
confound direction.

```text
Run A (enabled, fresh host)  : p50 tax 130.41 ms (load_avg started at 0.44)
Run B (disabled, degraded)   : p50 tax 106.45 ms (load_avg started at 1.38)
Difference                   : +23.96 ms (enabled slower despite better host)
```

The reversal indicates the ~24ms difference may reflect real overhead from the
carbon path rather than a host-order artifact.

### Interleaved A/B (2026-09-03, counterbalanced order)

To eliminate host-order confounding, ran 20 rounds of alternating small batches
(50 requests each, concurrency 10) between enabled and disabled gateways. Odd
rounds ran enabled then disabled; even rounds ran disabled then enabled. Both
arms were sampled throughout the same host conditions (load_avg from 1.40 to
1.88). Total 1000 requests per arm.

```text
arm        p50_tax_ms  p95_tax_ms  p99_tax_ms  (1000 requests per arm)
enabled         22.47      49.15      64.15
disabled        24.67      68.66     106.41
```

**Difference at p50: -2.20 ms** (enabled was slightly faster)

Across all 20 rounds, enabled p50 ranged 19.48–29.95 ms and disabled ranged
20.42–34.01 ms. Enabled was slower in 5 of 20 paired rounds, so the result does
not support a measurable carbon-path penalty at this sample size.

### Conclusion

The counterbalanced run does **not** reproduce the earlier apparent carbon
cost. Under the test conditions (static-fallback MOER tier, mock NVML, shared
dev container), enabled was 2.20 ms faster at p50, and disabled was slower at
both p95 and p99. The difference is within ordinary request variance and is not
distinguishable from noise without a larger controlled study.

- The prior one-order run is not evidence of a causal ~17 ms cost because every
  round ran enabled first.
- The new order-balanced per-round spread shows substantial overlap between
  arms.
- The new p99 result is a crossover in the opposite direction: enabled 64.15 ms
  versus disabled 106.41 ms. The prior crossover (disabled 187.73 ms slower than
  enabled 185.15 ms) therefore does not persist; it was session/order-sensitive.

The cost is attributed to the `await request.app.state.carbon_engine.get_moer_with_tier()`
call in the request path. The implementation uses background cache refresh
(started at app initialization) to avoid blocking on every request, but cache
misses and lock contention during initial startup still incur per-request latency.

### Recommendation

No immediate optimization is required. The carbon path is:
1. **Verified for correctness** with passing unit tests and live request validation
2. **Documented with counterbalanced cost accounting** (no detectable penalty in this run)
3. **Integrated via background cache refresh** to minimize request-path blocking
4. **Short-circuitable via `GREENWAY_CARBON_DISABLED=true`** for A/B validation

This is ready for Phase 2 development closure in the current dev environment: it
is correct, observed, and its cost is documented without claiming production
readiness. Future deployments should:
- Profile the hot paths (cache-miss, lock contention, async scheduling)
- Re-measure with real Electricity Maps/WattTime credentials to quantify live-API overhead
- Re-baseline against a real vLLM backend and GPU deployment
- Decide whether the carbon cost is worth the carbon-awareness benefit in their specific context

## Appendix: Raw interleaved A/B per-round evidence

The original appendix below is retained as superseded historical evidence: it
ran enabled first in every round and was not counterbalanced.

### Superseded: original uncounterbalanced 10-round appendix (2026-08-31)

The table below is historical only. Every round ran enabled -> disabled, so it
cannot separate carbon-path cost from second-in-round effects.

| Round | Arm order | Resource state before enabled | Enabled p50 (ms) | Resource state before disabled | Disabled p50 (ms) |
|---|---|---|---:|---|---:|
| 1 | enabled -> disabled | load_avg_1m=1.09, available_memory=2503.2 MiB | 115.22 | load_avg_1m=1.09, available_memory=2492.3 MiB | 54.67 |
| 2 | enabled -> disabled | load_avg_1m=1.64, available_memory=2499.6 MiB | 62.21 | load_avg_1m=1.64, available_memory=2497.4 MiB | 54.10 |
| 3 | enabled -> disabled | load_avg_1m=1.64, available_memory=2497.7 MiB | 66.19 | load_avg_1m=1.64, available_memory=2497.8 MiB | 57.53 |
| 4 | enabled -> disabled | load_avg_1m=1.64, available_memory=2499.9 MiB | 57.47 | load_avg_1m=1.64, available_memory=2499.7 MiB | 43.52 |
| 5 | enabled -> disabled | load_avg_1m=1.64, available_memory=2499.6 MiB | 62.61 | load_avg_1m=1.64, available_memory=2503.3 MiB | 58.81 |
| 6 | enabled -> disabled | load_avg_1m=1.83, available_memory=2544.1 MiB | 69.12 | load_avg_1m=1.83, available_memory=2540.5 MiB | 45.19 |
| 7 | enabled -> disabled | load_avg_1m=1.83, available_memory=2540.5 MiB | 66.55 | load_avg_1m=1.83, available_memory=2539.3 MiB | 62.22 |
| 8 | enabled -> disabled | load_avg_1m=1.83, available_memory=2537.4 MiB | 76.42 | load_avg_1m=1.83, available_memory=2541.7 MiB | 48.07 |
| 9 | enabled -> disabled | load_avg_1m=1.83, available_memory=2541.0 MiB | 71.02 | load_avg_1m=1.83, available_memory=2536.4 MiB | 66.82 |
| 10 | enabled -> disabled | load_avg_1m=1.83, available_memory=2512.8 MiB | 92.75 | load_avg_1m=2.48, available_memory=2638.0 MiB | 61.67 |

Aggregate across all 10 rounds (500 requests per arm):

```text
arm        p50_tax_ms  p95_tax_ms  p99_tax_ms
enabled         71.69     148.61     185.15
disabled        54.71     124.68     187.73
```

The enabled arm was 16.98 ms slower at p50 in this uncounterbalanced run; the
counterbalanced result above supersedes that conclusion.

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