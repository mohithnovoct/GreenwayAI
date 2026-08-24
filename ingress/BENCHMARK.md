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

The default Compose stack provides an independent mock backend on port 8001,
so this measures a real direct HTTP path rather than comparing two calls to the
gateway's internal fallback.

## Baseline

Measured 2026-08-24 in the dev container with the mock backend after warm-up.

200 requests, concurrency 20:

```text
path       p50_ms   p95_ms   p99_ms
gateway     125.79   281.26   363.61
backend      59.97   171.38   289.57
gateway tax p50/p95/p99: 65.81/109.88/74.04 ms
```

1000 requests, concurrency 50, using the exact command above:

```text
path       p50_ms   p95_ms   p99_ms
gateway     316.36   697.15   916.79
backend     146.97   674.23  1125.85
gateway tax p50/p95/p99: 169.39/22.92/-209.06 ms
```

At both requested loads, gateway latency is higher than direct backend latency
at p50 and p95. The p99 paired tax is negative in the 1000/50 sample because
the independently timed mock backend has a noisier tail; p99 is not used as the
gateway-tax acceptance criterion.

The original multi-second tail was caused by per-request `AsyncClient`
construction and synchronous normalization/tokenizer work on the event loop.
The gateway now owns one pooled client for its lifespan, offloads request
normalization/tokenization to a worker thread, and eagerly initializes the
fallback tokenizer. Current gateway logs for the 1000/50 run show
normalization p50/p95 of 13.315/29.75 ms and proxy p50/p95 of 133.347/523.207
ms. The remaining latency is predominantly the mock backend/proxy scheduling
path, not tokenizer cold-start blocking; a real vLLM backend should be used for
the serving baseline.