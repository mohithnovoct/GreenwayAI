# GreenwayAI

GreenwayAI is a carbon-aware LLM inference gateway. It sits between client
applications and model backends, measures the energy and carbon cost of every
request, and adds optimizations one layer at a time. Each layer is tested
against the measured baseline before it is trusted.

## Status

| Phase | Layer | State |
|---|---|---|
| 0-1 | Repo foundation; OpenAI-compatible ingress, canonicalization, content hashing | Done, tested |
| 2 | Telemetry and carbon engine (tiered energy and carbon sources, OpenTelemetry export) | Done, tested |
| 3 | Prompt optimization with A/B routing | Built; measured on a real model (Gemini 2.5 Flash): shorter outputs, slightly lower coverage. Energy savings not established |
| 4 | Semantic caching | In progress. Exact-match tier done and tested; semantic tier implemented but **disabled by default** until a similarity threshold is validated |
| 5-8 | Model routing, local vLLM serving, spatial routing, verification dashboard | Not started |

Full test suite: run `uv run pytest`. Evidence and caveats for every claim are
in [`docs/RESEARCH_CLAIMS.md`](docs/RESEARCH_CLAIMS.md) and
[`ingress/BENCHMARK.md`](ingress/BENCHMARK.md).

## What is measured and what is modeled

Read this before quoting any number from this repository.

- **Measured:** gateway overhead, provider-reported token counts, and output
  length. The Phase 3 result is a measured token reduction (about 78% and 82%
  on two prompt shapes, n=2 per arm, one model).
- **Modeled, not measured:** energy and carbon. Remote backends use a
  parametric model (`telemetry/parametric_coefficients.json`). The coefficients
  for Gemini are **placeholders**, not measurements. Energy and carbon figures
  derived from them are estimates only.
- **Not yet measured at all:** real GPU power draw. NVML sampling needs an
  NVIDIA GPU. Until one is used, the sampler runs in `mock_nvml` mode in
  development and CI. Every telemetry record carries `telemetry_source`,
  `energy_source_tier` and `carbon_source_tier`, so synthetic values are never
  mistaken for measured ones. Missing data is recorded as `null` and
  `unavailable`, never as a silent zero.
- **Modeled cache savings:** energy avoided by a cache hit is computed from the
  stored token counts. It is a model, not a measurement.

## Repository layout

- `ingress`: gateway, request normalization, benchmarks (`BENCHMARK.md`)
- `telemetry`: NVML sampling, parametric energy model, OpenTelemetry export
- `optimizers`: prompt optimization and the semantic/exact response cache
- `carbon`: carbon-intensity clients (static fallback, Electricity Maps, WattTime)
- `serving`: vLLM configuration (service not yet enabled)
- `verify`: experiments, preregistered rubrics, scoring scripts and raw results
- `shared`: common schemas and configuration
- `docs`: research claims tracker

## Run the stack

Copy `.env.example` to `.env`, then:

```bash
docker compose up
```

The gateway is at `http://localhost:8000` (API docs at `/docs`). A second
gateway for A/B runs is at `http://localhost:8002`; the two instances differ
only in `PROMPT_OPT_ENABLED` (`false` at 8000, `true` at 8002). Port 8001 is the
mock backend. Redis (Redis Stack, needed for vector search) is used by the
response cache and is not published to the host. The OpenTelemetry collector
receives gateway metrics. The vLLM service stays commented out until GPU-backed
serving is introduced.

Local development:

```bash
uv sync
uv run uvicorn ingress.app:app --reload
uv run pytest
```

### Backends

Set `GREENWAY_BACKEND_MODE` to `mock`, `vllm` or `gemini`. For Gemini, create a
key at [Google AI Studio](https://aistudio.google.com/apikey) and put it only
in the ignored `.env` file. Remote backends always use parametric energy
accounting. Full prompt logging is off by default (`GREENWAY_LOG_PROMPTS=false`).

### Response cache

Off by default (`CACHE_ENABLED=false`).

- **Exact tier:** key is a SHA-256 over the namespace and the canonical
  effective request (model, messages, all output-affecting parameters,
  backend endpoint, optimizer arm). Only non-streaming, `temperature=0`,
  tool-free requests are eligible, and only complete responses
  (`finish_reason=stop`, HTTP 200) are stored. Concurrent identical misses are
  coalesced into one backend call within a process. Redis failures fail open to
  the backend.
- **Semantic tier:** embeds only the final user turn with
  `all-MiniLM-L6-v2` and matches by vector similarity, but only among entries
  whose model, backend, optimizer arm, tenant scope, conversation context and
  parameters are identical. Disabled by default; no threshold is selected yet.
- Every request is tagged `cache_tier` in {`exact`, `semantic`, `miss`,
  `bypass`, `unavailable`}.
- **Limits:** the cache is shared by all callers of one gateway deployment and
  is not safe for multi-tenant use. Cached bodies are plaintext user content,
  kept up to `CACHE_TTL_SECONDS`. Redis has no password by default; it is only
  reachable on the Compose network.

## Reproducing the experiments

`verify/` holds the scripts, preregistered rubrics, raw responses, independent
scoring passes and adjudications behind the Phase 3 result. The commit history
shows each rubric and decision rule committed before the data it scores.
