# GreenwayAI

GreenwayAI is a carbon-aware LLM inference gateway. It measures request energy
and carbon cost before progressively adding optimization, routing, and serving
efficiency improvements.

## Repository layout

- `ingress`: L7 gateway and request normalization
- `telemetry`: NVML telemetry and carbon accounting
- `optimizers`: prompt optimization, semantic caching, and routing
- `serving`: vLLM configuration and wrappers
- `carbon`: carbon-intensity API clients
- `verify`: benchmarks and dashboards
- `shared`: common schemas and configuration loading

## Phase foundation

This repository currently contains the service skeleton for the foundation
phase. Later phases should be measured against a real telemetry baseline before
their optimizations are enabled.

## Run the stack

Copy `.env.example` to `.env`, then start the foundation services:

```bash
docker compose up
```

The gateway stub is available at `http://localhost:8000`, with API docs at
`http://localhost:8000/docs`. Redis and the OpenTelemetry collector are
included as integration placeholders. The vLLM service remains commented out
until GPU-backed serving is introduced.

For local Python development, install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync
uv run uvicorn ingress.app:app --reload
```

## NVML and GPU requirements

Hardware-backed NVML validation requires a deployment target with a real NVIDIA GPU and the corresponding `libnvidia-ml.so.1` runtime available in the container or host environment. This dev/CI environment does not provide that, so the project intentionally runs the sampler in `mock_nvml` mode in local and CI execution. The sampler logs a `telemetry_source` value so synthetic power data is never mistaken for real GPU-measured telemetry downstream.

The benchmark and dashboard workflow should treat `mock_nvml` values as development-only placeholders until a real GPU target is available for hardware validation.