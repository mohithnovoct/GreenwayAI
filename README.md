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