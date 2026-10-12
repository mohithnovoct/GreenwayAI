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

The gateway is available at `http://localhost:8000`, with API docs at
`http://localhost:8000/docs`. A second gateway instance for A/B runs is
available at `http://localhost:8002`; port 8001 is reserved for the mock
backend. Redis and the OpenTelemetry collector are included as integration
placeholders. The vLLM service remains commented out until GPU-backed serving
is introduced.
The two gateway instances share their application environment except for
`PROMPT_OPT_ENABLED` (`false` at port 8000, `true` at port 8002).

`uv run python verify/gemini_ab.py` sends both prompt arms through the gateway
instances on ports 8000 and 8002. Its `gateway_tax_ms` is an estimate based on
client-observed gateway latency minus the gateway-reported upstream round trip.

### Gemini backend

Create a free key at [Google AI Studio](https://aistudio.google.com/apikey),
put it only in the ignored `.env` file, and select the hosted backend:

```text
GREENWAY_BACKEND_MODE=gemini
GEMINI_API_KEY=your-key-here
GREENWAY_LOG_PROMPTS=false
```

The gateway sends OpenAI-compatible requests to Gemini 2.5 Flash. `mock` and
`vllm` remain available; switching modes requires only an environment change
and restart. Remote backends automatically use parametric energy accounting;
local NVML is reserved for local inference hardware. Full prompt logging is
disabled by default and should be enabled only for controlled debugging.

For local Python development, install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync
uv run uvicorn ingress.app:app --reload
```

## NVML and GPU requirements

Hardware-backed NVML validation requires a deployment target with a real NVIDIA GPU and the corresponding `libnvidia-ml.so.1` runtime available in the container or host environment. This dev/CI environment does not provide that, so the project intentionally runs the sampler in `mock_nvml` mode in local and CI execution. The sampler logs a `telemetry_source` value so synthetic power data is never mistaken for real GPU-measured telemetry downstream.

The benchmark and dashboard workflow should treat `mock_nvml` values as development-only placeholders until a real GPU target is available for hardware validation.