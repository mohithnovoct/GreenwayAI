FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY carbon ./carbon
COPY ingress ./ingress
COPY optimizers ./optimizers
COPY serving ./serving
COPY shared ./shared
COPY telemetry ./telemetry
COPY verify ./verify

RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "ingress.app:app", "--host", "0.0.0.0", "--port", "8000"]