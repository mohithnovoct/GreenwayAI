FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY carbon ./carbon
COPY ingress ./ingress
COPY shared ./shared
COPY telemetry ./telemetry

RUN pip install --no-cache-dir \
    "fastapi>=0.115,<1.0" \
    "uvicorn[standard]>=0.30,<1.0" \
    "pydantic>=2.9,<3.0" \
    "python-dotenv>=1.0,<2.0" \
    "httpx>=0.27,<1.0" \
    "tiktoken>=0.8,<1.0"

EXPOSE 8000
CMD ["uvicorn", "ingress.app:app", "--host", "0.0.0.0", "--port", "8000"]