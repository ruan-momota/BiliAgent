# ---- BiliAgent Backend ----
FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (better cache)
COPY pyproject.toml uv.lock ./
COPY README.md ./

# Install dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy source code
COPY src/ src/

# Install the project itself
RUN uv sync --frozen --no-dev

# Expose port
EXPOSE 8000

# Run with uvicorn
CMD ["uv", "run", "uvicorn", "biliagent.main:app", "--host", "0.0.0.0", "--port", "8000"]
