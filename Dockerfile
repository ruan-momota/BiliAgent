# ---- Builder: 装依赖、构建虚拟环境 ----
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/

RUN uv sync --frozen --no-dev

# ---- Runner: 只含运行期依赖 ----
FROM python:3.12-slim AS runner

# onnxruntime (chromadb) 和 torch (sentence-transformers) 需要 libgomp
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/README.md ./

# HF_HOME 指向挂载卷内目录，embedding 模型只在首次下载一次（~400MB）
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/data/hf_cache

EXPOSE 8000

CMD ["uvicorn", "biliagent.main:app", "--host", "0.0.0.0", "--port", "8000"]
