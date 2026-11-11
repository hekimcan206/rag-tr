# syntax=docker/dockerfile:1.7
# One image for the API, the migrator and the seeder. EXTRAS=ml adds sentence-transformers + CPU PyTorch
# for real embedding models and rerankers; the default image stays small and needs no model download.
ARG PYTHON_IMAGE=python:3.13-slim-bookworm

FROM ghcr.io/astral-sh/uv:0.9.5 AS uv

FROM ${PYTHON_IMAGE} AS build
ARG EXTRAS=""
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
# Dependencies first, in their own layer, so source edits do not reinstall them.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project ${EXTRAS:+--extra $EXTRAS}
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable ${EXTRAS:+--extra $EXTRAS}

FROM ${PYTHON_IMAGE} AS runtime
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app
COPY --from=build /opt/venv /opt/venv
COPY --chown=10001:10001 data /app/data
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface
WORKDIR /app
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --start-period=20s --retries=20 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2)"]
CMD ["uvicorn", "--factory", "ragtr.api.app:main", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
