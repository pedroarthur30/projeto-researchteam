FROM python:3.14-slim

# Copia o gerenciador uv oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Instala primeiro as dependências para aproveitar o cache de camadas do Docker
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Copia o código-fonte e instala o pacote do projeto
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Cria usuário não-root e diretórios graváveis para o ChromaDB e PDFs
RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/data /app/chroma_db && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uv", "run", "--no-dev", "uvicorn", "projeto_researchteam.main:app", "--host", "0.0.0.0", "--port", "8000"]