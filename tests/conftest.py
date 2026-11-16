import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

# Define a chave antes de importar main.py para satisfazer a validação fail-fast
os.environ["APP_API_KEY"] = "test-secret-api-key-123"

import pytest
from fastapi.testclient import TestClient

from projeto_researchteam.main import app, pipeline_store


class FakeResearchPipeline:
    """Simula o Runnable LCEL para testar rotas HTTP e SSE offline."""

    def __init__(self) -> None:
        self.last_input: dict[str, Any] | None = None
        self.last_config: dict[str, Any] | None = None

    async def ainvoke(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> str:
        self.last_input = input_data
        self.last_config = config
        return f"# Dossiê {input_data['empresa']}\nConteúdo gerado em teste."

    async def astream(
        self,
        input_data: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        self.last_input = input_data
        self.last_config = config
        for token in ["# Dossiê ", input_data["empresa"], "\nConcluído."]:
            yield token


@pytest.fixture
def fake_pipeline() -> FakeResearchPipeline:
    return FakeResearchPipeline()


@pytest.fixture
def api_client(fake_pipeline: FakeResearchPipeline) -> Generator[TestClient, None, None]:
    """Injeta o pipeline simulado na store da aplicação sem acionar APIs externas."""
    pipeline_store["research"] = fake_pipeline 
    yield TestClient(app)
    pipeline_store.clear()