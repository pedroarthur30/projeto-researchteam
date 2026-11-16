from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from projeto_researchteam.main import app, lifespan, pipeline_store
from tests.conftest import FakeResearchPipeline

VALID_HEADERS = {"X-API-Key": "test-secret-api-key-123"}


def test_health_check_publico(api_client: TestClient) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_dossie_rejeita_requisicao_sem_api_key(api_client: TestClient) -> None:
    response = api_client.post("/dossie", json={"empresa": "Nubank"})
    assert response.status_code == 401


def test_dossie_rejeita_api_key_invalida(api_client: TestClient) -> None:
    response = api_client.post(
        "/dossie",
        headers={"X-API-Key": "chave-incorreta"},
        json={"empresa": "Nubank"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Acesso não autorizado."


def test_dossie_valida_payload_empresa_em_branco(api_client: TestClient) -> None:
    response = api_client.post(
        "/dossie",
        headers=VALID_HEADERS,
        json={"empresa": "     "},
    )
    assert response.status_code == 422


def test_dossie_sucesso_e_propagacao_tenant(
    api_client: TestClient,
    fake_pipeline: FakeResearchPipeline,
) -> None:
    response = api_client.post(
        "/dossie",
        headers=VALID_HEADERS,
        json={"empresa": "  NexaTech Solutions  "},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["empresa"] == "NexaTech Solutions"
    assert "Dossiê NexaTech Solutions" in body["dossie_markdown"]
    assert fake_pipeline.last_config == {"configurable": {"tenant_id": "default_tenant"}}


def test_dossie_stream_sse_eventos(api_client: TestClient) -> None:
    with api_client.stream(
        "POST",
        "/dossie/stream",
        headers=VALID_HEADERS,
        json={"empresa": "Nubank"},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        conteudo = "".join(response.iter_text())

    assert "event: start" in conteudo
    assert "event: token" in conteudo
    assert "Nubank" in conteudo
    assert "event: complete" in conteudo


def test_dossie_retorna_503_quando_pipeline_indisponivel(api_client: TestClient) -> None:
    pipeline_store.clear()
    response = api_client.post(
        "/dossie",
        headers=VALID_HEADERS,
        json={"empresa": "Nubank"},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Serviço temporariamente indisponível."


def test_dossie_retorna_500_em_falha_interna_do_pipeline(api_client: TestClient) -> None:
    class FailingPipeline:
        async def ainvoke(
            self,
            input_data: dict[str, Any],
            config: dict[str, Any] | None = None,
        ) -> str:
            raise RuntimeError("Erro interno sensível")

    pipeline_store["research"] = FailingPipeline()  # type: ignore[assignment]
    response = api_client.post(
        "/dossie",
        headers=VALID_HEADERS,
        json={"empresa": "Nubank"},
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "Ocorreu um erro interno durante o processamento da pesquisa."


def test_dossie_stream_rejeita_sem_api_key_e_chave_invalida(api_client: TestClient) -> None:
    resp_sem_chave = api_client.post("/dossie/stream", json={"empresa": "Nubank"})
    assert resp_sem_chave.status_code == 401

    resp_invalida = api_client.post(
        "/dossie/stream",
        headers={"X-API-Key": "chave-errada"},
        json={"empresa": "Nubank"},
    )
    assert resp_invalida.status_code == 403


def test_dossie_stream_retorna_503_quando_pipeline_indisponivel(api_client: TestClient) -> None:
    pipeline_store.clear()
    response = api_client.post(
        "/dossie/stream",
        headers=VALID_HEADERS,
        json={"empresa": "Nubank"},
    )
    assert response.status_code == 503


def test_dossie_stream_emite_evento_error_em_falha_no_streaming(api_client: TestClient) -> None:
    class FailingStreamPipeline:
        async def astream(
            self,
            input_data: dict[str, Any],
            config: dict[str, Any] | None = None,
        ) -> AsyncGenerator[str, None]:
            yield "Primeiro token"
            raise RuntimeError("Falha no stream")

    pipeline_store["research"] = FailingStreamPipeline()  # type: ignore[assignment]
    with api_client.stream(
        "POST",
        "/dossie/stream",
        headers=VALID_HEADERS,
        json={"empresa": "Nubank"},
    ) as response:
        assert response.status_code == 200
        conteudo = "".join(response.iter_text())

    assert "event: start" in conteudo
    assert "event: error" in conteudo
    assert "Processamento interrompido." in conteudo
    assert "event: complete" not in conteudo


@pytest.mark.asyncio
async def test_lifespan_inicializa_e_limpa_pipeline_store() -> None:
    mock_vectorstore = MagicMock()
    mock_tavily = MagicMock()
    mock_pipeline = MagicMock()

    pipeline_store.clear()
    with (
        patch("projeto_researchteam.main.init_vectorstore", return_value=mock_vectorstore) as p_vs,
        patch("projeto_researchteam.main.TavilyClient", return_value=mock_tavily) as p_tav,
        patch("projeto_researchteam.main.build_research_pipeline", return_value=mock_pipeline) as p_pipe,
    ):
        async with lifespan(app):
            assert pipeline_store["research"] is mock_pipeline
            p_vs.assert_called_once()
            p_tav.assert_called_once()
            p_pipe.assert_called_once_with(
                vectorstore=mock_vectorstore,
                tavily_client=mock_tavily,
            )

    assert pipeline_store == {}