from fastapi.testclient import TestClient

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