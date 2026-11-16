from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from projeto_researchteam.rag import (
    buscar_no_banco_vetorial,
    get_chroma_path,
    get_data_path,
    ingerir_documentos,
    init_vectorstore,
)


def test_get_paths_respeita_variaveis_de_ambiente(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("DATA_PATH", raising=False)
    monkeypatch.delenv("CHROMA_PATH", raising=False)
    assert get_data_path().name == "data"
    assert get_chroma_path().name == "chroma_db"

    custom_data = tmp_path / "custom_data"
    custom_chroma = tmp_path / "custom_chroma"
    monkeypatch.setenv("DATA_PATH", str(custom_data))
    monkeypatch.setenv("CHROMA_PATH", str(custom_chroma))
    assert get_data_path() == custom_data
    assert get_chroma_path() == custom_chroma


def test_init_vectorstore_loga_aviso_quando_diretorio_inexistente(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    missing_chroma = tmp_path / "nao_existe"
    monkeypatch.setenv("CHROMA_PATH", str(missing_chroma))

    mock_embeddings = MagicMock()
    mock_chroma = MagicMock()
    with (
        patch("projeto_researchteam.rag.OpenAIEmbeddings", return_value=mock_embeddings) as p_emb,
        patch("projeto_researchteam.rag.Chroma", return_value=mock_chroma) as p_chr,
    ):
        vs = init_vectorstore()

    assert vs is mock_chroma
    assert "Banco vetorial não encontrado" in caplog.text
    p_emb.assert_called_once_with(model="text-embedding-3-small")
    p_chr.assert_called_once_with(
        persist_directory=str(missing_chroma),
        embedding_function=mock_embeddings,
    )


def test_ingerir_documentos_retorna_cedo_se_pasta_vazia_ou_inexistente(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    empty_data = tmp_path / "vazia"
    empty_data.mkdir()
    monkeypatch.setenv("DATA_PATH", str(empty_data))

    with patch("projeto_researchteam.rag.PyPDFDirectoryLoader") as p_loader:
        ingerir_documentos()

    p_loader.assert_not_called()
    assert "Nenhum arquivo encontrado" in caplog.text


def test_ingerir_documentos_enriquece_metadados_e_persiste_chunks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "pdfs"
    data_dir.mkdir()
    (data_dir / "relatorio.pdf").write_bytes(b"%PDF-1.4 fake")
    chroma_dir = tmp_path / "chroma"

    monkeypatch.setenv("DATA_PATH", str(data_dir))
    monkeypatch.setenv("CHROMA_PATH", str(chroma_dir))

    docs = [Document(page_content="Receita anual de R$ 50M em 2025.", metadata={"source": "relatorio.pdf", "page": 1})]
    mock_loader = MagicMock()
    mock_loader.load.return_value = docs
    mock_embeddings = MagicMock()

    with (
        patch("projeto_researchteam.rag.PyPDFDirectoryLoader", return_value=mock_loader),
        patch("projeto_researchteam.rag.OpenAIEmbeddings", return_value=mock_embeddings),
        patch("projeto_researchteam.rag.Chroma.from_documents") as p_from_docs,
    ):
        ingerir_documentos(tenant_id="tenant_abc")

    p_from_docs.assert_called_once()
    call_kwargs = p_from_docs.call_args.kwargs
    chunks = call_kwargs["documents"]
    assert len(chunks) >= 1
    assert chunks[0].metadata["chunk_id"] == 0
    assert chunks[0].metadata["tenant_id"] == "tenant_abc"
    assert "data_ingestao" in chunks[0].metadata
    assert call_kwargs["persist_directory"] == str(chroma_dir)


@pytest.mark.asyncio
async def test_buscar_no_banco_vetorial_indisponivel() -> None:
    res = await buscar_no_banco_vetorial(query="receita", vectorstore=None)  # type: ignore[arg-type]
    assert "Conexão com o banco vetorial indisponível" in res


@pytest.mark.asyncio
async def test_buscar_no_banco_vetorial_sem_resultados_e_filtro_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_SCORE_THRESHOLD", "0.5")
    mock_retriever = MagicMock()
    mock_retriever.ainvoke = AsyncMock(return_value=[])
    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever

    res = await buscar_no_banco_vetorial(
        query="valuation",
        vectorstore=mock_vs,
        k=6,
        tenant_id="tenant_xyz",
    )

    assert "Nenhuma informação relevante encontrada" in res
    mock_vs.as_retriever.assert_called_once_with(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 6,
            "score_threshold": 0.5,
            "filter": {"tenant_id": "tenant_xyz"},
        },
    )


@pytest.mark.asyncio
async def test_buscar_no_banco_vetorial_formata_referencias() -> None:
    docs = [
        Document(
            page_content="Faturamento cresceu 40%.",
            metadata={
                "source": "/app/data/relatorio_nexatech.pdf",
                "page": 2,
                "chunk_id": 5,
                "data_ingestao": "2026-11-10 10:00:00",
            },
        ),
        Document(
            page_content="Sem metadados completos.",
            metadata={},
        ),
    ]
    mock_retriever = MagicMock()
    mock_retriever.ainvoke = AsyncMock(return_value=docs)
    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever

    res = await buscar_no_banco_vetorial(query="faturamento", vectorstore=mock_vs)

    assert "REFERÊNCIA | Arquivo: relatorio_nexatech.pdf | Pág: 2 | ID: 5 | Ingestão: 2026-11-10 10:00:00" in res
    assert "Faturamento cresceu 40%." in res
    assert "\n\n---\n\n" in res
    assert "Arquivo: Desconhecida | Pág: N/A | ID: N/A | Ingestão: N/A" in res
