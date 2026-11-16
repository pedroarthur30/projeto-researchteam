import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from projeto_researchteam.agent import MAX_RESEARCH_TIME_SECONDS, build_pesquisador_chain
from projeto_researchteam.pipeline import (
    _build_writer_llm_with_fallback,
    _format_for_writer,
    build_research_pipeline,
)
from projeto_researchteam.schemas import AnaliseFinanceira, AnaliseReputacao


@pytest.mark.asyncio
async def test_pesquisador_chain_propaga_tenant_config_e_retorna_pesquisa() -> None:
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content="Relatório bruto da NexaTech.")]}
    )

    with (
        patch("projeto_researchteam.agent.ChatOpenAI"),
        patch("projeto_researchteam.agent.create_agent", return_value=mock_agent),
        patch("projeto_researchteam.agent.build_tools", return_value=[]) as p_tools,
    ):
        chain = build_pesquisador_chain(
            vectorstore=MagicMock(),
            tavily_client=MagicMock(),
        )
        resultado = await chain.ainvoke(
            {"empresa": "NexaTech"},
            config={
                "configurable": {"tenant_id": "tenant_42"},
                "metadata": {"origem": "teste"},
                "tags": ["unit"],
            },
        )

    assert resultado == {
        "empresa": "NexaTech",
        "pesquisa_bruta": "Relatório bruto da NexaTech.",
    }
    assert p_tools.call_args.kwargs["tenant_id"] == "tenant_42"
    agent_cfg = mock_agent.ainvoke.call_args.kwargs["config"]
    assert agent_cfg["configurable"] == {"tenant_id": "tenant_42"}
    assert agent_cfg["recursion_limit"] == 8
    assert agent_cfg["metadata"] == {"origem": "teste"}
    assert agent_cfg["tags"] == ["unit"]


@pytest.mark.asyncio
async def test_pesquisador_chain_converte_conteudo_nao_string() -> None:
    mock_msg = MagicMock()
    mock_msg.content = [{"type": "text", "text": "bloco"}]
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(return_value={"messages": [mock_msg]})

    with (
        patch("projeto_researchteam.agent.ChatOpenAI"),
        patch("projeto_researchteam.agent.create_agent", return_value=mock_agent),
        patch("projeto_researchteam.agent.build_tools", return_value=[]),
    ):
        chain = build_pesquisador_chain(MagicMock(), MagicMock())
        resultado = await chain.ainvoke({"empresa": "Nubank"})

    assert "bloco" in resultado["pesquisa_bruta"]


@pytest.mark.asyncio
async def test_pesquisador_chain_trata_timeout() -> None:
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError)

    with (
        patch("projeto_researchteam.agent.ChatOpenAI"),
        patch("projeto_researchteam.agent.create_agent", return_value=mock_agent),
        patch("projeto_researchteam.agent.build_tools", return_value=[]),
    ):
        chain = build_pesquisador_chain(MagicMock(), MagicMock())
        resultado = await chain.ainvoke({"empresa": "Nubank"})

    assert resultado["empresa"] == "Nubank"
    assert f"{MAX_RESEARCH_TIME_SECONDS}s" in resultado["pesquisa_bruta"]
    assert "dados podem estar incompletos" in resultado["pesquisa_bruta"].lower()


def test_format_for_writer_serializa_modelos_pydantic() -> None:
    fin = AnaliseFinanceira(
        empresa="NexaTech",
        modelo_monetizacao="B2B SaaS",
        receita_ou_valuation_estimado="R$ 20M",
        principais_concorrentes=["Concorrente A"],
        fontes=["https://exemplo.com"],
    )
    rep = AnaliseReputacao(
        sentimento_geral="Misto",
        resumo_noticias_recentes=["Expansão regional"],
    )

    formatted = _format_for_writer(
        {"empresa": "NexaTech", "financeiro": fin, "reputacao": rep}
    )

    assert formatted["empresa"] == "NexaTech"
    assert '"modelo_monetizacao": "B2B SaaS"' in formatted["dados_financeiros"]
    assert '"sentimento_geral": "Misto"' in formatted["dados_reputacao"]


def test_build_writer_llm_with_fallback_usa_openai_ou_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = MagicMock()
    primary.with_fallbacks.return_value = "fallback_chain"
    backup_openai = MagicMock()
    backup_anthropic = MagicMock()

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with (
        patch("projeto_researchteam.pipeline.ChatOpenAI", side_effect=[primary, backup_openai]) as p_oai,
        patch("projeto_researchteam.pipeline.ChatAnthropic") as p_ant,
    ):
        res = _build_writer_llm_with_fallback()
    assert res == "fallback_chain"
    assert p_oai.call_count == 2
    p_ant.assert_not_called()
    primary.with_fallbacks.assert_called_once_with([backup_openai])

    primary.reset_mock()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    with (
        patch("projeto_researchteam.pipeline.ChatOpenAI", return_value=primary),
        patch("projeto_researchteam.pipeline.ChatAnthropic", return_value=backup_anthropic) as p_ant,
    ):
        res_ant = _build_writer_llm_with_fallback()
    assert res_ant == "fallback_chain"
    p_ant.assert_called_once()
    primary.with_fallbacks.assert_called_once_with([backup_anthropic])


@pytest.mark.asyncio
async def test_build_research_pipeline_orquestra_etapas_lcel() -> None:
    fake_pesquisador = RunnableLambda(
        lambda x: {"empresa": x["empresa"], "pesquisa_bruta": "Dados brutos coletados."}
    )
    fake_fin = AnaliseFinanceira(
        empresa="Nubank",
        modelo_monetizacao="Serviços financeiros digitais",
        receita_ou_valuation_estimado="US$ 10B",
        principais_concorrentes=["Inter"],
        fontes=["https://nubank.com.br"],
    )
    fake_rep = AnaliseReputacao(
        sentimento_geral="Positivo",
        resumo_noticias_recentes=["Crescimento de base de clientes"],
    )

    mock_extractor = MagicMock()
    mock_extractor.with_structured_output.side_effect = [
        RunnableLambda(lambda _x: fake_fin),
        RunnableLambda(lambda _x: fake_rep),
    ]
    fake_writer = RunnableLambda(lambda _x: AIMessage(content="# Dossiê Nubank\nRelatório final."))

    with (
        patch("projeto_researchteam.pipeline.build_pesquisador_chain", return_value=fake_pesquisador),
        patch("projeto_researchteam.pipeline.ChatOpenAI", return_value=mock_extractor),
        patch("projeto_researchteam.pipeline._build_writer_llm_with_fallback", return_value=fake_writer),
    ):
        pipeline = build_research_pipeline(vectorstore=MagicMock(), tavily_client=MagicMock())
        saida = await pipeline.ainvoke({"empresa": "Nubank"})

    assert saida == "# Dossiê Nubank\nRelatório final."
