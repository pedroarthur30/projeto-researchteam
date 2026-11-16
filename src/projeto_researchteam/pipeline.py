import os
from operator import itemgetter
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from projeto_researchteam.agent import build_pesquisador_chain
from projeto_researchteam.schemas import AnaliseFinanceira, AnaliseReputacao


def _build_writer_llm_with_fallback() -> Runnable[Any, Any]:
    """Configura o modelo redator com retentativas nativas do SDK e fallback entre provedores."""
    primary_llm: BaseChatModel = ChatOpenAI(
        model=os.getenv("OPENAI_WRITER_MODEL", "gpt-4o"),
        temperature=0.2,
        timeout=45,
        max_retries=2,
    )

    if os.getenv("ANTHROPIC_API_KEY"):
        backup_llm: BaseChatModel = ChatAnthropic(
            model=os.getenv("ANTHROPIC_BACKUP_MODEL", "claude-3-5-haiku-latest"),
            temperature=0.2,
            timeout=45,
            max_retries=2,
        )
    else:
        backup_llm = ChatOpenAI(
            model=os.getenv("OPENAI_AGENT_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            timeout=45,
            max_retries=2,
        )

    return primary_llm.with_fallbacks([backup_llm])


def _format_for_writer(data: dict[str, Any]) -> dict[str, str]:
    """Serializa os objetos Pydantic em JSON legível para o prompt do redator."""
    financeiro: AnaliseFinanceira = data["financeiro"]
    reputacao: AnaliseReputacao = data["reputacao"]

    return {
        "empresa": data["empresa"],
        "dados_financeiros": financeiro.model_dump_json(indent=2),
        "dados_reputacao": reputacao.model_dump_json(indent=2),
    }


def build_research_pipeline(
    vectorstore: Chroma,
    tavily_client: TavilyClient | None = None,
) -> Runnable[dict[str, str], str]:
    """Constrói e retorna o pipeline completo de Inteligência Competitiva em LCEL."""
    client = tavily_client or TavilyClient()
    pesquisador_chain = build_pesquisador_chain(
        vectorstore=vectorstore,
        tavily_client=client,
    )

    extractor_llm = ChatOpenAI(
        model=os.getenv("OPENAI_EXTRACTOR_MODEL", "gpt-4o-mini"),
        temperature=0,
    )

    financial_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extraia exclusivamente os dados financeiros, de monetização e concorrência do texto abaixo.",
            ),
            ("human", "Empresa: {empresa}\n\nPesquisa Bruta:\n{pesquisa_bruta}"),
        ]
    )
    financial_chain = (
        financial_prompt | extractor_llm.with_structured_output(AnaliseFinanceira)
    ).with_retry(stop_after_attempt=2)

    reputation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extraia o sentimento de mercado, notícias recentes, parcerias e riscos do texto abaixo.",
            ),
            ("human", "Empresa: {empresa}\n\nPesquisa Bruta:\n{pesquisa_bruta}"),
        ]
    )
    reputation_chain = (
        reputation_prompt | extractor_llm.with_structured_output(AnaliseReputacao)
    ).with_retry(stop_after_attempt=2)

    writer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você é um Diretor de Estratégia redigindo um Dossiê Executivo de Inteligência Competitiva.\n"
                "Com base nos dados estruturados fornecidos, gere um relatório bem formatado em Markdown com as seções:\n"
                "1. Visão Geral & Modelo de Negócio\n"
                "2. Cenário Financeiro & Concorrentes\n"
                "3. Reputação, Movimentos Recentes & Riscos\n"
                "4. Fontes Consultadas\n\n"
                "Seja objetivo, analítico e jamais invente informações fora dos JSONs recebidos.",
            ),
            (
                "human",
                "Empresa: {empresa}\n\n"
                "### Análise Financeira Estruturada:\n{dados_financeiros}\n\n"
                "### Análise de Reputação Estruturada:\n{dados_reputacao}",
            ),
        ]
    )

    writer_llm = _build_writer_llm_with_fallback()

    pipeline: Runnable[dict[str, str], str] = (
        pesquisador_chain
        | RunnableParallel(
            empresa=itemgetter("empresa"),
            financeiro=financial_chain,
            reputacao=reputation_chain,
        )
        | RunnableLambda(_format_for_writer)
        | writer_prompt
        | writer_llm
        | StrOutputParser()
    )

    return pipeline