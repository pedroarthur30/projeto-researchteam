import asyncio
import logging
import sys

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from tavily import TavilyClient

from projeto_researchteam.pipeline import build_research_pipeline
from projeto_researchteam.rag import init_vectorstore

logger = logging.getLogger("researchteam_cli")


async def _executar_cli(empresa: str) -> None:
    vectorstore = init_vectorstore()
    tavily_client = TavilyClient()
    pipeline = build_research_pipeline(
        vectorstore=vectorstore,
        tavily_client=tavily_client,
    )

    config: RunnableConfig = {"configurable": {"tenant_id": "default_tenant"}}
    
    async for chunk in pipeline.astream({"empresa": empresa}, config=config):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    sys.stdout.write("\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    load_dotenv()
    empresa = " ".join(sys.argv[1:]).strip()
    if not empresa:
        logger.error("Nenhuma empresa informada. Uso: uv run projeto-researchteam 'Nome'")
        return

    logger.info(f"Iniciando pesquisa competitiva sobre '{empresa}'...")
    asyncio.run(_executar_cli(empresa))