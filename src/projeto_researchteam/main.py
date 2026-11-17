import json
import logging
import os
import secrets
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from langchain_core.runnables import Runnable, RunnableConfig
from tavily import TavilyClient

from projeto_researchteam.pipeline import build_research_pipeline
from projeto_researchteam.rag import init_vectorstore
from projeto_researchteam.schemas import DossieRequest, DossieResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("researchteam_api")

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
SECRET_API_KEY = os.getenv("APP_API_KEY")
if not SECRET_API_KEY:
    raise RuntimeError(
        "A variável de ambiente APP_API_KEY é obrigatória e não foi definida."
    )

async def validar_api_key(api_key: str = Security(api_key_header)) -> str:
    """Valida a chave da API com tempo constante para evitar Timing Attacks."""
    if not secrets.compare_digest(api_key, SECRET_API_KEY):
        logger.warning("Tentativa de acesso bloqueada: API Key inválida.")
        raise HTTPException(status_code=403, detail="Acesso não autorizado.")
    return api_key


pipeline_store: dict[str, Runnable[dict[str, str], str]] = {}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Inicializa clientes e injeta dependências no pipeline na subida da API."""
    vectorstore = init_vectorstore()
    tavily_client = TavilyClient()
    pipeline_store["research"] = build_research_pipeline(
        vectorstore=vectorstore,
        tavily_client=tavily_client,
    )
    logger.info("Pipeline, Tavily e Vectorstore inicializados com sucesso.")
    yield
    pipeline_store.clear()


app = FastAPI(
    title="Research Team AI API",
    description="Pipeline Corporativo de Inteligência Competitiva com suporte a Agentic RAG.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*", "X-API-Key"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/dossie", response_model=DossieResponse, dependencies=[Depends(validar_api_key)])
async def gerar_dossie(payload: DossieRequest) -> DossieResponse:
    """Executa o pipeline completo e retorna o dossiê. Requer autenticação."""
    pipeline = pipeline_store.get("research")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Serviço temporariamente indisponível.")

    try:
        config: RunnableConfig = {"configurable": {"tenant_id": "default_tenant"}}
        resultado = await pipeline.ainvoke({"empresa": payload.empresa}, config=config)
        return DossieResponse(empresa=payload.empresa, dossie_markdown=resultado)
    except Exception as exc:
        logger.error(f"Falha ao gerar dossiê para '{payload.empresa}': {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Ocorreu um erro interno durante o processamento da pesquisa.",
        )


@app.post("/dossie/stream", dependencies=[Depends(validar_api_key)])
async def gerar_dossie_stream(payload: DossieRequest) -> StreamingResponse:
    """Transmite a redação final em tempo real usando Server-Sent Events (SSE)."""
    pipeline = pipeline_store.get("research")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Serviço temporariamente indisponível.")

    async def sse_generator() -> AsyncGenerator[str]:
        try:
            yield f"event: start\ndata: {json.dumps({'status': 'Iniciando pesquisa agentica...'})}\n\n"

            config: RunnableConfig = {"configurable": {"tenant_id": "default_tenant"}}
            async for chunk in pipeline.astream({"empresa": payload.empresa}, config=config):
                dados = json.dumps({"token": chunk})
                yield f"event: token\ndata: {dados}\n\n"

            yield f"event: complete\ndata: {json.dumps({'status': 'Geração concluída'})}\n\n"

        except Exception as exc:
            logger.error(f"Erro no streaming para '{payload.empresa}': {exc}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'detail': 'Processamento interrompido.'})}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )