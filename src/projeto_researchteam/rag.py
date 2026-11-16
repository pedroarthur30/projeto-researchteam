import datetime
import logging
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def get_data_path() -> Path:
    base = os.getenv("DATA_PATH")
    return Path(base) if base else Path(__file__).parent.parent.parent / "data"


def get_chroma_path() -> Path:
    base = os.getenv("CHROMA_PATH")
    return Path(base) if base else Path(__file__).parent.parent.parent / "chroma_db"


def init_vectorstore() -> Chroma:
    """Inicializa a conexão com o banco vetorial para injeção de dependência."""
    chroma_path = get_chroma_path()
    if not chroma_path.exists():
        logger.warning(
            f"Banco vetorial não encontrado em {chroma_path}. Execute a ingestão antes de consultar."
        )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=str(chroma_path),
        embedding_function=embeddings,
    )


def ingerir_documentos(tenant_id: str = "default_tenant") -> None:
    """Lê PDFs, enriquece metadados com tenant_id e salva no banco vetorial."""
    data_path = get_data_path()
    if not data_path.exists() or not any(data_path.iterdir()):
        logger.error(f"Nenhum arquivo encontrado na pasta {data_path}.")
        return

    logger.info("Carregando documentos...")
    loader = PyPDFDirectoryLoader(str(data_path))
    documentos = loader.load()

    logger.info(f"{len(documentos)} páginas carregadas. Quebrando em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(documentos)

    data_ingestao = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["data_ingestao"] = data_ingestao
        chunk.metadata["tenant_id"] = tenant_id

    logger.info(f"{len(chunks)} chunks gerados. Criando embeddings no ChromaDB...")
    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=str(get_chroma_path()),
    )
    logger.info("Ingestão concluída com sucesso!")


async def buscar_no_banco_vetorial(
    query: str,
    vectorstore: Chroma,
    k: int = 4,
    tenant_id: str = "default_tenant",
) -> str:
    """Busca trechos relevantes respeitando o isolamento de tenant e o limiar de similaridade."""
    if not vectorstore:
        return "Erro do sistema: Conexão com o banco vetorial indisponível."

    score_threshold = float(os.getenv("RAG_SCORE_THRESHOLD", "0.35"))

    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": score_threshold,
            "filter": {"tenant_id": tenant_id},
        },
    )

    docs: list[Document] = await retriever.ainvoke(query)

    if not docs:
        return "Nenhuma informação relevante encontrada nos documentos internos autorizados."

    blocos_formatados: list[str] = []
    for doc in docs:
        raw_source = doc.metadata.get("source")
        source = Path(raw_source).name if raw_source else "Desconhecida"
        page = doc.metadata.get("page", "N/A")
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        data_ingestao = doc.metadata.get("data_ingestao", "N/A")

        cabecalho = (
            f"REFERÊNCIA | Arquivo: {source} | Pág: {page} | "
            f"ID: {chunk_id} | Ingestão: {data_ingestao}"
        )
        blocos_formatados.append(f"[{cabecalho}]\n{doc.page_content}")

    return "\n\n---\n\n".join(blocos_formatados)


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    ingerir_documentos()