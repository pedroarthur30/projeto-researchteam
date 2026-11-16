# Research Team — Pipeline de Inteligência Competitiva

Pipeline de pesquisa e geração de dossiês executivos sobre empresas, combinando busca em documentos internos (RAG com ChromaDB) e pesquisa web (Tavily + scraping controlado). Construído com FastAPI, LangChain (LCEL) e Pydantic, com suporte a execução via API REST (síncrona ou streaming SSE), CLI e Docker.

## Como funciona

O fluxo completo em `src/projeto_researchteam/pipeline.py` segue três etapas encadeadas via LCEL:

1. **Coleta agêntica (`agent.py` / `tools.py`)**: um agente (`gpt-4o-mini` por padrão) consulta primeiro a base vetorial interna (`consultar_documentos_internos`) para verificar relatórios proprietários e, em seguida, complementa com buscas na web (`buscar_na_web`) e leitura pontual de páginas (`extrair_conteudo_url`). A etapa tem timeout de 45s, `recursion_limit=8` e orçamento por requisição (`MAX_WEB_SEARCHES=3`, `MAX_URL_EXTRACTIONS=2`).
2. **Extração estruturada em paralelo (`pipeline.py` / `schemas.py`)**: um `RunnableParallel` dispara duas chains simultâneas com `with_structured_output` para consolidar os dados brutos em dois schemas Pydantic: `AnaliseFinanceira` e `AnaliseReputacao`.
3. **Redação com fallback**: um modelo redator (`gpt-4o` por padrão) transforma os JSONs estruturados em um dossiê em Markdown dividido em quatro seções (Visão Geral & Modelo de Negócio, Cenário Financeiro & Concorrentes, Reputação, Movimentos Recentes & Riscos e Fontes Consultadas). Caso `ANTHROPIC_API_KEY` esteja configurada, o Claude (`claude-3-5-haiku-latest` por padrão) entra como fallback automático; caso contrário, o fallback usa `OPENAI_AGENT_MODEL`.

## Controles de segurança

- **Autenticação na API**: os endpoints `/dossie` e `/dossie/stream` exigem o header `X-API-Key`, validado em tempo constante com `secrets.compare_digest`. A aplicação falha no boot se `APP_API_KEY` não estiver definida.
- **Proteção contra SSRF**: `extrair_conteudo_url` resolve o DNS antes de conectar e bloqueia IPs privados, loopback, link-local, multicast e reservados — inclusive em redirecionamentos HTTP (máximo de 3). Também restringe esquemas a `http`/`https`, aceita apenas `text/html` ou `text/plain` e limita o payload a 2 MB.
- **Mitigação de Prompt Injection**: trechos retornados do RAG são encapsulados em tags `<dados>` com diretiva explícita instruindo o modelo a tratá-los como dados passivos.
- **Isolamento por tenant**: as consultas ao ChromaDB filtram pelo metadado `tenant_id` (propagado via `RunnableConfig`).

## Pré-requisitos

- Python `>= 3.14` (conforme `.python-version`)
- [uv](https://docs.astral.sh/uv/) para gerenciamento do ambiente virtual e dependências
- Chaves de API da OpenAI e do Tavily

## Configuração do ambiente

Clone o repositório e instale as dependências com o `uv`:

```bash
uv sync
```

Crie o arquivo `.env` a partir do modelo e preencha as chaves obrigatórias:

```bash
cp .env.example .env
```

Variáveis principais no `.env`:

| Variável | Obrigatória | Descrição |
| --- | --- | --- |
| `OPENAI_API_KEY` | Sim | Chave da OpenAI (embeddings `text-embedding-3-small` e modelos LLM) |
| `TAVILY_API_KEY` | Sim | Chave do Tavily para buscas na web |
| `APP_API_KEY` | Sim (API) | Chave esperada no header `X-API-Key` |
| `OPENAI_AGENT_MODEL` | Não | Modelo do agente pesquisador (padrão: `gpt-4o-mini`) |
| `OPENAI_EXTRACTOR_MODEL` | Não | Modelo de extração estruturada (padrão: `gpt-4o-mini`) |
| `OPENAI_WRITER_MODEL` | Não | Modelo da redação final (padrão: `gpt-4o`) |
| `ANTHROPIC_API_KEY` | Não | Ativa o Claude como fallback do redator |
| `ANTHROPIC_BACKUP_MODEL` | Não | Modelo Anthropic de fallback (padrão: `claude-3-5-haiku-latest`) |
| `RAG_SCORE_THRESHOLD` | Não | Limiar de similaridade do ChromaDB (padrão: `0.35`) |
| `DATA_PATH` / `CHROMA_PATH` | Não | Caminhos customizados para PDFs e banco vetorial (padrões: `./data` e `./chroma_db`) |
| `LANGSMITH_*` | Não | Variáveis de tracing do LangSmith |

Para gerar uma `APP_API_KEY` aleatória:

```bash
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Ingestão de documentos (RAG)

Os arquivos PDF usados na base interna ficam em `data/`. O repositório já inclui `data/relatorio_nexatech.pdf`, que trata de uma **empresa fictícia** (`NexaTech Solutions`) e serve **apenas para exemplificar o funcionamento do RAG** — você pode adicionar ou substituir pelos seus próprios PDFs antes de indexar. Para indexar os documentos no ChromaDB:

```bash
uv run python -m projeto_researchteam.rag
```

O script quebra os PDFs em chunks de 1000 caracteres (overlap de 200), gera os embeddings com `text-embedding-3-small` e persiste o índice em `chroma_db/` com `tenant_id="default_tenant"`.

## Executando o projeto

Na execução (via CLI ou API), você pode informar **qualquer empresa** — caso não existam documentos sobre ela na base interna do RAG, o agente fará a pesquisa normalmente utilizando a busca na web e extração de páginas. Os exemplos abaixo usam `NexaTech Solutions` apenas para exercitar o PDF fictício de exemplo.

### Via CLI

Faz a pesquisa e imprime o dossiê em streaming direto no terminal:

```bash
uv run projeto-researchteam "NexaTech Solutions"
```

### Via API FastAPI

Suba o servidor local:

```bash
uv run uvicorn projeto_researchteam.main:app --reload
```

Endpoints disponíveis:

- `GET /health` — healthcheck público (`{"status": "ok"}`).
- `POST /dossie` — executa o pipeline completo e retorna JSON com `empresa` e `dossie_markdown`.
- `POST /dossie/stream` — retorna o dossiê em tempo real via Server-Sent Events (`start`, `token`, `complete`, `error`).

Exemplo de chamada síncrona:

```bash
curl -X POST http://localhost:8000/dossie \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sua-app-api-key" \
  -d '{"empresa": "NexaTech Solutions"}'
```

Exemplo consumindo o stream SSE:

```bash
curl -N -X POST http://localhost:8000/dossie/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sua-app-api-key" \
  -d '{"empresa": "Nubank"}'
```

### Via Docker

O `Dockerfile` usa `python:3.14-slim`, instala as dependências de produção via `uv sync --frozen --no-dev` e roda o processo com o usuário não-root `appuser`:

```bash
docker build -t projeto-researchteam .
docker run --env-file .env -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/chroma_db:/app/chroma_db \
  projeto-researchteam
```

## Testes

A suíte de testes cobre validação de payload, autenticação, streaming SSE, bloqueio de SSRF (IPs internos e redirecionamentos), orçamento das ferramentas e encapsulamento de dados do RAG. Tudo roda com mocks/fakes locais, sem consumir APIs externas:

```bash
uv run pytest
```

## Estrutura do projeto

```text
├── data/                         # PDFs internos para ingestão no RAG
├── chroma_db/                    # Persistência local do ChromaDB
├── src/projeto_researchteam/
│   ├── agent.py                  # Configuração do agente pesquisador e prompt
│   ├── cli.py                    # Entrypoint CLI (projeto-researchteam)
│   ├── main.py                   # API FastAPI, CORS, auth e SSE
│   ├── pipeline.py               # Orquestração LCEL, extração paralela e fallback
│   ├── rag.py                    # Ingestão de PDFs e consulta no ChromaDB
│   ├── schemas.py                # Modelos Pydantic de entrada, saída e extração
│   └── tools.py                  # Ferramentas (Tavily, fetch anti-SSRF, RAG)
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── test_security_and_tools.py
├── Dockerfile
├── pyproject.toml
└── uv.lock
```
