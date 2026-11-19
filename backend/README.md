# Agentic RAG Backend (`FastAPI` + `LangGraph` + `PostgreSQL 18 / pgvector` + `MCP`)

This backend implements a multi-agent RAG system orchestrated with **LangGraph**, backed by a dedicated **PostgreSQL 18.4** server with the **`pgvector` (`0.8.6`)** extension enabled, a document **Ingestion Pipeline**, and **Model Context Protocol (MCP)** tools.

## Directory Structure

```text
backend/
├── app/
│   ├── main.py                      # FastAPI application entrypoint & lifespan bootstrap
│   ├── routes/
│   │   ├── ingestion_routes.py      # File upload, text chunking & pgvector similarity search endpoints
│   │   ├── chat_routes.py           # LangGraph multi-agent RAG query & chat history endpoints
│   │   └── data_routes.py           # PostgreSQL relational data & MCP tool endpoints
│   ├── services/
│   │   ├── guardrail_agent.py       # Guardrail Agent (prompt injection, SQL injection, safety & PII checks)
│   │   ├── retrieval_agent.py       # Data Retrieval Agent (pgvector DB + PostgreSQL DB + MCP Tools)
│   │   ├── answer_agent.py          # Answer Generation Agent (basic RAG prompt synthesis)
│   │   ├── rag_graph_service.py     # LangGraph StateGraph workflow compilation & execution
│   │   ├── ingestion_service.py     # Document parsing, chunking, embedding & pgvector storage
│   │   └── mcp_tools_service.py     # MCP tool discovery and intent-based tool execution
│   ├── repository/
│   │   ├── vector_repository.py     # pgvector chunk persistence & cosine distance (<=>) search
│   │   ├── document_repository.py   # Document metadata & ChatHistory CRUD
│   │   └── sql_data_repository.py   # PostgreSQL relational queries (products, customer_orders, support_policies)
│   ├── models/
│   │   ├── document_models.py       # SQLAlchemy ORM models: Document, DocumentChunk (Vector), ChatHistory
│   │   └── business_models.py       # SQLAlchemy ORM models: ProductRecord, CustomerOrder, SupportPolicy
│   ├── db/
│   │   ├── database.py              # SQLAlchemy engine, session, pgvector & HNSW index initialization
│   │   └── schemas/                 # Pydantic v2 request/response schemas
│   │       ├── chat_schemas.py
│   │       ├── document_schemas.py
│   │       └── data_schemas.py
│   └── utilities/
│       ├── config.py                # Environment & database configuration
│       ├── file_parser.py           # PDF, TXT, Markdown, CSV, JSON, HTML text extractor
│       ├── text_chunker.py          # RecursiveCharacterTextSplitter chunking utility
│       └── llm_provider.py          # OpenAI / Gemini / Ollama / deterministic local embeddings & synthesis
├── mcp_server.py                    # Standalone & in-process FastMCP server
├── verify_backend.py                # End-to-end verification test suite
├── requirements.txt
└── .env
```

## Managing the Dedicated PostgreSQL + `pgvector` Server

A dedicated PostgreSQL 18.4 cluster is initialized at `backend/.pgdata` and runs on port **`5433`** with database **`agentic_rag`** and extension **`vector` (`0.8.6`)**:

```bash
# Check status
pg_ctl -D ./backend/.pgdata status

# Start server
pg_ctl -D ./backend/.pgdata -l ./backend/.pgdata/server.log start

# Stop server
pg_ctl -D ./backend/.pgdata stop
```

## Running the FastAPI Backend Server

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Interactive OpenAPI docs: `http://localhost:8001/docs`
