# Agentic RAG System (`LangGraph` + `FastAPI` + `PostgreSQL / pgvector` + `MCP` + `Next.js` + `Ollama`)

A full-stack Multi-Agent Retrieval-Augmented Generation (RAG) platform orchestrated with **LangGraph**, backed by **PostgreSQL + `pgvector`**, **Model Context Protocol (MCP)** tools, **Ollama (`llama3.2`)** word-by-word answer streaming, and a **Next.js** ChatGPT-style interface with live step-by-step agent execution progress.

---

## Architecture Overview

1. **Guardrail Agent (`guardrail_agent.py`)**:
   - Screens all incoming user queries for system prompt extraction, prompt injection, jailbreaks, destructive SQL commands, credential probing, and inline PII (SSN/credit card redaction) plus a secondary Ollama security classifier.
2. **Data Retrieval Agent (`retrieval_agent.py`)**:
   - Retrieves context across 3 sources:
     1. **Vector DB (`pgvector`)**: Cosine similarity search (`<=>`) over chunked document embeddings (`document_chunks`).
     2. **PostgreSQL Relational DB**: Structured lookups across `products`, `customer_orders`, and `support_policies`.
     3. **MCP Tools (`mcp_server.py`)**: Executes FastMCP tools (`calculate_expression`, `get_inventory_and_order_analytics`, `lookup_policy_by_topic`, `get_system_status`).
3. **Answer Generation Agent (`answer_agent.py`)**:
   - Combines retrieved facts with a grounded RAG prompt and streams the answer **word-by-word** from Ollama (`llama3.2:latest`).

---

## Project Structure

```text
Agentic RAG/
├── .gitignore
├── README.md
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routes/
│   │   ├── services/
│   │   ├── repository/
│   │   ├── models/
│   │   ├── db/
│   │   │   ├── database.py
│   │   │   └── schemas/
│   │   └── utilities/
│   ├── mcp_server.py
│   ├── verify_backend.py
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── package.json
    ├── tsconfig.json
    ├── next.config.mjs
    └── src/
        └── app/
            ├── globals.css
            ├── layout.tsx
            └── page.tsx
```

---

## Step-by-Step Installation Guide

### Prerequisites
- **Python 3.11+**
- **Node.js 18+** & **npm**
- **PostgreSQL 16+** with **`pgvector`** extension (`brew install postgresql@18 pgvector` on macOS)
- **Ollama** installed locally (`https://ollama.com`)

---

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd "Agentic RAG"
```

---

### Step 2: Set Up PostgreSQL & Enable `pgvector`

Initialize and start a dedicated local PostgreSQL server on port `5433` (or use your existing PostgreSQL server):

```bash
initdb -D ./backend/.pgdata -U postgres --auth=trust --encoding=UTF8 --no-locale
pg_ctl -D ./backend/.pgdata -l ./backend/.pgdata/server.log -o "-p 5433 -k /tmp" start

# Create the database and enable pgvector
psql -h localhost -p 5433 -U postgres -c "CREATE DATABASE agentic_rag;"
psql -h localhost -p 5433 -U postgres -d agentic_rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

---

### Step 3: Pull the Ollama Small Model

Make sure Ollama is running and pull `llama3.2:latest`:

```bash
ollama pull llama3.2:latest
```

---

### Step 4: Configure & Start the FastAPI Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Start the FastAPI server on port 8001
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

- **Backend API**: `http://localhost:8001`
- **Swagger UI Docs**: `http://localhost:8001/docs`

---

### Step 5: Configure & Start the Next.js Frontend

Open a new terminal window:

```bash
cd frontend
npm install

# Configure backend API URL
echo 'NEXT_PUBLIC_API_BASE_URL="http://localhost:8001"' > .env.local

# Start Next.js development server on port 3000
npm run dev
```

- **Chat UI**: `http://localhost:3000`
