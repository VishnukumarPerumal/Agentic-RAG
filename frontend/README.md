# Agentic RAG Frontend (`Next.js` App Router)

This Next.js frontend provides the interactive chat UI for the **LangGraph Multi-Agent RAG System**, connected to the FastAPI backend (`http://localhost:8000`).

## Features

1. **Live Multi-Agent Pipeline Execution Stepper (In-Place Loading UI)**:
   - Instead of a generic spinner, submitting a query renders an animated, real-time **LangGraph Agent Pipeline Stepper** streamed via Server-Sent Events (`POST /api/v1/chat/stream`).
   - Shows live status, elapsed timer, progress bar, and sub-step badges as the query progresses through:
     1. **Step 1: Guardrail Agent** (`Prompt Injection Check`, `SQL Injection Filter`, `PII Redaction`, `Safety Policy`)
     2. **Step 2: Data Retrieval Agent** (`pgvector HNSW Search`, `PostgreSQL Relational DB`, `FastMCP Tools`)
     3. **Step 3: Answer Generation Agent** (`Basic RAG System Prompt Synthesis`)
   - Once completed, collapses into an expandable `<details>` disclosure above the answer so you can inspect every step and source count at any time.

2. **Interactive Retrieval Source Toggles**:
   - Enable or disable `Vector DB (pgvector)`, `PostgreSQL DB`, and `MCP Tools` per query from the sidebar.

3. **Document Upload & `pgvector` Ingestion Modal (`<dialog closedby="any">`)**:
   - Upload `.pdf`, `.md`, `.txt`, `.csv`, `.json`, or `.html` files (or paste raw text) with configurable `chunk_size` and `chunk_overlap` directly into PostgreSQL `pgvector`.

## Running the Frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000` in your browser.
