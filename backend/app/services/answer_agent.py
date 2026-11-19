"""Answer Generation Agent: synthesizes final response from retrieved multi-source context using a basic RAG prompt and streams word-by-word via Ollama."""

from collections.abc import Generator
import json
import re
import time
from app.utilities.llm_provider import generate_llm_response, stream_llm_response

BASIC_RAG_SYSTEM_PROMPT = """You are a helpful and accurate Answer Generation Agent in an Agentic RAG system.
Answer the user's question using ONLY the provided context retrieved from:
1. Vector DB Document Chunks (pgvector)
2. PostgreSQL Relational Database Records
3. MCP Tool Outputs

Formatting & Grounding Rules:
- Format your response in clean Markdown using short section headings (`###`), bullet points (`-`), and **bold** text for key numbers, names, and specifications.
- Cite the source inline (e.g., `[Vector DB: filename]`, `[PostgreSQL: products]`, or `[MCP Tool: tool_name]`) when referencing facts.
- If the retrieved context does not contain enough information to answer the question, state that clearly without guessing.
"""


class AnswerGenerationAgent:
    """
    Third node in the LangGraph RAG workflow.
    Formats retrieved context from Vector DB, PostgreSQL DB, and MCP Tools into a basic prompt
    and generates or streams a grounded answer via Ollama (llama3.2:latest).
    """

    def build_user_prompt(
        self,
        query: str,
        vector_chunks: list[dict],
        postgres_records: list[dict],
        mcp_tool_outputs: list[dict],
    ) -> str:
        sections: list[str] = []

        if vector_chunks:
            chunk_lines = [
                f"- [Vector DB | {c['filename']} | Chunk #{c['chunk_index']} | Similarity={c['similarity_score']}]: {c['content']}"
                for c in vector_chunks
            ]
            sections.append("### Vector DB Chunks (pgvector)\n" + "\n".join(chunk_lines))
        else:
            sections.append("### Vector DB Chunks (pgvector)\nNone retrieved.")

        if postgres_records:
            pg_lines = [
                f"- [PostgreSQL | Table={r['table']} | ID={r['record_id']}]: {r['summary']}"
                for r in postgres_records
            ]
            sections.append("### PostgreSQL Relational Data\n" + "\n".join(pg_lines))
        else:
            sections.append("### PostgreSQL Relational Data\nNone retrieved.")

        if mcp_tool_outputs:
            mcp_lines = [
                f"- [MCP Tool | {m.get('tool', 'tool')}]: {json.dumps(m, ensure_ascii=False)}"
                for m in mcp_tool_outputs
            ]
            sections.append("### MCP Tool Outputs\n" + "\n".join(mcp_lines))
        else:
            sections.append("### MCP Tool Outputs\nNone executed.")

        context_block = "\n\n".join(sections)
        return f"{context_block}\n\n### User Question\n{query}\n\n### Answer"

    def generate_answer(
        self,
        query: str,
        vector_chunks: list[dict],
        postgres_records: list[dict],
        mcp_tool_outputs: list[dict],
    ) -> tuple[str, str]:
        """Generate grounded RAG answer synchronously and return (answer_text, provider_used)."""
        user_prompt = self.build_user_prompt(
            query=query,
            vector_chunks=vector_chunks,
            postgres_records=postgres_records,
            mcp_tool_outputs=mcp_tool_outputs,
        )

        llm_text, provider_name = generate_llm_response(
            system_prompt=BASIC_RAG_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        if llm_text:
            return llm_text, provider_name

        return (
            self._synthesize_grounded_response(
                query=query,
                vector_chunks=vector_chunks,
                postgres_records=postgres_records,
                mcp_tool_outputs=mcp_tool_outputs,
            ),
            provider_name,
        )

    def stream_answer(
        self,
        query: str,
        vector_chunks: list[dict],
        postgres_records: list[dict],
        mcp_tool_outputs: list[dict],
    ) -> Generator[tuple[str, str], None, None]:
        """
        Stream the generated answer token-by-token / word-by-word from Ollama (llama3.2:latest).
        Yields (token_chunk, provider_name).
        """
        user_prompt = self.build_user_prompt(
            query=query,
            vector_chunks=vector_chunks,
            postgres_records=postgres_records,
            mcp_tool_outputs=mcp_tool_outputs,
        )

        yielded = False
        for token, provider_name in stream_llm_response(
            system_prompt=BASIC_RAG_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        ):
            yielded = True
            yield token, provider_name

        if yielded:
            return

        # Fallback word-by-word stream if Ollama daemon is offline
        fallback_text = self._synthesize_grounded_response(
            query=query,
            vector_chunks=vector_chunks,
            postgres_records=postgres_records,
            mcp_tool_outputs=mcp_tool_outputs,
        )
        words = re.split(r"(\s+)", fallback_text)
        for w in words:
            if w:
                yield w, "local_rag_synthesizer"
                time.sleep(0.015)

    def _synthesize_grounded_response(
        self,
        query: str,
        vector_chunks: list[dict],
        postgres_records: list[dict],
        mcp_tool_outputs: list[dict],
    ) -> str:
        """Synthesize a clean, source-cited response directly from the retrieved context."""
        q_tokens = {
            tok
            for tok in re.findall(r"[a-z0-9]+", (query or "").lower())
            if len(tok) > 2
        }

        relevant_chunks = []
        for c in vector_chunks:
            c_lower = c["content"].lower()
            token_hits = sum(1 for t in q_tokens if t in c_lower)
            if c["similarity_score"] >= 0.08 or token_hits > 0:
                relevant_chunks.append((token_hits, c["similarity_score"], c))
        relevant_chunks.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top_chunks = [item[2] for item in relevant_chunks[:3]]

        if not top_chunks and not postgres_records and not mcp_tool_outputs:
            return (
                "I could not find relevant information in the Vector DB (pgvector), "
                "PostgreSQL database tables, or MCP tools to answer your question. "
                "Please upload relevant documents via `/api/v1/ingest/upload` or refine your query."
            )

        parts: list[str] = ["### Retrieved Multi-Source Summary\n"]

        if top_chunks:
            parts.append("#### Vector DB (`pgvector`)")
            for c in top_chunks:
                snippet = c["content"].strip()
                parts.append(
                    f"- **[{c['filename']} (Chunk #{c['chunk_index']}, similarity: {c['similarity_score']})]:** {snippet}"
                )

        if postgres_records:
            parts.append("\n#### PostgreSQL Relational Database")
            for r in postgres_records[:6]:
                parts.append(f"- **[PostgreSQL `{r['table']}`]:** {r['summary']}")

        if mcp_tool_outputs:
            parts.append("\n#### MCP Tools")
            for m in mcp_tool_outputs:
                tool_name = m.get("tool", "mcp_tool")
                if tool_name == "calculate_expression":
                    parts.append(
                        f"- **[MCP Tool `{tool_name}`]:** `{m.get('expression')}` = **{m.get('result')}**"
                    )
                elif tool_name == "get_inventory_and_order_analytics":
                    parts.append(
                        f"- **[MCP Tool `{tool_name}`]:** {m.get('total_product_skus')} product SKUs "
                        f"({m.get('total_inventory_units')} total units valued at "
                        f"${m.get('total_inventory_valuation_usd'):,.2f}), "
                        f"{m.get('total_orders')} customer orders totaling "
                        f"${m.get('total_order_revenue_usd'):,.2f} revenue."
                    )
                elif tool_name == "get_system_status":
                    parts.append(
                        f"- **[MCP Tool `{tool_name}`]:** Connected to `{m.get('pg_database')}` on "
                        f"`{m.get('pg_host')}:{m.get('pg_port')}` | "
                        f"Documents: {m.get('ingested_documents')}, Vector Chunks: {m.get('vector_chunks_stored')}, "
                        f"Timestamp: `{m.get('timestamp_utc')}`."
                    )
                elif tool_name == "lookup_policy_by_topic":
                    matches = m.get("matches", [])
                    if matches:
                        for pol in matches:
                            parts.append(
                                f"- **[MCP Tool `{tool_name}` | {pol['policy_code']}]:** {pol['title']} — {pol['content']}"
                            )
                else:
                    parts.append(f"- **[MCP Tool `{tool_name}`]:** {json.dumps(m)}")

        return "\n".join(parts)
