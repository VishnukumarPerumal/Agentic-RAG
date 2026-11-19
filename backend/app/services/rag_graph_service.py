"""LangGraph StateGraph orchestrating the Guardrail Agent, Data Retrieval Agent, and Answer Generation Agent."""

from typing import Literal, TypedDict
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session
from app.db.schemas.chat_schemas import (
    AgentStepTraceOut,
    ChatQueryRequest,
    ChatQueryResponse,
    GuardrailEvaluationOut,
    RetrievedContextOut,
)
from app.repository.document_repository import DocumentRepository
from app.services.answer_agent import AnswerGenerationAgent
from app.services.guardrail_agent import GuardrailAgent
from app.services.retrieval_agent import DataRetrievalAgent


class RAGGraphState(TypedDict):
    """Shared state dictionary passed across LangGraph nodes."""

    session_id: str
    query: str
    top_k: int
    use_vector_db: bool
    use_postgres_db: bool
    use_mcp_tools: bool
    guardrail: dict
    vector_chunks: list[dict]
    postgres_records: list[dict]
    mcp_tool_outputs: list[dict]
    answer: str
    llm_provider: str
    agent_trace: list[dict]


class RAGGraphService:
    """Builds and executes the 3-agent LangGraph RAG pipeline."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.guardrail_agent = GuardrailAgent()
        self.retrieval_agent = DataRetrievalAgent(db)
        self.answer_agent = AnswerGenerationAgent()
        self.doc_repo = DocumentRepository(db)
        self.compiled_graph = self._build_graph()

    def _guardrail_node(self, state: RAGGraphState) -> dict:
        evaluation = self.guardrail_agent.evaluate(state["query"])
        trace = list(state.get("agent_trace", []))
        trace.append(
            {
                "agent": "guardrail_agent",
                "status": "PASSED" if evaluation["allowed"] else "BLOCKED",
                "summary": evaluation["reason"],
                "details": {
                    "category": evaluation["category"],
                    "risk_score": evaluation["risk_score"],
                },
            }
        )
        return {
            "guardrail": evaluation,
            "agent_trace": trace,
        }

    def _route_after_guardrail(
        self, state: RAGGraphState
    ) -> Literal["data_retrieval_agent", "guardrail_blocked_node"]:
        if state["guardrail"].get("allowed", False):
            return "data_retrieval_agent"
        return "guardrail_blocked_node"

    def _guardrail_blocked_node(self, state: RAGGraphState) -> dict:
        reason = state["guardrail"].get(
            "reason", "Blocked by Guardrail Agent due to policy violation."
        )
        category = state["guardrail"].get("category", "POLICY_VIOLATION")
        refusal_message = (
            f"⚠️ **Request Blocked by Guardrail Agent** (`{category}`)\n\n"
            f"{reason}\n\n"
            "Please rephrase your request without prompt overrides, destructive commands, or prohibited content."
        )
        trace = list(state.get("agent_trace", []))
        trace.append(
            {
                "agent": "guardrail_blocked_node",
                "status": "SHORT_CIRCUITED",
                "summary": "Retrieval and generation skipped due to guardrail block.",
                "details": {"category": category},
            }
        )
        return {
            "answer": refusal_message,
            "llm_provider": "guardrail_short_circuit",
            "agent_trace": trace,
        }

    def _data_retrieval_node(self, state: RAGGraphState) -> dict:
        sanitized_query = state["guardrail"].get("sanitized_query", state["query"])
        retrieved = self.retrieval_agent.retrieve(
            query=sanitized_query,
            top_k=state["top_k"],
            use_vector_db=state["use_vector_db"],
            use_postgres_db=state["use_postgres_db"],
            use_mcp_tools=state["use_mcp_tools"],
        )

        v_chunks = retrieved["vector_chunks"]
        pg_rows = retrieved["postgres_records"]
        mcp_out = retrieved["mcp_tool_outputs"]

        trace = list(state.get("agent_trace", []))
        trace.append(
            {
                "agent": "data_retrieval_agent",
                "status": "COMPLETED",
                "summary": (
                    f"Retrieved {len(v_chunks)} pgvector chunks, "
                    f"{len(pg_rows)} PostgreSQL records, and "
                    f"{len(mcp_out)} MCP tool outputs."
                ),
                "details": {
                    "vector_chunks_count": len(v_chunks),
                    "postgres_records_count": len(pg_rows),
                    "mcp_tools_executed": [m.get("tool") for m in mcp_out],
                },
            }
        )
        return {
            "vector_chunks": v_chunks,
            "postgres_records": pg_rows,
            "mcp_tool_outputs": mcp_out,
            "agent_trace": trace,
        }

    def _answer_generation_node(self, state: RAGGraphState) -> dict:
        sanitized_query = state["guardrail"].get("sanitized_query", state["query"])
        answer_text, provider_name = self.answer_agent.generate_answer(
            query=sanitized_query,
            vector_chunks=state.get("vector_chunks", []),
            postgres_records=state.get("postgres_records", []),
            mcp_tool_outputs=state.get("mcp_tool_outputs", []),
        )

        trace = list(state.get("agent_trace", []))
        trace.append(
            {
                "agent": "answer_generation_agent",
                "status": "COMPLETED",
                "summary": f"Generated grounded response using basic RAG prompt ({provider_name}).",
                "details": {
                    "provider": provider_name,
                    "answer_length": len(answer_text),
                },
            }
        )
        return {
            "answer": answer_text,
            "llm_provider": provider_name,
            "agent_trace": trace,
        }

    def _build_graph(self):
        workflow = StateGraph(RAGGraphState)

        workflow.add_node("guardrail_agent", self._guardrail_node)
        workflow.add_node("guardrail_blocked_node", self._guardrail_blocked_node)
        workflow.add_node("data_retrieval_agent", self._data_retrieval_node)
        workflow.add_node("answer_generation_agent", self._answer_generation_node)

        workflow.add_edge(START, "guardrail_agent")
        workflow.add_conditional_edges(
            "guardrail_agent",
            self._route_after_guardrail,
            {
                "data_retrieval_agent": "data_retrieval_agent",
                "guardrail_blocked_node": "guardrail_blocked_node",
            },
        )
        workflow.add_edge("guardrail_blocked_node", END)
        workflow.add_edge("data_retrieval_agent", "answer_generation_agent")
        workflow.add_edge("answer_generation_agent", END)

        return workflow.compile()

    def run_chat_query(self, payload: ChatQueryRequest) -> ChatQueryResponse:
        initial_state: RAGGraphState = {
            "session_id": payload.session_id,
            "query": payload.query,
            "top_k": payload.top_k,
            "use_vector_db": payload.use_vector_db,
            "use_postgres_db": payload.use_postgres_db,
            "use_mcp_tools": payload.use_mcp_tools,
            "guardrail": {},
            "vector_chunks": [],
            "postgres_records": [],
            "mcp_tool_outputs": [],
            "answer": "",
            "llm_provider": "",
            "agent_trace": [],
        }

        final_state = self.compiled_graph.invoke(initial_state)

        sources_summary = {
            "vector_chunks_count": len(final_state["vector_chunks"]),
            "postgres_records_count": len(final_state["postgres_records"]),
            "mcp_tools_count": len(final_state["mcp_tool_outputs"]),
        }

        history_record = self.doc_repo.save_chat_history(
            session_id=payload.session_id,
            user_query=payload.query,
            assistant_response=final_state["answer"],
            guardrail_passed=bool(final_state["guardrail"].get("allowed", False)),
            sources_used=sources_summary,
            agent_trace=final_state["agent_trace"],
        )

        return ChatQueryResponse(
            session_id=payload.session_id,
            query=payload.query,
            answer=final_state["answer"],
            guardrail=GuardrailEvaluationOut(**final_state["guardrail"]),
            retrieved_context=RetrievedContextOut(
                vector_chunks=final_state["vector_chunks"],
                postgres_records=final_state["postgres_records"],
                mcp_tool_outputs=final_state["mcp_tool_outputs"],
            ),
            agent_trace=[AgentStepTraceOut(**step) for step in final_state["agent_trace"]],
            llm_provider=final_state["llm_provider"],
            created_at=history_record.created_at,
        )

    def stream_chat_query(self, payload: ChatQueryRequest):
        """
        Execute the LangGraph workflow and yield real-time Server-Sent Events (SSE):
        1. Guardrail Agent step_start -> step_complete
        2. Data Retrieval Agent step_start -> step_complete
        3. Answer Generation Agent step_start -> word-by-word answer_token events from Ollama -> step_complete -> final_response
        """
        import json
        import re
        import time

        initial_state: RAGGraphState = {
            "session_id": payload.session_id,
            "query": payload.query,
            "top_k": payload.top_k,
            "use_vector_db": payload.use_vector_db,
            "use_postgres_db": payload.use_postgres_db,
            "use_mcp_tools": payload.use_mcp_tools,
            "guardrail": {},
            "vector_chunks": [],
            "postgres_records": [],
            "mcp_tool_outputs": [],
            "answer": "",
            "llm_provider": "",
            "agent_trace": [],
        }

        running_state: dict = dict(initial_state)

        # --- STEP 1: Guardrail Agent ---
        yield f"data: {json.dumps({'event': 'step_start', 'agent': 'guardrail_agent', 'title': 'Guardrail Agent', 'detail': 'Scanning query for prompt injection, SQL injection, safety policy, and PII...'})}\n\n"
        time.sleep(0.25)

        guardrail_update = self._guardrail_node(running_state)
        running_state.update(guardrail_update)
        g_trace = running_state["agent_trace"][-1]
        yield f"data: {json.dumps({'event': 'step_complete', 'agent': 'guardrail_agent', 'trace': g_trace})}\n\n"

        if not running_state["guardrail"].get("allowed", False):
            blocked_update = self._guardrail_blocked_node(running_state)
            running_state.update(blocked_update)
            b_trace = running_state["agent_trace"][-1]
            yield f"data: {json.dumps({'event': 'step_complete', 'agent': 'guardrail_blocked_node', 'trace': b_trace})}\n\n"

            for tok in re.split(r"(\s+)", running_state["answer"]):
                if tok:
                    yield f"data: {json.dumps({'event': 'answer_token', 'token': tok, 'provider': running_state['llm_provider']})}\n\n"
                    time.sleep(0.015)
        else:
            # --- STEP 2: Data Retrieval Agent ---
            active_sources = []
            if payload.use_vector_db:
                active_sources.append("pgvector DB")
            if payload.use_postgres_db:
                active_sources.append("PostgreSQL Tables")
            if payload.use_mcp_tools:
                active_sources.append("MCP Tools")
            sources_label = ", ".join(active_sources) or "configured sources"

            yield f"data: {json.dumps({'event': 'step_start', 'agent': 'data_retrieval_agent', 'title': 'Data Retrieval Agent', 'detail': f'Querying {sources_label}...'})}\n\n"
            time.sleep(0.25)

            retrieval_update = self._data_retrieval_node(running_state)
            running_state.update(retrieval_update)
            r_trace = running_state["agent_trace"][-1]
            yield f"data: {json.dumps({'event': 'step_complete', 'agent': 'data_retrieval_agent', 'trace': r_trace})}\n\n"

            # --- STEP 3: Answer Generation Agent (Word-by-Word Stream via Ollama) ---
            yield f"data: {json.dumps({'event': 'step_start', 'agent': 'answer_generation_agent', 'title': 'Answer Generation Agent', 'detail': 'Streaming answer from Ollama small model (llama3.2:latest)...'})}\n\n"

            sanitized_query = running_state["guardrail"].get(
                "sanitized_query", running_state["query"]
            )
            collected_tokens: list[str] = []
            provider_used = "ollama:llama3.2:latest"

            for token_chunk, prov in self.answer_agent.stream_answer(
                query=sanitized_query,
                vector_chunks=running_state.get("vector_chunks", []),
                postgres_records=running_state.get("postgres_records", []),
                mcp_tool_outputs=running_state.get("mcp_tool_outputs", []),
            ):
                provider_used = prov
                collected_tokens.append(token_chunk)
                yield f"data: {json.dumps({'event': 'answer_token', 'token': token_chunk, 'provider': provider_used})}\n\n"

            full_answer = "".join(collected_tokens).strip()
            running_state["answer"] = full_answer
            running_state["llm_provider"] = provider_used

            a_trace = {
                "agent": "answer_generation_agent",
                "status": "COMPLETED",
                "summary": f"Streamed grounded response via {provider_used} ({len(full_answer)} chars).",
                "details": {
                    "provider": provider_used,
                    "answer_length": len(full_answer),
                },
            }
            running_state["agent_trace"] = list(running_state["agent_trace"]) + [a_trace]
            yield f"data: {json.dumps({'event': 'step_complete', 'agent': 'answer_generation_agent', 'trace': a_trace})}\n\n"

        sources_summary = {
            "vector_chunks_count": len(running_state["vector_chunks"]),
            "postgres_records_count": len(running_state["postgres_records"]),
            "mcp_tools_count": len(running_state["mcp_tool_outputs"]),
        }

        history_record = self.doc_repo.save_chat_history(
            session_id=payload.session_id,
            user_query=payload.query,
            assistant_response=running_state["answer"],
            guardrail_passed=bool(running_state["guardrail"].get("allowed", False)),
            sources_used=sources_summary,
            agent_trace=running_state["agent_trace"],
        )

        final_response = ChatQueryResponse(
            session_id=payload.session_id,
            query=payload.query,
            answer=running_state["answer"],
            guardrail=GuardrailEvaluationOut(**running_state["guardrail"]),
            retrieved_context=RetrievedContextOut(
                vector_chunks=running_state["vector_chunks"],
                postgres_records=running_state["postgres_records"],
                mcp_tool_outputs=running_state["mcp_tool_outputs"],
            ),
            agent_trace=[AgentStepTraceOut(**step) for step in running_state["agent_trace"]],
            llm_provider=running_state["llm_provider"],
            created_at=history_record.created_at,
        )

        yield f"data: {json.dumps({'event': 'final_response', 'data': final_response.model_dump(mode='json')})}\n\n"
