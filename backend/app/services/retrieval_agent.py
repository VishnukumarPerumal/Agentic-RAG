"""Data Retrieval Agent: retrieves multi-source context from pgvector DB, PostgreSQL relational tables, and MCP tools."""

from sqlalchemy.orm import Session
from app.repository.sql_data_repository import SQLDataRepository
from app.repository.vector_repository import VectorRepository
from app.services.mcp_tools_service import MCPToolsService
from app.utilities.llm_provider import generate_embedding


class DataRetrievalAgent:
    """
    Second node in the LangGraph RAG workflow.
    Queries:
      1. Vector DB (PostgreSQL pgvector document_chunks table)
      2. PostgreSQL DB Data (structured relational tables: products, customer_orders, support_policies)
      3. MCP Tools (FastMCP registered tools: analytics, math calculator, system status, policy lookup)
    """

    def __init__(self, db: Session) -> None:
        self.db = db
        self.vector_repo = VectorRepository(db)
        self.sql_repo = SQLDataRepository(db)
        self.mcp_service = MCPToolsService()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_vector_db: bool = True,
        use_postgres_db: bool = True,
        use_mcp_tools: bool = True,
    ) -> dict:
        vector_chunks: list[dict] = []
        postgres_records: list[dict] = []
        mcp_tool_outputs: list[dict] = []

        # 1. Vector DB Retrieval (pgvector cosine similarity search)
        if use_vector_db:
            query_embedding = generate_embedding(query)
            vector_chunks = self.vector_repo.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k,
            )

        # 2. PostgreSQL Structured Relational Data Retrieval
        if use_postgres_db:
            postgres_records = self.sql_repo.search_relational_data_for_query(query)

        # 3. MCP Tools Execution
        if use_mcp_tools:
            mcp_tool_outputs = self.mcp_service.execute_relevant_tools_for_query(query)

        return {
            "vector_chunks": vector_chunks,
            "postgres_records": postgres_records,
            "mcp_tool_outputs": mcp_tool_outputs,
        }
