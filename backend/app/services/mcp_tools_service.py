"""Service layer for discovering and invoking Model Context Protocol (MCP) tools."""

import re
from mcp_server import (
    calculate_expression,
    get_inventory_and_order_analytics,
    get_system_status,
    lookup_policy_by_topic,
    mcp_server,
)


class MCPToolsService:
    """Provides MCP tool discovery and intent-driven invocation for the Data Retrieval Agent."""

    def list_tools(self) -> list[dict]:
        """List all registered MCP tools and their descriptions."""
        return [
            {
                "server": mcp_server.name,
                "name": "get_system_status",
                "description": "Get live RAG system status, current UTC timestamp, and PostgreSQL + pgvector counts.",
                "parameters": {},
            },
            {
                "server": mcp_server.name,
                "name": "calculate_expression",
                "description": "Safely evaluate a mathematical expression (e.g., '2 * 14500 + 499').",
                "parameters": {"expression": "string"},
            },
            {
                "server": mcp_server.name,
                "name": "get_inventory_and_order_analytics",
                "description": "Compute live inventory valuation and order revenue analytics from PostgreSQL.",
                "parameters": {},
            },
            {
                "server": mcp_server.name,
                "name": "lookup_policy_by_topic",
                "description": "Search company policies in PostgreSQL by policy code, title, or keyword.",
                "parameters": {"topic": "string"},
            },
        ]

    def call_tool(self, tool_name: str, arguments: dict | None = None) -> dict:
        """Invoke a specific MCP tool by name."""
        args = arguments or {}
        if tool_name == "get_system_status":
            return get_system_status()
        if tool_name == "calculate_expression":
            return calculate_expression(expression=str(args.get("expression", "0")))
        if tool_name == "get_inventory_and_order_analytics":
            return get_inventory_and_order_analytics()
        if tool_name == "lookup_policy_by_topic":
            return lookup_policy_by_topic(topic=str(args.get("topic", "")))
        raise ValueError(f"Unknown MCP tool: {tool_name}")

    def execute_relevant_tools_for_query(self, query: str) -> list[dict]:
        """
        Inspect user query intent and automatically execute matching MCP tools.
        Returns a list of tool execution outputs.
        """
        q_lower = (query or "").lower()
        outputs: list[dict] = []

        # 1. Detect arithmetic expressions in query (e.g. "14500 * 2" or "calculate 499 * 12")
        math_match = re.search(
            r"(\d+(?:\.\d+)?(?:\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)+)", q_lower
        )
        if math_match:
            expr = math_match.group(1)
            try:
                outputs.append(self.call_tool("calculate_expression", {"expression": expr}))
            except Exception:
                pass

        # 2. Detect analytics / revenue / valuation / total inventory queries
        if any(
            kw in q_lower
            for kw in [
                "total", "revenue", "analytics", "valuation", "inventory",
                "summary", "how many orders", "how many products", "statistics", "stats",
            ]
        ):
            outputs.append(self.call_tool("get_inventory_and_order_analytics"))

        # 3. Detect system status / database health / chunk count / time queries
        if any(
            kw in q_lower
            for kw in [
                "status", "health", "time", "timestamp", "database",
                "chunks", "vector", "system", "mcp",
            ]
        ):
            outputs.append(self.call_tool("get_system_status"))

        # 4. Detect policy / SLA / refund / compliance lookup via MCP tool
        for topic_kw in ["refund", "return", "sla", "uptime", "privacy", "guardrail", "security"]:
            if topic_kw in q_lower:
                outputs.append(self.call_tool("lookup_policy_by_topic", {"topic": topic_kw}))
                break

        return outputs
