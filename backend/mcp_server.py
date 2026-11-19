"""Standalone & in-process Model Context Protocol (MCP) Server using FastMCP."""

import ast
import operator
from datetime import datetime, timezone
from mcp.server.fastmcp import FastMCP
from sqlalchemy import func
from app.db.database import SessionLocal
from app.models.business_models import CustomerOrder, ProductRecord, SupportPolicy
from app.models.document_models import Document, DocumentChunk
from app.utilities.config import settings

mcp_server = FastMCP("Agentic-RAG-Data-Tools")

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _safe_eval_expr(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        left = _safe_eval_expr(node.left)
        right = _safe_eval_expr(node.right)
        return float(_ALLOWED_OPERATORS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        operand = _safe_eval_expr(node.operand)
        return float(_ALLOWED_OPERATORS[type(node.op)](operand))
    raise ValueError("Unsupported mathematical expression")


@mcp_server.tool()
def get_system_status() -> dict:
    """Get live RAG system status, current UTC timestamp, and PostgreSQL + pgvector counts."""
    with SessionLocal() as db:
        doc_count = db.query(func.count(Document.id)).scalar() or 0
        chunk_count = db.query(func.count(DocumentChunk.id)).scalar() or 0
        product_count = db.query(func.count(ProductRecord.id)).scalar() or 0
        order_count = db.query(func.count(CustomerOrder.id)).scalar() or 0

    return {
        "tool": "get_system_status",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pg_host": settings.pg_host,
        "pg_port": settings.pg_port,
        "pg_database": settings.pg_database,
        "embedding_dimension": settings.embedding_dimension,
        "ingested_documents": doc_count,
        "vector_chunks_stored": chunk_count,
        "postgres_products": product_count,
        "postgres_orders": order_count,
    }


@mcp_server.tool()
def calculate_expression(expression: str) -> dict:
    """Safely evaluate a mathematical expression (e.g., '2 * 14500 + 499')."""
    cleaned = (expression or "").strip()
    parsed = ast.parse(cleaned, mode="eval")
    result = _safe_eval_expr(parsed.body)
    return {
        "tool": "calculate_expression",
        "expression": cleaned,
        "result": round(result, 4),
    }


@mcp_server.tool()
def get_inventory_and_order_analytics() -> dict:
    """Compute live inventory valuation and order revenue analytics from PostgreSQL."""
    with SessionLocal() as db:
        products = db.query(ProductRecord).all()
        orders = db.query(CustomerOrder).all()

        total_inventory_units = sum(p.stock_quantity for p in products)
        total_inventory_value = sum(p.price * p.stock_quantity for p in products)
        total_order_revenue = sum(o.total_amount for o in orders)
        orders_by_status: dict[str, int] = {}
        for o in orders:
            orders_by_status[o.status] = orders_by_status.get(o.status, 0) + 1

    return {
        "tool": "get_inventory_and_order_analytics",
        "total_product_skus": len(products),
        "total_inventory_units": total_inventory_units,
        "total_inventory_valuation_usd": round(total_inventory_value, 2),
        "total_orders": len(orders),
        "total_order_revenue_usd": round(total_order_revenue, 2),
        "orders_by_status": orders_by_status,
    }


@mcp_server.tool()
def lookup_policy_by_topic(topic: str) -> dict:
    """Search company policies in PostgreSQL by policy code, title, or keyword."""
    pattern = f"%{(topic or '').strip()}%"
    with SessionLocal() as db:
        matches = (
            db.query(SupportPolicy)
            .filter(
                (SupportPolicy.policy_code.ilike(pattern))
                | (SupportPolicy.title.ilike(pattern))
                | (SupportPolicy.content.ilike(pattern))
            )
            .all()
        )
        return {
            "tool": "lookup_policy_by_topic",
            "topic": topic,
            "matches": [
                {
                    "policy_code": m.policy_code,
                    "title": m.title,
                    "department": m.department,
                    "content": m.content,
                }
                for m in matches
            ],
        }


if __name__ == "__main__":
    mcp_server.run()
