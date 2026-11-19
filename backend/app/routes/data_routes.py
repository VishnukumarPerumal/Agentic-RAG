"""FastAPI routes for inspecting/managing PostgreSQL relational data and MCP tools."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.schemas.data_schemas import (
    CustomerOrderCreate,
    CustomerOrderOut,
    MCPToolCallRequest,
    ProductCreate,
    ProductOut,
    SupportPolicyCreate,
    SupportPolicyOut,
)
from app.repository.sql_data_repository import SQLDataRepository
from app.services.mcp_tools_service import MCPToolsService

router = APIRouter(prefix="/api/v1/data", tags=["PostgreSQL Data & MCP Tools"])


@router.get("/products", response_model=list[ProductOut])
def list_products(db: Session = Depends(get_db)) -> list[ProductOut]:
    repo = SQLDataRepository(db)
    return [ProductOut.model_validate(p) for p in repo.list_products()]


@router.post("/products", response_model=ProductOut, status_code=status.HTTP_201_CREATED)
def create_product(
    payload: ProductCreate,
    db: Session = Depends(get_db),
) -> ProductOut:
    repo = SQLDataRepository(db)
    created = repo.create_product(payload.model_dump())
    return ProductOut.model_validate(created)


@router.get("/orders", response_model=list[CustomerOrderOut])
def list_orders(db: Session = Depends(get_db)) -> list[CustomerOrderOut]:
    repo = SQLDataRepository(db)
    return [CustomerOrderOut.model_validate(o) for o in repo.list_orders()]


@router.post("/orders", response_model=CustomerOrderOut, status_code=status.HTTP_201_CREATED)
def create_order(
    payload: CustomerOrderCreate,
    db: Session = Depends(get_db),
) -> CustomerOrderOut:
    repo = SQLDataRepository(db)
    created = repo.create_order(payload.model_dump())
    return CustomerOrderOut.model_validate(created)


@router.get("/policies", response_model=list[SupportPolicyOut])
def list_policies(db: Session = Depends(get_db)) -> list[SupportPolicyOut]:
    repo = SQLDataRepository(db)
    return [SupportPolicyOut.model_validate(p) for p in repo.list_policies()]


@router.post("/policies", response_model=SupportPolicyOut, status_code=status.HTTP_201_CREATED)
def create_policy(
    payload: SupportPolicyCreate,
    db: Session = Depends(get_db),
) -> SupportPolicyOut:
    repo = SQLDataRepository(db)
    created = repo.create_policy(payload.model_dump())
    return SupportPolicyOut.model_validate(created)


@router.get("/mcp/tools")
def list_mcp_tools() -> list[dict]:
    """List all registered Model Context Protocol (MCP) tools available to the Data Retrieval Agent."""
    mcp_service = MCPToolsService()
    return mcp_service.list_tools()


@router.post("/mcp/call")
def call_mcp_tool(payload: MCPToolCallRequest) -> dict:
    """Invoke a registered MCP tool directly."""
    mcp_service = MCPToolsService()
    try:
        return mcp_service.call_tool(payload.tool_name, payload.arguments)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
