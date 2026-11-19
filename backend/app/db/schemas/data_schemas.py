"""Pydantic schemas for PostgreSQL relational data and MCP tools endpoints."""

from datetime import datetime
from pydantic import BaseModel, Field


class ProductCreate(BaseModel):
    sku: str = Field(..., min_length=2, max_length=64)
    name: str = Field(..., min_length=2, max_length=255)
    category: str = Field(..., min_length=2, max_length=100)
    price: float = Field(..., ge=0)
    stock_quantity: int = Field(default=0, ge=0)
    description: str = Field(..., min_length=2)


class ProductOut(ProductCreate):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class CustomerOrderCreate(BaseModel):
    order_number: str = Field(..., min_length=2, max_length=64)
    customer_name: str = Field(..., min_length=2, max_length=255)
    customer_email: str = Field(..., min_length=3, max_length=255)
    product_sku: str = Field(..., min_length=2, max_length=64)
    quantity: int = Field(default=1, ge=1)
    total_amount: float = Field(..., ge=0)
    status: str = Field(default="PROCESSING")


class CustomerOrderOut(CustomerOrderCreate):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class SupportPolicyCreate(BaseModel):
    policy_code: str = Field(..., min_length=2, max_length=64)
    title: str = Field(..., min_length=2, max_length=255)
    department: str = Field(..., min_length=2, max_length=100)
    content: str = Field(..., min_length=5)


class SupportPolicyOut(SupportPolicyCreate):
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class MCPToolCallRequest(BaseModel):
    tool_name: str
    arguments: dict = Field(default_factory=dict)
