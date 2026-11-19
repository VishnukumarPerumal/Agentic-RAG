"""Repository for querying and managing structured PostgreSQL relational data."""

import re
from sqlalchemy import or_
from sqlalchemy.orm import Session
from app.models.business_models import CustomerOrder, ProductRecord, SupportPolicy


class SQLDataRepository:
    """Data access layer for structured PostgreSQL tables (products, customer_orders, support_policies)."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def list_products(self) -> list[ProductRecord]:
        return self.db.query(ProductRecord).order_by(ProductRecord.id.asc()).all()

    def create_product(self, data: dict) -> ProductRecord:
        product = ProductRecord(**data)
        self.db.add(product)
        self.db.commit()
        self.db.refresh(product)
        return product

    def list_orders(self) -> list[CustomerOrder]:
        return self.db.query(CustomerOrder).order_by(CustomerOrder.id.asc()).all()

    def create_order(self, data: dict) -> CustomerOrder:
        order = CustomerOrder(**data)
        self.db.add(order)
        self.db.commit()
        self.db.refresh(order)
        return order

    def list_policies(self) -> list[SupportPolicy]:
        return self.db.query(SupportPolicy).order_by(SupportPolicy.id.asc()).all()

    def create_policy(self, data: dict) -> SupportPolicy:
        policy = SupportPolicy(**data)
        self.db.add(policy)
        self.db.commit()
        self.db.refresh(policy)
        return policy

    def search_relational_data_for_query(self, query: str) -> list[dict]:
        """
        Intelligently query structured PostgreSQL tables (products, customer_orders,
        support_policies) based on user query keywords and intent.
        """
        q_lower = (query or "").lower()
        keywords = [
            tok
            for tok in re.findall(r"[a-z0-9_-]+", q_lower)
            if len(tok) >= 3
            and tok
            not in {
                "what", "which", "where", "when", "who", "how", "many", "much",
                "the", "and", "for", "from", "with", "that", "this", "are", "was",
                "were", "can", "you", "tell", "about", "show", "list", "all",
                "please", "give", "have", "has", "does",
            }
        ]

        records: list[dict] = []

        # 1. Check Products table
        wants_all_products = any(
            w in q_lower
            for w in ["product", "products", "catalog", "inventory", "price", "pricing", "stock", "sku", "hardware", "server"]
        )
        if wants_all_products:
            products = self.list_products()
        elif keywords:
            filters = []
            for kw in keywords[:6]:
                pattern = f"%{kw}%"
                filters.extend(
                    [
                        ProductRecord.sku.ilike(pattern),
                        ProductRecord.name.ilike(pattern),
                        ProductRecord.category.ilike(pattern),
                        ProductRecord.description.ilike(pattern),
                    ]
                )
            products = self.db.query(ProductRecord).filter(or_(*filters)).limit(5).all()
        else:
            products = []

        for p in products:
            records.append(
                {
                    "table": "products",
                    "record_id": p.sku,
                    "summary": (
                        f"Product [{p.sku}] {p.name} ({p.category}): "
                        f"Price=${p.price:,.2f}, Stock={p.stock_quantity}. {p.description}"
                    ),
                    "row": {
                        "sku": p.sku,
                        "name": p.name,
                        "category": p.category,
                        "price": p.price,
                        "stock_quantity": p.stock_quantity,
                        "description": p.description,
                    },
                }
            )

        # 2. Check Customer Orders table
        wants_all_orders = any(
            w in q_lower
            for w in ["order", "orders", "customer", "customers", "purchase", "delivered", "processing", "ord-"]
        )
        if wants_all_orders:
            orders = self.list_orders()
        elif keywords:
            filters = []
            for kw in keywords[:6]:
                pattern = f"%{kw}%"
                filters.extend(
                    [
                        CustomerOrder.order_number.ilike(pattern),
                        CustomerOrder.customer_name.ilike(pattern),
                        CustomerOrder.product_sku.ilike(pattern),
                        CustomerOrder.status.ilike(pattern),
                    ]
                )
            orders = self.db.query(CustomerOrder).filter(or_(*filters)).limit(5).all()
        else:
            orders = []

        for o in orders:
            records.append(
                {
                    "table": "customer_orders",
                    "record_id": o.order_number,
                    "summary": (
                        f"Order [{o.order_number}] Customer={o.customer_name} ({o.customer_email}), "
                        f"Product SKU={o.product_sku}, Qty={o.quantity}, Total=${o.total_amount:,.2f}, Status={o.status}"
                    ),
                    "row": {
                        "order_number": o.order_number,
                        "customer_name": o.customer_name,
                        "customer_email": o.customer_email,
                        "product_sku": o.product_sku,
                        "quantity": o.quantity,
                        "total_amount": o.total_amount,
                        "status": o.status,
                    },
                }
            )

        # 3. Check Support Policies table
        wants_all_policies = any(
            w in q_lower
            for w in ["policy", "policies", "refund", "return", "sla", "uptime", "compliance", "security", "guardrail"]
        )
        if wants_all_policies:
            policies = self.list_policies()
        elif keywords:
            filters = []
            for kw in keywords[:6]:
                pattern = f"%{kw}%"
                filters.extend(
                    [
                        SupportPolicy.policy_code.ilike(pattern),
                        SupportPolicy.title.ilike(pattern),
                        SupportPolicy.department.ilike(pattern),
                        SupportPolicy.content.ilike(pattern),
                    ]
                )
            policies = self.db.query(SupportPolicy).filter(or_(*filters)).limit(5).all()
        else:
            policies = []

        for pol in policies:
            records.append(
                {
                    "table": "support_policies",
                    "record_id": pol.policy_code,
                    "summary": (
                        f"Policy [{pol.policy_code}] {pol.title} (Dept: {pol.department}): {pol.content}"
                    ),
                    "row": {
                        "policy_code": pol.policy_code,
                        "title": pol.title,
                        "department": pol.department,
                        "content": pol.content,
                    },
                }
            )

        return records
