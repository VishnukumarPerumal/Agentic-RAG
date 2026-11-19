"""Database connection, session management, pgvector extension bootstrap, and initial data seeding."""

from collections.abc import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from app.utilities.config import settings


class Base(DeclarativeBase):
    pass


engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency providing a scoped SQLAlchemy database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Initialize the PostgreSQL database:
    1. Ensure pgvector extension is enabled.
    2. Create all vector and relational tables.
    3. Create HNSW cosine index on document_chunks.embedding.
    4. Seed sample PostgreSQL relational records if tables are empty.
    """
    from app.models import document_models, business_models  # noqa: F401

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

    Base.metadata.create_all(bind=engine)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw
                ON document_chunks
                USING hnsw (embedding vector_cosine_ops);
                """
            )
        )

    _seed_initial_postgres_data()


def _seed_initial_postgres_data() -> None:
    """Seed realistic sample records into PostgreSQL relational tables if empty."""
    from app.models.business_models import ProductRecord, CustomerOrder, SupportPolicy

    with SessionLocal() as db:
        if db.query(ProductRecord).count() == 0:
            products = [
                ProductRecord(
                    sku="AI-SRV-100",
                    name="Nexus GPU Inference Server X1",
                    category="Hardware",
                    price=14500.00,
                    stock_quantity=24,
                    description="Enterprise 4x GPU rackmount server optimized for LLM & RAG inference workloads.",
                ),
                ProductRecord(
                    sku="VEC-CLD-200",
                    name="VectorCloud Enterprise Plan",
                    category="SaaS Subscription",
                    price=499.00,
                    stock_quantity=999,
                    description="Managed PostgreSQL + pgvector cluster with automatic HNSW indexing and backups.",
                ),
                ProductRecord(
                    sku="MCP-GTW-300",
                    name="Agentic MCP Gateway Pro",
                    category="Software License",
                    price=1250.00,
                    stock_quantity=150,
                    description="Secure Model Context Protocol (MCP) tool router with RBAC and audit logging.",
                ),
                ProductRecord(
                    sku="SEC-GRD-400",
                    name="Guardrail Shield Enterprise",
                    category="Security",
                    price=850.00,
                    stock_quantity=300,
                    description="Real-time prompt injection, PII redaction, and toxicity filter for AI agents.",
                ),
            ]
            db.add_all(products)

        if db.query(CustomerOrder).count() == 0:
            orders = [
                CustomerOrder(
                    order_number="ORD-2026-1001",
                    customer_name="Acme Robotics Corp",
                    customer_email="procurement@acmerobotics.example.com",
                    product_sku="AI-SRV-100",
                    quantity=2,
                    total_amount=29000.00,
                    status="DELIVERED",
                ),
                CustomerOrder(
                    order_number="ORD-2026-1002",
                    customer_name="FinGuard Analytics",
                    customer_email="ops@finguard.example.com",
                    product_sku="SEC-GRD-400",
                    quantity=5,
                    total_amount=4250.00,
                    status="PROCESSING",
                ),
                CustomerOrder(
                    order_number="ORD-2026-1003",
                    customer_name="NovaHealth Systems",
                    customer_email="it@novahealth.example.com",
                    product_sku="VEC-CLD-200",
                    quantity=1,
                    total_amount=499.00,
                    status="ACTIVE",
                ),
            ]
            db.add_all(orders)

        if db.query(SupportPolicy).count() == 0:
            policies = [
                SupportPolicy(
                    policy_code="POL-REFUND-01",
                    title="Enterprise Refund & Return Policy",
                    department="Billing & Finance",
                    content=(
                        "Hardware orders (such as Nexus GPU Inference Server X1) are eligible for a full refund "
                        "within 30 days of delivery if returned in original packaging. SaaS subscriptions and "
                        "software licenses can be canceled within 14 days for a full pro-rated refund."
                    ),
                ),
                SupportPolicy(
                    policy_code="POL-SLA-02",
                    title="Production Uptime & Incident Response SLA",
                    department="Site Reliability Engineering",
                    content=(
                        "VectorCloud Enterprise and Agentic MCP Gateway guarantee 99.95% monthly uptime. "
                        "Severity-1 critical production incidents receive an initial engineer response within 15 minutes "
                        "and continuous updates every 30 minutes until resolution."
                    ),
                ),
                SupportPolicy(
                    policy_code="POL-SEC-03",
                    title="Data Privacy & Guardrail Compliance Policy",
                    department="Information Security",
                    content=(
                        "All customer queries processed by the RAG pipeline pass through the Guardrail Agent first. "
                        "Prompt injection attempts, destructive SQL commands, and raw SSN/credit-card leakage requests "
                        "are blocked and logged prior to data retrieval."
                    ),
                ),
            ]
            db.add_all(policies)

        db.commit()
