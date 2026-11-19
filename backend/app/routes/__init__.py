"""Routes package exporting chat, ingestion, and data API routers."""

from app.routes.chat_routes import router as chat_router
from app.routes.data_routes import router as data_router
from app.routes.ingestion_routes import router as ingestion_router

__all__ = ["chat_router", "data_router", "ingestion_router"]
