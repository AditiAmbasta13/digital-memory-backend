"""Digital Memory System — FastAPI Backend Entry Point."""
# Triggering uvicorn hot-reload to load updated MongoDB URI
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.session import init_db
from app.api.routes import content, graph, search, health

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-powered knowledge management system",
    version="1.0.0",
)

# CORS — allow Next.js frontend (origins configured via ALLOWED_ORIGINS env var)
_allowed_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(content.router)
app.include_router(graph.router)
app.include_router(search.router)
app.include_router(health.router)


@app.on_event("startup")
async def startup():
    """Initialize database and services on startup."""
    import os
    logger.info(f"Starting {settings.APP_NAME}...")
    init_db()
    logger.info("Database initialized")

    # Rebuild graph from DB if graph store doesn't exist yet (e.g. after a fresh restart)
    if not os.path.exists("./graph_store.json"):
        logger.info("graph_store.json not found — rebuilding graph from existing documents...")
        try:
            from app.db.session import SessionLocal
            from app.models.database import Document, DocumentConcept
            from app.services.graph_manager import add_document_to_graph
            db = SessionLocal()
            docs = db.query(Document).all()
            for doc in docs:
                dc_rows = db.query(DocumentConcept).filter(DocumentConcept.document_id == doc.id).all()
                concepts = [{"name": r.concept_name, "relevance_score": r.relevance_score} for r in dc_rows]
                add_document_to_graph(doc.id, doc.title, concepts, [])
            db.close()
            logger.info(f"Graph rebuilt: {len(docs)} documents re-added")
        except Exception as e:
            logger.warning(f"Graph rebuild failed (non-fatal): {e}")



@app.on_event("shutdown")
async def shutdown():
    """Clean up resources on shutdown."""
    from app.db.neo4j_driver import Neo4jDriver
    Neo4jDriver.close()
    
    try:
        from app.services.mongo_service import close_mongo_connection
        close_mongo_connection()
    except Exception as e:
        logger.warning(f"Error closing MongoDB: {e}")
        
    logger.info("Shutdown complete")


@app.get("/")
async def root():
    return {
        "service": settings.APP_NAME,
        "version": "1.0.0",
        "docs": "/docs",
    }
