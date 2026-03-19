"""Health check endpoint."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.neo4j_driver import Neo4jDriver
from app.models.database import Document

router = APIRouter(tags=["system"])


@router.get("/api/health")
async def health_check():
    """System health check."""
    return {
        "status": "healthy",
        "service": "Digital Memory System",
        "neo4j_connected": Neo4jDriver.is_connected(),
    }


@router.get("/api/stats")
async def system_stats(db: Session = Depends(get_db)):
    """System-wide statistics."""
    total_docs = db.query(Document).count()
    processed = db.query(Document).filter(Document.processed == True).count()

    from app.services.graph_manager import get_graph_data
    graph = get_graph_data(limit=1000)

    return {
        "total_documents": total_docs,
        "processed_documents": processed,
        "graph_nodes": len(graph["nodes"]),
        "graph_edges": len(graph["edges"]),
    }
