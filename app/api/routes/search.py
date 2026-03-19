"""Semantic search API routes."""
from fastapi import APIRouter, Query, Depends
from sqlalchemy.orm import Session
from typing import List

from app.db.session import get_db
from app.models.database import DocumentConcept
from app.services.search_service import semantic_search, find_similar

router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("/")
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=50),
):
    """Semantic search across all saved content."""
    results = semantic_search(q, limit=limit)
    return {
        "query": q,
        "results": results,
        "total": len(results),
    }


@router.get("/similar/{doc_id}")
async def similar_documents(doc_id: int, limit: int = Query(5, ge=1, le=20)):
    """Find documents similar to a given document."""
    results = find_similar(doc_id, limit=limit)
    # Filter out the source document
    results = [r for r in results if r["document_id"] != doc_id]
    return {
        "document_id": doc_id,
        "similar": results[:limit],
    }


@router.get("/concepts")
async def search_by_concept(
    concept: str = Query(..., min_length=1),
    db: Session = Depends(get_db),
):
    """Find documents containing a specific concept."""
    doc_concepts = (
        db.query(DocumentConcept)
        .filter(DocumentConcept.concept_name.ilike(f"%{concept}%"))
        .all()
    )

    results = {}
    for dc in doc_concepts:
        if dc.document_id not in results:
            doc = dc.document
            results[dc.document_id] = {
                "document_id": dc.document_id,
                "title": doc.title if doc else "Untitled",
                "content_type": doc.content_type if doc else "unknown",
                "matched_concept": dc.concept_name,
                "relevance": dc.relevance_score,
            }

    return {
        "concept": concept,
        "results": list(results.values()),
        "total": len(results),
    }
