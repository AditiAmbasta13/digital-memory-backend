"""Knowledge graph API routes."""
from fastapi import APIRouter, Query, Depends
from sqlalchemy.orm import Session
from typing import Optional
import json
import re

from app.services.graph_manager import get_graph_data, get_node_details
from app.services import groq_service
from app.db.session import get_db
from app.models.database import Document

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/")
async def get_graph(
    limit: int = Query(100, ge=1, le=500), 
    doc_ids: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get the knowledge graph data for visualization. Defaults to last 3 documents if none specified."""
    parsed_ids = None
    if doc_ids:
        parsed_ids = [int(i.strip()) for i in doc_ids.split(",") if i.strip().isdigit()]
    else:
        # Default to top 3 most recent documents to prevent showing "everything"
        recent_docs = db.query(Document).order_by(Document.created_at.desc()).limit(3).all()
        parsed_ids = [d.id for d in recent_docs]
        
    data = get_graph_data(limit=limit, doc_ids=parsed_ids)
    return data


@router.get("/node/{node_id}")
async def get_node(node_id: str):
    """Get details for a specific node in the graph."""
    details = get_node_details(node_id)
    if details is None:
        return {"error": "Node not found"}
    return details


@router.get("/stats")
async def graph_stats():
    """Get graph statistics."""
    data = get_graph_data(limit=1000)
    doc_nodes = [n for n in data["nodes"] if n.get("type") == "document"]
    concept_nodes = [n for n in data["nodes"] if n.get("type") == "concept"]
    return {
        "total_nodes": len(data["nodes"]),
        "total_edges": len(data["edges"]),
        "documents": len(doc_nodes),
        "concepts": len(concept_nodes),
    }


@router.get("/explain")
async def explain_graph(doc_ids: Optional[str] = None, db: Session = Depends(get_db)):
    """Use Groq AI to explain the knowledge graph in simple bullet points."""
    parsed_ids = [int(i.strip()) for i in doc_ids.split(",") if i.strip().isdigit()] if doc_ids else None
    data = get_graph_data(limit=200, doc_ids=parsed_ids)
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    if not nodes:
        return {"sections": [], "available": False, "message": "No graph data yet."}

    doc_titles = []
    if parsed_ids:
        docs = db.query(Document).filter(Document.id.in_(parsed_ids)).all()
        doc_titles = [d.title for d in docs]
    else:
        docs = db.query(Document).order_by(Document.id.desc()).limit(15).all()
        doc_titles = [d.title for d in docs]

    raw = groq_service.groq_explain_graph(nodes, edges, doc_titles)
    if raw is None:
        # Groq not available — build a simple fallback explanation
        main_nodes = [n for n in nodes if n.get("is_main")]
        main_label = main_nodes[0]["label"] if main_nodes else "the subject"
        categories: dict = {}
        for n in nodes:
            cat = n.get("category", "ENTITY")
            categories.setdefault(cat, []).append(n["label"])

        sections = []
        COLOR_MAP = {
            "PERSON": "#FFD700", "SKILL": "#7b2ff7", "TECHNOLOGY": "#a78bfa",
            "COMPANY": "#38bdf8", "PROJECT": "#c084fc", "EDUCATION": "#4ade80",
            "LOCATION": "#f472b6", "TOPIC": "#34d399", "EVENT": "#f87171",
            "ENTITY": "#64748b", "ROLE": "#fb923c", "CONCEPT": "#fbbf24",
        }
        ICON_MAP = {
            "PERSON": "👤", "SKILL": "⚡", "TECHNOLOGY": "💻",
            "COMPANY": "🏢", "PROJECT": "🚀", "EDUCATION": "🎓",
            "LOCATION": "📍", "TOPIC": "📚", "EVENT": "🏆",
            "ENTITY": "🔷", "ROLE": "💼", "CONCEPT": "💡",
        }
        for cat, names in categories.items():
            if cat == "PERSON":
                continue
            sections.append({
                "category": cat.title(),
                "color": COLOR_MAP.get(cat, "#64748b"),
                "icon": ICON_MAP.get(cat, "🔹"),
                "points": [f"{main_label} is connected to {n}." for n in names[:4]],
            })
        return {"sections": sections[:5], "available": False, "message": "Basic summary (Groq not configured)."}

    # Parse Groq JSON
    sections = None
    try:
        sections = json.loads(raw)
    except Exception:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            sections = json.loads(cleaned)
        except Exception:
            m = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if m:
                try:
                    sections = json.loads(m.group(0))
                except Exception:
                    pass

    if not sections:
        return {"sections": [], "available": True, "message": "Could not parse AI response."}

    return {"sections": sections, "available": True, "message": ""}

