from pydantic import BaseModel
from typing import List, Optional


class GraphNode(BaseModel):
    id: str
    label: str
    type: str  # "document", "concept", "tag"
    size: float = 1.0
    color: Optional[str] = None
    metadata: dict = {}


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str
    weight: float = 1.0


class GraphData(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class NodeDetail(BaseModel):
    id: str
    label: str
    type: str
    connections: int
    related_documents: List[dict] = []
    related_concepts: List[str] = []
