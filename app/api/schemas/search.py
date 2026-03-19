from pydantic import BaseModel
from typing import List, Optional


class SearchQuery(BaseModel):
    query: str
    limit: int = 10


class SearchResult(BaseModel):
    document_id: int
    title: str
    content_type: str
    snippet: str
    score: float
    concepts: List[str] = []


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int
