from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class ContentUpload(BaseModel):
    title: str
    content_type: str  # "pdf", "url", "note"
    source_url: Optional[str] = None
    raw_text: Optional[str] = None


class ContentResponse(BaseModel):
    id: int
    title: str
    content_type: str
    source_url: Optional[str]
    raw_text: Optional[str]
    processed: bool
    created_at: datetime
    summary: Optional[str] = None
    concepts: List[str] = []

    class Config:
        from_attributes = True


class ContentListItem(BaseModel):
    id: int
    title: str
    content_type: str
    processed: bool
    created_at: datetime
    concept_count: int = 0
    summary_preview: Optional[str] = None
