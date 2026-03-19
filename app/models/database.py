from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    content_type = Column(String(20), nullable=False)  # pdf, url, note
    source_url = Column(String(2000), nullable=True)
    raw_text = Column(Text, nullable=True)
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    summaries = relationship("Summary", back_populates="document", cascade="all, delete-orphan")
    concepts = relationship("DocumentConcept", back_populates="document", cascade="all, delete-orphan")


class Summary(Base):
    __tablename__ = "summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    summary_text = Column(Text, nullable=False)
    method = Column(String(50), default="extractive")  # extractive / abstractive
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="summaries")


class DocumentConcept(Base):
    __tablename__ = "document_concepts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    concept_name = Column(String(200), nullable=False)
    category = Column(String(100), nullable=True)
    relevance_score = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    document = relationship("Document", back_populates="concepts")
