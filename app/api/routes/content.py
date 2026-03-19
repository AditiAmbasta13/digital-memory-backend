"""Content management API routes."""
import os
import shutil
import logging
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from typing import Optional, List

from app.db.session import get_db
from app.models.database import Document, Summary, DocumentConcept
from app.api.schemas.content import ContentResponse, ContentListItem
from app.services.content_parser import parse_pdf, parse_url, parse_note
from app.services.nlp_processor import extract_concepts, extract_relationships
from app.services.summarizer import generate_summary
from app.services.search_service import index_document, delete_document_index
from app.services.graph_manager import add_document_to_graph
from app.services import groq_service
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/content", tags=["content"])


@router.post("/upload")
async def upload_content(
    title: str = Form(...),
    content_type: str = Form(...),  # pdf, url, note
    source_url: Optional[str] = Form(None),
    raw_text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
):
    """Upload and process new content with Groq AI (falls back to rule-based NLP)."""
    text = ""

    # ── Parse raw content ────────────────────────────────────────────────────
    if content_type == "pdf" and file:
        temp_dir = os.path.join(settings.UPLOAD_DIR, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        text = parse_pdf(file_path)
        
        try:
            from app.services.mongo_service import upload_file_to_mongo
            source_url = await run_in_threadpool(upload_file_to_mongo, file_path, file.filename, file.content_type)
        except Exception as e:
            logger.error(f"MongoDB upload failed: {e}", exc_info=True)
            # Make sure we clean up the file
            try:
                os.remove(file_path)
            except:
                pass
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to upload file to MongoDB Storage: {str(e)}"
            )
        
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to remove temp file {file_path}: {e}")
            
    elif content_type == "url" and source_url:
        result = parse_url(source_url)
        text = result["text"]
        if not title or title == "Untitled":
            title = result["title"]
    elif content_type == "note" and raw_text:
        text = parse_note(raw_text)
    else:
        raise HTTPException(status_code=400, detail="Invalid content. Provide a file, URL, or note text.")

    if not text:
        raise HTTPException(status_code=400, detail="Could not extract any text from the provided content.")

    # ── Save raw document ────────────────────────────────────────────────────
    doc = Document(
        title=title,
        content_type=content_type,
        source_url=source_url,
        raw_text=text,
        processed=False,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # ── AI processing pipeline ───────────────────────────────────────────────
    try:
        use_groq = groq_service.groq_available()
        logger.info(f"Processing doc {doc.id} ('{title}') — Groq AI: {use_groq}")

        # 1. Concept extraction — Groq first, regex fallback
        concepts = None
        if use_groq:
            concepts = groq_service.groq_extract_concepts(text, title=title)
            if concepts:
                logger.info(f"Groq extracted {len(concepts)} concepts.")
            else:
                logger.warning("Groq concept extraction failed — falling back to regex.")
        if not concepts:
            concepts = extract_concepts(text, title=title)
            use_groq = False

        # 2. Relationship extraction — Groq first, regex fallback
        relationships = None
        if groq_service.groq_available() and concepts:
            relationships = groq_service.groq_extract_relationships(concepts, text, title=title)
            if relationships:
                logger.info(f"Groq inferred {len(relationships)} relationships.")
            else:
                logger.warning("Groq relationship extraction failed — falling back to regex.")
        if not relationships:
            relationships = extract_relationships(concepts, text)

        # 3. Summarisation — Groq first, extractive fallback
        summary_text = None
        if groq_service.groq_available():
            summary_text = groq_service.groq_summarise(text, title=title)
            if summary_text:
                logger.info("Groq summary generated.")
        if not summary_text:
            summary_text = generate_summary(text)

        # ── Persist results ──────────────────────────────────────────────────
        db.add(Summary(
            document_id=doc.id,
            summary_text=summary_text,
            method="groq-llm" if use_groq else "extractive",
        ))

        for concept in concepts:
            db.add(DocumentConcept(
                document_id=doc.id,
                concept_name=concept["name"],
                category=concept.get("category", "CONCEPT"),
                relevance_score=concept.get("relevance_score", 1.0),
            ))

        doc.processed = True
        db.commit()

        # Index for semantic search
        index_document(doc.id, text, title, content_type)

        # Add to knowledge graph
        add_document_to_graph(doc.id, title, concepts, relationships)

    except Exception as e:
        logger.error(f"Processing failed for doc {doc.id}: {e}", exc_info=True)
        db.commit()
        raise HTTPException(status_code=500, detail=f"Processing partially failed: {str(e)}")

    return {
        "id": doc.id,
        "title": doc.title,
        "content_type": doc.content_type,
        "processed": doc.processed,
        "summary": summary_text,
        "concepts": [c["name"] for c in concepts],
        "ai_powered": groq_service.groq_available(),
        "message": f"Processed {'with Groq AI 🤖' if groq_service.groq_available() else 'with rule-based NLP'}",
    }


@router.get("/files/{file_id}")
async def serve_file(file_id: str):
    """Serve a file directly from MongoDB GridFS."""
    from app.services.mongo_service import get_file_from_mongo
    from fastapi.responses import StreamingResponse
    import mimetypes
    
    try:
        grid_out = await run_in_threadpool(get_file_from_mongo, file_id)
        if not grid_out:
            raise HTTPException(status_code=404, detail="File not found in storage")
            
        def iterfile():
            yield grid_out.read()
            
        content_type = grid_out.content_type
        if not content_type:
            content_type, _ = mimetypes.guess_type(grid_out.filename)
            content_type = content_type or "application/octet-stream"
            
        return StreamingResponse(
            iterfile(), 
            media_type=content_type,
            headers={"Content-Disposition": f'inline; filename="{grid_out.filename}"'}
        )
    except Exception as e:
        logger.error(f"Error serving file {file_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving file from storage")


@router.get("/", response_model=List[ContentListItem])
async def list_content(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """List all saved content."""
    docs = db.query(Document).order_by(Document.created_at.desc()).offset(skip).limit(limit).all()

    results = []
    for doc in docs:
        concept_count = db.query(DocumentConcept).filter(DocumentConcept.document_id == doc.id).count()
        summary = db.query(Summary).filter(Summary.document_id == doc.id).first()
        results.append(ContentListItem(
            id=doc.id,
            title=doc.title,
            content_type=doc.content_type,
            processed=doc.processed,
            created_at=doc.created_at,
            concept_count=concept_count,
            summary_preview=summary.summary_text[:200] + "..." if summary and len(summary.summary_text) > 200 else (summary.summary_text if summary else None),
        ))

    return results


@router.get("/{doc_id}")
async def get_content(doc_id: int, db: Session = Depends(get_db)):
    """Get a specific document with its summary and concepts."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    summary = db.query(Summary).filter(Summary.document_id == doc_id).first()
    concepts = db.query(DocumentConcept).filter(DocumentConcept.document_id == doc_id).all()

    return {
        "id": doc.id,
        "title": doc.title,
        "content_type": doc.content_type,
        "source_url": doc.source_url,
        "raw_text": doc.raw_text,
        "processed": doc.processed,
        "created_at": doc.created_at.isoformat(),
        "summary": summary.summary_text if summary else None,
        "concepts": [{"name": c.concept_name, "category": c.category, "relevance": c.relevance_score} for c in concepts],
    }


@router.delete("/{doc_id}")
async def delete_content(doc_id: int, db: Session = Depends(get_db)):
    """Delete a document and all its associated data."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.content_type == "pdf" and doc.source_url:
        try:
            from app.services.mongo_service import delete_file_from_mongo
            await run_in_threadpool(delete_file_from_mongo, doc.source_url)
        except Exception as e:
            logger.warning(f"Failed to delete file from MongoDB: {e}")

    delete_document_index(doc_id)

    try:
        from app.services.graph_manager import remove_document_from_graph
        remove_document_from_graph(doc_id)
    except Exception:
        pass  # Non-fatal

    db.delete(doc)
    db.commit()

    return {"message": f"Document {doc_id} deleted successfully"}
