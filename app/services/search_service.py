"""Semantic search using the pure-Python vector store."""
import logging
from typing import List, Dict
from app.db import vector_store as vs
from app.services.embedding_service import generate_embedding, generate_embeddings_batch, chunk_text

logger = logging.getLogger(__name__)


def index_document(doc_id: int, text: str, title: str, content_type: str):
    """Index a document for semantic search."""
    try:
        chunks = chunk_text(text)
        ids = [f"doc_{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"document_id": doc_id, "title": title, "content_type": content_type, "chunk_index": i}
            for i in range(len(chunks))
        ]
        embeddings = generate_embeddings_batch(chunks)
        vs.add_documents(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
        logger.info(f"Indexed document {doc_id} ({len(chunks)} chunks)")
    except Exception as e:
        logger.error(f"Indexing failed for document {doc_id}: {e}")


def semantic_search(query: str, limit: int = 10) -> List[Dict]:
    """Search documents semantically using the query string."""
    try:
        query_embedding = generate_embedding(query)
        results = vs.query(query_embedding=query_embedding, n_results=min(limit * 2, 50))

        if not results["documents"][0]:
            return []

        seen_docs = {}
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            doc_id = metadata["document_id"]
            score = max(0.0, 1.0 - distance)
            if doc_id not in seen_docs or score > seen_docs[doc_id]["score"]:
                seen_docs[doc_id] = {
                    "document_id": doc_id,
                    "title": metadata.get("title", "Untitled"),
                    "content_type": metadata.get("content_type", "note"),
                    "snippet": doc[:300] + "..." if len(doc) > 300 else doc,
                    "score": round(score, 4),
                }

        return sorted(seen_docs.values(), key=lambda x: x["score"], reverse=True)[:limit]
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        return []


def find_similar(doc_id: int, limit: int = 5) -> List[Dict]:
    """Find documents similar to a given document."""
    try:
        ref = vs.get_by_ids([f"doc_{doc_id}_chunk_0"])
        if not ref["documents"]:
            return []
        results = semantic_search(ref["documents"][0], limit=limit + 1)
        return [r for r in results if r["document_id"] != doc_id][:limit]
    except Exception as e:
        logger.error(f"Similar search failed: {e}")
        return []


def delete_document_index(doc_id: int):
    """Remove all chunks of a document from the vector store."""
    try:
        found = vs.get_by_metadata(where={"document_id": doc_id})
        if found["ids"]:
            vs.delete_by_ids(found["ids"])
    except Exception as e:
        logger.error(f"Failed to delete index for document {doc_id}: {e}")
