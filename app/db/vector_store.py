"""
Pure-Python in-memory vector store — replaces ChromaDB.
No external dependencies. Uses cosine similarity over numpy arrays.
Documents + embeddings are kept in memory and optionally saved to a JSON file.
"""
import json
import os
import logging
import math
from typing import List, Dict, Optional, Any, Tuple

logger = logging.getLogger(__name__)

_STORE_PATH = "./vector_store.json"

# In-memory store: list of {id, document, embedding, metadata}
_store: List[Dict] = []
_loaded = False


def _load():
    global _store, _loaded
    if _loaded:
        return
    _loaded = True
    if os.path.exists(_STORE_PATH):
        try:
            with open(_STORE_PATH, "r", encoding="utf-8") as f:
                _store = json.load(f)
            logger.info(f"Vector store loaded: {len(_store)} entries")
        except Exception as e:
            logger.warning(f"Could not load vector store: {e}. Starting fresh.")
            _store = []


def _save():
    try:
        with open(_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(_store, f)
    except Exception as e:
        logger.warning(f"Could not save vector store: {e}")


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def add_documents(ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
    """Add documents with embeddings to the store."""
    _load()
    existing_ids = {entry["id"] for entry in _store}
    added = 0
    for id_, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
        if id_ not in existing_ids:
            _store.append({"id": id_, "document": doc, "embedding": emb, "metadata": meta})
            added = int(added) + 1
    if added > 0:
        _save()
        logger.info(f"Added {added} entries to vector store (total: {len(_store)})")


def query(query_embedding: List[float], n_results: int = 10, where: Optional[Dict] = None) -> Dict:
    """Find the top-n most similar documents."""
    _load()

    candidates = _store
    if where:
        candidates = [
            e for e in _store
            if all(e["metadata"].get(k) == v for k, v in where.items())
        ]

    if not candidates:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    scored: List[Tuple[float, Any]] = []
    for entry in candidates:
        sim = _cosine_similarity(query_embedding, entry["embedding"])
        scored.append((sim, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    top: List[Tuple[float, Any]] = list(scored[:n_results])

    return {
        "documents": [[e["document"] for _, e in top]],
        "metadatas": [[e["metadata"] for _, e in top]],
        "distances": [[1.0 - sim for sim, _ in top]],  # distance = 1 - similarity
    }


def get_by_ids(ids: List[str]) -> Dict:
    """Get specific entries by ID."""
    _load()
    found = [e for e in _store if e["id"] in ids]
    return {
        "documents": [e["document"] for e in found],
        "metadatas": [e["metadata"] for e in found],
    }


def get_by_metadata(where: Dict) -> Dict:
    """Get entries matching metadata filter."""
    _load()
    found = [
        e for e in _store
        if all(e["metadata"].get(k) == v for k, v in where.items())
    ]
    return {
        "ids": [e["id"] for e in found],
        "documents": [e["document"] for e in found],
        "metadatas": [e["metadata"] for e in found],
    }


def delete_by_ids(ids: List[str]):
    """Delete entries by ID."""
    global _store
    _load()
    before = len(_store)
    _store = [e for e in _store if e["id"] not in ids]
    if len(_store) < before:
        _save()
        logger.info(f"Deleted {before - len(_store)} entries from vector store")
