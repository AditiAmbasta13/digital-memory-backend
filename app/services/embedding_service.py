"""Embedding generation using sentence-transformers."""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazy-load the embedding model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            from app.config import settings
            _model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info(f"Embedding model '{settings.EMBEDDING_MODEL}' loaded")
        except Exception as e:
            logger.warning(f"Embedding model not available: {e}")
            _model = "fallback"
    return _model


def generate_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding vector for a text string."""
    model = _get_model()
    if model == "fallback":
        return _fallback_embedding(text)

    embedding = model.encode(text, show_progress_bar=False)
    return embedding.tolist()


def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a batch of texts."""
    model = _get_model()
    if model == "fallback":
        return [_fallback_embedding(t) for t in texts]

    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    return [e.tolist() for e in embeddings]


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
    return chunks if chunks else [text]


def _fallback_embedding(text: str) -> List[float]:
    """Simple character-frequency based embedding fallback."""
    import hashlib
    h = hashlib.sha256(text.encode()).hexdigest()
    return [int(h[i:i+2], 16) / 255.0 for i in range(0, 64, 2)]
