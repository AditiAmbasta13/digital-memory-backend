import chromadb
from app.config import settings
import logging

logger = logging.getLogger(__name__)

_client = None
_collection = None


def _get_embedding_function():
    """
    Return a no-op embedding function so ChromaDB doesn't try to
    load onnxruntime at import time. We supply our own embeddings
    when indexing, so ChromaDB only needs to store/retrieve them.
    """
    try:
        from chromadb.utils.embedding_functions import EmbeddingFunction

        class NoOpEmbeddingFunction(EmbeddingFunction):
            def __call__(self, texts):
                # Return zero vectors — real embeddings are added via .add(documents=..., embeddings=...)
                return [[0.0] * 384 for _ in texts]

        return NoOpEmbeddingFunction()
    except Exception:
        return None


def get_chroma_client():
    global _client
    if _client is None:
        try:
            _client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
            logger.info("ChromaDB initialized (persistent)")
        except Exception as e:
            logger.warning(f"ChromaDB PersistentClient failed: {e}. Using ephemeral.")
            try:
                _client = chromadb.EphemeralClient()
            except Exception as e2:
                logger.error(f"ChromaDB EphemeralClient also failed: {e2}")
                _client = None
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = get_chroma_client()
        if client is None:
            return None
        ef = _get_embedding_function()
        kwargs = {
            "name": settings.CHROMA_COLLECTION,
            "metadata": {"hnsw:space": "cosine"},
        }
        if ef is not None:
            kwargs["embedding_function"] = ef
        try:
            _collection = client.get_or_create_collection(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create ChromaDB collection: {e}")
            _collection = None
    return _collection
