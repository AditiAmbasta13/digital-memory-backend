"""Content parser for PDFs, URLs, and text notes."""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def parse_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        logger.error(f"PDF parsing failed: {e}")
        return ""


def parse_url(url: str) -> dict:
    """Extract article content from a URL."""
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            return {
                "text": text or "",
                "title": _extract_title_from_url(url),
            }
    except Exception as e:
        logger.error(f"URL parsing failed: {e}")
    return {"text": "", "title": url}


def parse_note(text: str) -> str:
    """Process a text note (pass through with basic cleaning)."""
    return text.strip()


def _extract_title_from_url(url: str) -> str:
    """Extract a readable title from URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.strip("/").split("/")[-1] if parsed.path else parsed.netloc
        return path.replace("-", " ").replace("_", " ").title() or parsed.netloc
    except Exception:
        return url
