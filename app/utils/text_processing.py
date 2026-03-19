"""Utility functions for text processing."""
import re


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\-\'\"()\[\]{}]', '', text)
    return text.strip()


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to a maximum length, breaking at word boundary."""
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:
        truncated = truncated[:last_space]
    return truncated + "..."
