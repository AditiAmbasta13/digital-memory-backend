"""Text summarization using extractive methods."""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def generate_summary(text: str, max_sentences: int = 5) -> str:
    """
    Generate an extractive summary by scoring sentences based on word frequency.
    This is a fast, dependency-light approach.
    """
    if not text or len(text.strip()) < 50:
        return text.strip()

    sentences = _split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    # Score sentences by word importance
    word_freq = _compute_word_frequencies(text)
    sentence_scores = []

    for i, sentence in enumerate(sentences):
        words = sentence.lower().split()
        if len(words) < 4:
            continue
        score = sum(word_freq.get(w, 0) for w in words) / len(words)
        # Boost early sentences (position bias)
        position_boost = 1.0 + (0.5 if i < 3 else 0.0)
        sentence_scores.append((i, sentence, score * position_boost))

    # Select top sentences, keep original order
    sentence_scores.sort(key=lambda x: x[2], reverse=True)
    top = sorted(sentence_scores[:max_sentences], key=lambda x: x[0])

    return " ".join(s[1] for s in top)


def _split_sentences(text: str) -> list:
    """Split text into sentences."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _compute_word_frequencies(text: str) -> dict:
    """Compute normalized word frequencies, ignoring stop words."""
    import re
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "just", "because", "but", "and", "or", "if", "while", "that", "this",
        "it", "its", "he", "she", "they", "them", "his", "her", "their", "we",
        "you", "i", "me", "my", "your", "our", "which", "what", "who", "whom",
    }

    words = re.findall(r'\b[a-z]+\b', text.lower())
    freq = {}
    for w in words:
        if w not in stop_words and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1

    # Normalize
    if freq:
        max_freq = max(freq.values())
        freq = {w: c / max_freq for w, c in freq.items()}

    return freq
