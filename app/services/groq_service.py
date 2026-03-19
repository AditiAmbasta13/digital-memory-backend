"""
Groq AI service — powers smart concept extraction, relationship inference,
and summarisation for Digital Memory.

Falls back gracefully to the regex-based nlp_processor if GROQ_API_KEY is
not set or if the API call fails.
"""
import json
import logging
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

_client: Optional[object] = None


def _get_client():
    """Lazy-init Groq client. Returns None if key is missing."""
    global _client
    if _client is not None:
        return _client
    try:
        from app.config import settings
        if not settings.GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not set — Groq AI disabled.")
            return None
        from groq import Groq
        _client = Groq(api_key=settings.GROQ_API_KEY)
        logger.info("Groq client initialised.")
        return _client
    except Exception as e:
        logger.error(f"Groq init failed: {e}")
        return None


def _chat(prompt: str, system: str, model: str = "llama-3.3-70b-versatile",
          temperature: float = 0.1, max_tokens: int = 1500) -> Optional[str]:
    """Make a single Groq chat completion and return the text content."""
    client = _get_client()
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def groq_extract_concepts(text: str, title: str = "") -> Optional[List[Dict]]:
    """
    Ask Groq to extract meaningful entities / concepts from the text.

    Returns a list of dicts:
        {name, category, relevance_score (0-1), is_main (bool)}

    Returns None if Groq is unavailable so the caller can fall back.

    Categories Groq should use:
        PERSON | COMPANY | ROLE | EDUCATION | PROJECT | SKILL |
        TECHNOLOGY | LOCATION | TOPIC | EVENT | CONCEPT | ENTITY
    """
    # Truncate to ~6 000 words to stay well within context limits
    trimmed = " ".join(text.split()[:6000])

    system = (
        "You are a knowledge-graph extraction engine. "
        "Your job is to identify every meaningful entity, concept, and topic "
        "inside a piece of text and output them as structured JSON. "
        "Be comprehensive — extract as many distinct, meaningful nodes as possible (aim for 20-35). "
        "Every concept must be a real, meaningful entity — never output generic words. "
        "Always mark exactly ONE concept as the main subject (is_main: true). "
        "The main subject is the primary topic/person/thing the content is ABOUT. "
        "Output ONLY valid JSON — no markdown fences, no explanation."
    )

    prompt = f"""Document title: {title or '(untitled)'}

Text:
\"\"\"
{trimmed}
\"\"\"

Extract all meaningful entities, concepts, and topics. Return JSON array:
[
  {{
    "name": "entity name (concise, properly capitalised)",
    "category": "one of: PERSON|COMPANY|ROLE|EDUCATION|PROJECT|SKILL|TECHNOLOGY|LOCATION|TOPIC|EVENT|CONCEPT|ENTITY",
    "relevance_score": 0.0-1.0,
    "is_main": true or false
  }},
  ...
]

Rules:
- Exactly ONE item must have is_main: true (the primary subject)
- Aim for 20-35 nodes — be thorough, cover all important entities
- relevance_score: 1.0 = central, 0.5 = supporting, 0.3 = peripheral
- Names must be concise proper nouns or short phrases (1-5 words)
- No generic/stop-words as names
"""

    raw = _chat(prompt, system, max_tokens=2000)
    if raw is None:
        return None

    return _parse_json_list(raw)


def groq_extract_relationships(concepts: List[Dict], text: str, title: str = "") -> Optional[List[Dict]]:
    """
    Ask Groq to infer typed relationships among the extracted concepts.

    Returns a list of dicts:
        {source, target, type, label, weight}

    Returns None if Groq is unavailable.
    """
    if not concepts:
        return []

    trimmed = " ".join(text.split()[:5000])
    concept_names = [c["name"] for c in concepts]

    system = (
        "You are a knowledge-graph relationship engine. "
        "Given a list of concepts and the source text, infer meaningful, typed edges between them. "
        "Every edge must be semantically meaningful — no generic 'RELATES_TO' unless nothing else fits. "
        "Output ONLY valid JSON — no markdown fences, no explanation."
    )

    prompt = f"""Document title: {title or '(untitled)'}

Concepts (nodes already in the graph):
{json.dumps(concept_names, indent=2)}

Source text:
\"\"\"
{trimmed}
\"\"\"

Infer relationships between the concepts. Return JSON array:
[
  {{
    "source": "concept name (must exactly match one from the list above)",
    "target": "concept name (must exactly match one from the list above)",
    "type": "RELATIONSHIP_TYPE_IN_UPPER_SNAKE_CASE",
    "label": "human readable label",
    "weight": 0.5-1.0
  }},
  ...
]

Rules:
- The main subject must connect to EVERY other concept with a typed edge
- Also add peer-to-peer edges where a real relationship exists
- Use specific types like: HAS_SKILL, WORKED_AT, STUDIED_AT, CREATED, USES,
  BUILT_WITH, PART_OF, RELATED_TO, LEADS, FOUNDED, PUBLISHED, PARTICIPATED_IN,
  LOCATED_IN, COVERS, DISCUSSES, TAUGHT_BY, COLLABORATED_WITH, etc.
- source and target must exactly match concept names in the list above
- weight 1.0 = very strong, 0.5 = moderate
- Aim for a rich graph — at least one edge per concept
"""

    raw = _chat(prompt, system, max_tokens=2500)
    if raw is None:
        return None

    rels = _parse_json_list(raw)
    if rels is None:
        return None

    # Validate: both ends must exist in the concept list
    name_set = {c["name"] for c in concepts}
    valid = []
    for r in rels:
        src = r.get("source", "")
        tgt = r.get("target", "")
        if src in name_set and tgt in name_set and src != tgt:
            valid.append({
                "source": src,
                "target": tgt,
                "type": r.get("type", "RELATES_TO"),
                "label": r.get("label", r.get("type", "relates to")).replace("_", " ").lower().title(),
                "weight": float(r.get("weight", 0.7)),
            })
    return valid


def groq_summarise(text: str, title: str = "") -> Optional[str]:
    """
    Ask Groq to produce a concise, insightful summary (3-5 sentences).

    Returns None if Groq is unavailable so the caller can fall back.
    """
    trimmed = " ".join(text.split()[:7000])

    system = (
        "You are an expert summariser. Write clear, dense, insightful summaries. "
        "Capture the main subject, key ideas, and important details. "
        "No bullet points — flowing prose only. 3-5 sentences."
    )

    prompt = f"""Title: {title or '(untitled)'}

Text:
\"\"\"
{trimmed}
\"\"\"

Write a concise 3-5 sentence summary that captures the core subject, key ideas, and most important details."""

    return _chat(prompt, system, temperature=0.3, max_tokens=400)


def groq_available() -> bool:
    """Quick check — returns True if Groq is configured and reachable."""
    return _get_client() is not None


def groq_explain_graph(nodes: list, edges: list, doc_titles: list = None) -> Optional[str]:
    """
    Ask Groq to explain the knowledge graph in simple, colour-coded bullet points.
    Returns raw JSON string (list of section dicts) or None if unavailable.
    """
    if not nodes:
        return None

    if doc_titles and len(doc_titles) > 1:
        subject_str = f"Multiple document sources being compared: {', '.join(doc_titles)}"
    elif doc_titles and len(doc_titles) == 1:
        subject_str = f"Source document: {doc_titles[0]}"
    else:
        main_nodes = [n for n in nodes if n.get("is_main")]
        main_names = [n["label"] for n in main_nodes]
        if len(main_names) > 1:
            subject_str = f"Multiple main subjects: {', '.join(main_names)}"
        elif len(main_names) == 1:
            subject_str = f"Main subject: {main_names[0]}"
        else:
            subject_str = "The extracted concepts"

    node_summary = ", ".join(
        f"{n['label']} ({n.get('category', 'concept')})"
        for n in nodes[:400]
    )
    edge_summary = "; ".join(
        f"{e['source'].replace('concept_', '')} -> {e['target'].replace('concept_', '')} [{e.get('label', '')}]"
        for e in edges[:400]
    )

    system = (
        "You are an intelligent knowledge-graph narrator. "
        "Given a knowledge graph representing concepts extracted from specific documents, explain it in simple English using short bullet points grouped by theme. "
        "If there are multiple document sources, YOU MUST explain how they are related to each other based on the concepts they share and how they compare. Dedicate a specific section to their connections. "
        "Output ONLY valid JSON — no markdown fences, no explanation outside JSON."
    )

    prompt = (
        f"Knowledge graph {subject_str}\n\n"
        f"Nodes: {node_summary}\n\n"
        f"Key relationships: {edge_summary}\n\n"
        "Explain this knowledge graph in simple terms. You MUST group insights into EXACTLY 8 distinct thematic sections.\n"
        "If multiple documents/subjects are present, explicitly dedicate sections to 'Shared Connections' and 'Comparative Insights'.\n"
        "Example themes: Overview, Key Skills, Shared Connections, Projects, Comparative Insights, Organizations/Entities, Topics/Concepts, Career/Events.\n"
        "Return a JSON array:\n"
        "[\n"
        "  {\n"
        '    "category": "section title",\n'
        '    "color": "one of: #00d2ff | #4ade80 | #fb923c | #c084fc | #f472b6 | #34d399 | #FFD700 | #f87171 | #a78bfa",\n'
        '    "icon": "a single relevant emoji",\n'
        '    "points": ["short bullet point 1", "short bullet point 2", "short bullet point 3"]\n'
        "  }\n"
        "]\n\n"
        "Rules:\n"
        "- The JSON array MUST contain exactly 8 objects.\n"
        "- Each bullet point must be ONE simple sentence (max 15 words)\n"
        "- Provide 3-4 bullet points per section\n"
        "- Use plain language — no jargon\n"
        "- Make it feel like an insightful human summary\n"
    )

    return _chat(prompt, system, temperature=0.4, max_tokens=1200)


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _parse_json_list(raw: str) -> Optional[List[Dict]]:
    """
    Extract and parse a JSON array from a (possibly messy) LLM response.
    Tries a few strategies before giving up.
    """
    # 1. Direct parse
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # 3. Extract first [...] block
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group(0))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    logger.warning(f"Could not parse JSON list from Groq response: {raw[:200]}")
    return None
