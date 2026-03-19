"""NLP processing pipeline for concept extraction and relationship analysis."""
import re
import logging
from typing import List, Dict, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

_nlp = None


def _get_nlp():
    """Lazy-load spaCy model."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded")
        except Exception as e:
            logger.warning(f"spaCy not available: {e}. Using fallback NLP.")
            _nlp = "fallback"
    return _nlp


# ──────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def extract_concepts(text: str, title: str = "", max_concepts: int = 30) -> List[Dict]:
    """
    Extract meaningful concepts from text.

    Always returns one concept marked  is_main=True  — this is the hub node
    (the primary subject of the content).  All other concepts are connected to
    it in extract_relationships().

    Returns list of dicts: {name, category, relevance_score, is_main}.
    """
    nlp = _get_nlp()
    if nlp == "fallback":
        return _fallback_extract_concepts(text, title, max_concepts)
    return _spacy_extract_concepts(nlp, text, title, max_concepts)


def extract_relationships(concepts: List[Dict], text: str) -> List[Dict]:
    """
    Build a rich edge list from the concept list.

    Strategy:
    1. Hub–spoke: the main-subject node connects to EVERY other node with a
       semantically typed edge (STUDIES_AT, HAS_SKILL, CREATED, WORKED_AS …).
    2. Peer–peer: concepts that co-occur in the same sentence are linked with
       a typed edge (RELATED_TECH, BUILT_WITH, USED_IN, LEARNED_AT, PART_OF …).
    """
    if not concepts:
        return []

    text_lower = text.lower()
    relationships: List[Dict] = []
    seen: set = set()

    # ── find hub ──────────────────────────────────────────────────────────────
    main = next((c for c in concepts if c.get("is_main")), None)
    if main is None:
        main = concepts[0]
    main_name = main["name"]

    def add_rel(src: str, tgt: str, rel: str, weight: float):
        key = tuple(sorted([src, tgt]))
        if key in seen:
            return
        seen.add(key)
        relationships.append({
            "source": src,
            "target": tgt,
            "type": rel,
            "label": rel.replace("_", " "),
            "weight": round(weight, 2),
        })

    # ── 1. Hub → every other concept (typed) ─────────────────────────────────
    CAT_TO_REL = {
        "EDUCATION": ("STUDIES_AT",   0.95),
        "ROLE":      ("WORKED_AS",    0.90),
        "PROJECT":   ("CREATED",      0.88),
        "SKILL":     ("HAS_SKILL",    0.85),
        "COMPANY":   ("WORKED_AT",    0.85),
        "ENTITY":    ("ASSOCIATED_WITH", 0.65),
        "TOPIC":     ("COVERS",       0.70),
        "CONCEPT":   ("DISCUSSES",    0.65),
        "LOCATION":  ("LOCATED_IN",   0.70),
        "PERSON":    ("KNOWS",        0.60),
        "EVENT":     ("PARTICIPATED_IN", 0.70),
        "TECHNOLOGY":("USES",         0.80),
    }
    for c in concepts:
        cname = c["name"]
        if cname == main_name:
            continue
        cat = c.get("category", "ENTITY")
        rel, w = CAT_TO_REL.get(cat, ("RELATES_TO", 0.55))
        add_rel(main_name, cname, rel, w)

    # ── 2. Peer–peer from sentence co-occurrence ──────────────────────────────
    sentences = re.split(r"[.\n!?;]", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    concept_names = [c["name"] for c in concepts]
    cat_map = {c["name"]: c.get("category", "ENTITY") for c in concepts}

    ACTION_WORDS = {
        "built", "build", "developed", "develop", "created", "create",
        "designed", "implemented", "integrated", "used", "using",
        "leveraged", "deployed", "launched", "worked", "working",
        "collaborated", "managed", "led", "powered", "based",
        "project", "application", "system", "platform", "intern", "internship",
    }

    MAX_PEER_EDGES = 40

    for sentence in sentences:
        if len(relationships) >= MAX_PEER_EDGES + len(concepts):
            break
        sent_lower = sentence.lower()
        # which concepts (excluding hub) appear here?
        present = [
            name for name in concept_names
            if name != main_name and name.lower() in sent_lower
        ]
        if len(present) < 2:
            continue

        has_action = any(w in sent_lower for w in ACTION_WORDS)

        for i, c1 in enumerate(present):
            for c2 in present[i + 1:]:
                r1 = cat_map.get(c1, "ENTITY")
                r2 = cat_map.get(c2, "ENTITY")
                pair = {r1, r2}

                rel, weight = _peer_rel(r1, r2, pair, has_action)
                if rel:
                    add_rel(c1, c2, rel, weight)

    return relationships


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _peer_rel(r1: str, r2: str, pair: set, has_action: bool) -> Tuple[Optional[str], float]:
    """Return (relationship_label, weight) for a peer pair, or (None, 0)."""
    if r1 == "SKILL" and r2 == "SKILL":
        return ("RELATED_TECH", 0.75) if has_action else (None, 0)
    if "SKILL" in pair and "PROJECT" in pair:
        return ("BUILT_WITH", 0.82)
    if "SKILL" in pair and "ROLE" in pair:
        return ("USED_IN", 0.80)
    if "SKILL" in pair and "EDUCATION" in pair:
        return ("LEARNED_AT", 0.72)
    if "SKILL" in pair and "TECHNOLOGY" in pair:
        return ("USED_WITH", 0.75)
    if "PROJECT" in pair and "ROLE" in pair:
        return ("DEVELOPED_AT", 0.75)
    if "PROJECT" in pair and "COMPANY" in pair:
        return ("BUILT_FOR", 0.75)
    if "PROJECT" in pair and "TECHNOLOGY" in pair:
        return ("USES", 0.72)
    if "ROLE" in pair and "COMPANY" in pair:
        return ("AT_COMPANY", 0.85)
    if "EDUCATION" in pair and "LOCATION" in pair:
        return ("LOCATED_IN", 0.70)
    if "CONCEPT" in pair or "TOPIC" in pair:
        return ("RELATES_TO", 0.55) if has_action else (None, 0)
    if has_action:
        return ("RELATES_TO", 0.50)
    return (None, 0)


def _get_entity_role(name: str, text_lower: str) -> str:
    """Determine contextual role for a concept name."""
    idx = text_lower.find(name.lower())
    if idx == -1:
        return "ENTITY"
    window = text_lower[max(0, idx - 150): idx + 150]

    if any(w in window for w in ["intern", "internship", "worked at", "working at",
                                  "employed", "position at", "role at", "joined",
                                  "software engineer", "developer at", "engineer at"]):
        return "ROLE"
    if any(w in window for w in ["college", "university", "institute", "school",
                                  "b.tech", "btech", "b.e", "bachelor", "master",
                                  "degree", "graduation", "engineering"]):
        return "EDUCATION"
    if any(w in window for w in ["built", "created", "developed", "designed",
                                  "project", "app", "application", "platform",
                                  "system", "website", "tool", "launched"]):
        return "PROJECT"
    if any(w in window for w in ["city", "india", "mumbai", "pune", "delhi",
                                  "bangalore", "located", "based in", "country",
                                  "state", "region"]):
        return "LOCATION"
    return "ENTITY"


# ──────────────────────────────────────────────────────────────────────────────
#  SPACY PATH
# ──────────────────────────────────────────────────────────────────────────────

def _spacy_extract_concepts(nlp, text: str, title: str, max_concepts: int) -> List[Dict]:
    """Extract concepts using spaCy NER + noun chunks."""
    doc = nlp(text[:100_000])

    entity_counts: Counter = Counter()
    entity_labels: Dict[str, str] = {}
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART",
                          "EVENT", "LAW", "NORP", "FAC", "LOC"):
            name = ent.text.strip()
            if len(name) > 2:
                entity_counts[name] += 1
                entity_labels[name] = ent.label_

    chunk_counts: Counter = Counter()
    for chunk in doc.noun_chunks:
        name = chunk.text.strip().lower()
        if 3 < len(name) <= 50 and len(name.split()) <= 5:
            chunk_counts[name] += 1

    LABEL_MAP = {
        "PERSON": "PERSON", "ORG": "COMPANY", "GPE": "LOCATION",
        "LOC": "LOCATION", "FAC": "LOCATION", "PRODUCT": "TECHNOLOGY",
        "WORK_OF_ART": "PROJECT", "EVENT": "EVENT", "NORP": "ENTITY",
        "LAW": "ENTITY",
    }

    concepts: List[Dict] = []
    seen: set = set()

    for name, count in entity_counts.most_common(max_concepts):
        low = name.lower()
        if low in seen:
            continue
        seen.add(low)
        label = entity_labels.get(name, "ENTITY")
        concepts.append({
            "name": name,
            "category": LABEL_MAP.get(label, "ENTITY"),
            "relevance_score": min(count / 5.0, 1.0),
            "is_main": False,
        })

    for name, count in chunk_counts.most_common(max_concepts * 2):
        if len(concepts) >= max_concepts:
            break
        if name in seen:
            continue
        seen.add(name)
        concepts.append({
            "name": name.title(),
            "category": "TOPIC",
            "relevance_score": min(count / 10.0, 0.8),
            "is_main": False,
        })

    _mark_main_subject(concepts, text, title)
    return concepts[:max_concepts]


# ──────────────────────────────────────────────────────────────────────────────
#  FALLBACK PATH  (no spaCy)
# ──────────────────────────────────────────────────────────────────────────────

def _fallback_extract_concepts(text: str, title: str = "", max_concepts: int = 30) -> List[Dict]:
    """
    Regex-based concept extractor.

    Categories: PERSON | COMPANY | ROLE | EDUCATION | PROJECT |
                SKILL | TECHNOLOGY | LOCATION | ENTITY | TOPIC
    """
    concepts: List[Dict] = []
    seen: set = set()
    text_lower = text.lower()

    def add(name: str, category: str, score: float, is_main: bool = False):
        key = name.lower().strip()
        if key in seen or len(key) < 3:
            return
        seen.add(key)
        concepts.append({
            "name": name.strip(),
            "category": category,
            "relevance_score": round(score, 2),
            "is_main": is_main,
        })

    # ── 1. PERSON ─────────────────────────────────────────────────────────────
    person_pat = re.compile(r'\b([A-Z][a-z]{1,14})\s+([A-Z][a-z]{1,14})\b')
    first_lines = [l.strip() for l in text.split("\n") if l.strip()][:6]
    HEADER_NOISE = {
        "technical skills", "work experience", "education section",
        "contact information", "personal information", "summary",
    }
    for line in first_lines:
        m = person_pat.search(line)
        if m and m.group(0).lower() not in HEADER_NOISE:
            add(m.group(0), "PERSON", 1.0)
            break

    # ── 2. ROLE / Job titles ──────────────────────────────────────────────────
    ROLE_PATTERNS = [
        r'\b((?:Research|Software|Frontend|Backend|Full[- ]?Stack|Data|ML|AI|Web|Mobile|'
        r'Cloud|DevOps|Product|Machine Learning|Deep Learning)\s+'
        r'(?:Engineer|Developer|Intern(?:ship)?|Analyst|Scientist|Architect|Lead|Manager))\b',
        r'\b((?:Senior|Junior|Associate|Staff)\s+(?:Engineer|Developer|Analyst|Designer|Manager))\b',
    ]
    for pat in ROLE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            add(m.group(0).title(), "ROLE", 0.90)
        if sum(1 for c in concepts if c["category"] == "ROLE") >= 5:
            break

    # ── 3. EDUCATION ──────────────────────────────────────────────────────────
    EDU_PAT = (
        r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\s+'
        r'(?:College|University|Institute|School|Academy)'
        r'(?:\s+of\s+[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)?)\b'
    )
    for m in re.finditer(EDU_PAT, text):
        name = m.group(0).strip()
        if len(name) > 6:
            add(name, "EDUCATION", 0.90)
        if sum(1 for c in concepts if c["category"] == "EDUCATION") >= 3:
            break

    # ── 4. COMPANY — proper noun + company suffix ─────────────────────────────
    COMPANY_PAT = (
        r'\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\s+'
        r'(?:Inc|Ltd|LLC|Corp|Technologies|Solutions|Systems|Labs|AI|Software|'
        r'Analytics|Consulting|Ventures|Group|Digital|Networks|Services|Platform)\.?)\b'
    )
    for m in re.finditer(COMPANY_PAT, text):
        name = m.group(0).strip().rstrip(".")
        if 3 < len(name) < 50:
            add(name, "COMPANY", 0.85)
    # Also grab well-known 1–2 word CamelCase proper nouns near work contexts
    WORK_CONTEXT = re.compile(
        r'(?:at|with|for|join(?:ed)?|intern(?:ed)?|work(?:ed)?\s+at)\s+'
        r'([A-Z][a-zA-Z0-9]{2,}(?:\s+[A-Z][a-zA-Z0-9]{2,})?)',
        re.IGNORECASE,
    )
    for m in WORK_CONTEXT.finditer(text):
        name = m.group(1).strip()
        if 3 < len(name) < 40 and name[0].isupper():
            add(name, "COMPANY", 0.80)

    # ── 5. PROJECT ────────────────────────────────────────────────────────────
    PROJ_PAT = re.compile(
        r'(?:built|developed|created|designed|launched|implemented|worked on)\s+'
        r'(?:a\s+|an\s+|the\s+)?([A-Z][A-Za-z0-9\-\.]+(?:\s+[A-Z][A-Za-z0-9\-\.]+){0,3})',
        re.IGNORECASE,
    )
    for m in PROJ_PAT.finditer(text):
        pname = m.group(1).strip()
        if 3 < len(pname) < 45:
            add(pname.title(), "PROJECT", 0.85)
    # Short "Title: …" lines often are project/section names
    for line in text.split("\n"):
        line = line.strip()
        if 4 < len(line) < 40 and line[0].isupper() and ":" in line[:30]:
            candidate = line.split(":")[0].strip()
            if candidate and not any(
                w in candidate.lower()
                for w in ["skill", "education", "experience", "contact", "email",
                          "phone", "language", "framework", "tool", "project",
                          "certification", "summary", "objective"]
            ):
                add(candidate, "PROJECT", 0.75)
    # Cap projects at 5
    proj = [c for c in concepts if c["category"] == "PROJECT"][:5]
    concepts = [c for c in concepts if c["category"] != "PROJECT"] + proj

    # ── 6. SKILLS (tech) ──────────────────────────────────────────────────────
    TECH_SKILLS = [
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
        "kotlin", "swift", "react", "next.js", "nextjs", "vue", "angular",
        "node.js", "nodejs", "express", "fastapi", "django", "flask", "spring",
        "tensorflow", "pytorch", "scikit-learn", "huggingface", "langchain",
        "pandas", "numpy", "sql", "postgresql", "mysql", "mongodb", "redis",
        "firebase", "supabase", "graphql", "rest api", "docker", "kubernetes",
        "aws", "gcp", "azure", "git", "html", "css", "tailwind", "figma",
        "machine learning", "deep learning", "data science", "artificial intelligence",
        "natural language processing", "computer vision", "reinforcement learning",
        "socket.io", "hadoop", "spark", "kafka", "elasticsearch", "jenkins",
        "terraform", "linux", "bash", "matlab", "r", "scala",
    ]
    skill_counts = []
    for skill in TECH_SKILLS:
        n = len(re.findall(r'\b' + re.escape(skill) + r'\b', text_lower))
        if n > 0:
            skill_counts.append((skill, n))
    skill_counts.sort(key=lambda x: x[1], reverse=True)
    for skill, count in skill_counts[:12]:
        add(skill.title(), "SKILL", min(0.5 + count * 0.12, 1.0))

    # ── 7. LOCATIONS ──────────────────────────────────────────────────────────
    CITIES = [
        "mumbai", "pune", "delhi", "bangalore", "bengaluru", "hyderabad",
        "chennai", "kolkata", "ahmedabad", "london", "new york", "san francisco",
        "seattle", "boston", "berlin", "toronto", "singapore", "dubai",
        "india", "usa", "united states", "uk", "canada", "australia",
    ]
    for city in CITIES:
        if city in text_lower:
            add(city.title(), "LOCATION", 0.65)

    # ── 8. TOPIC — key noun phrases for article/note content ──────────────────
    # (only if we don't already have a lot of concepts)
    if len(concepts) < 20:
        # Extract noun phrases: Capitalized 2–4 word phrases appearing ≥ 2 times
        noun_phrases = Counter(
            re.findall(r'\b([A-Z][a-z]+(?:\s+[a-z]+){1,3})\b', text)
        )
        PDF_NOISE = {
            "adobe", "acrobat", "mozilla", "firefox", "chrome", "portable",
            "unicode", "postscript", "ghostscript", "copyright", "rights",
            "reserved", "version", "figure", "table", "section", "chapter",
        }
        STOP_PHRASES = {
            "the", "and", "for", "with", "this", "that", "from", "have", "been",
        }
        for phrase, count in noun_phrases.most_common(30):
            if len(concepts) >= max_concepts:
                break
            low = phrase.lower()
            if count < 2:
                continue
            if any(n in low for n in PDF_NOISE):
                continue
            if low in STOP_PHRASES:
                continue
            if any(w in low for w in ["college", "university", "engineer",
                                       "developer", "intern"]):
                continue
            add(phrase, "TOPIC", min(0.4 + count * 0.08, 0.85))

    # ── 9. ENTITY — remaining proper nouns ───────────────────────────────────
    multi_word = re.findall(r'\b[A-Z][a-z]{1,}(?:\s+[A-Z][a-z]{1,})+\b', text)
    ent_counts: Counter = Counter(multi_word)
    ENTITY_NOISE = {
        "the", "and", "for", "with", "this", "that", "from",
        "adobe", "acrobat", "mozilla", "firefox", "chrome",
    }
    for phrase, _ in ent_counts.most_common(max_concepts):
        if len(concepts) >= max_concepts:
            break
        low = phrase.lower()
        if any(n in low for n in ENTITY_NOISE):
            continue
        if any(w in low for w in ["college", "university", "institute",
                                   "intern", "engineer", "developer"]):
            continue
        add(phrase, "ENTITY", 0.55)

    # ── Mark main subject ─────────────────────────────────────────────────────
    _mark_main_subject(concepts, text, title)

    concepts.sort(key=lambda x: x["relevance_score"], reverse=True)
    return concepts[:max_concepts]


def _mark_main_subject(concepts: List[Dict], text: str, title: str):
    """
    Set is_main=True on the single best hub node.

    Priority:
      1. A PERSON concept that appears in the first 5 lines (resume hub)
      2. A concept whose name closely matches the document title
      3. The highest-relevance COMPANY or ROLE (professional doc)
      4. The concept with the highest relevance_score overall
    """
    if not concepts:
        return

    # Reset
    for c in concepts:
        c["is_main"] = False

    first_lines = " ".join(text.split("\n")[:5]).lower()
    title_lower = title.lower().strip()

    # 1. Person in first lines
    for c in concepts:
        if c["category"] == "PERSON" and c["name"].lower() in first_lines:
            c["is_main"] = True
            return

    # 2. Title match
    if title_lower:
        best_match = None
        best_score = 0.0
        for c in concepts:
            name_lower = c["name"].lower()
            if name_lower in title_lower or title_lower in name_lower:
                if c["relevance_score"] > best_score:
                    best_score = c["relevance_score"]
                    best_match = c
        if best_match:
            best_match["is_main"] = True
            return
        # Partial word overlap
        title_words = set(title_lower.split())
        for c in concepts:
            name_words = set(c["name"].lower().split())
            overlap = len(title_words & name_words)
            score = overlap / max(len(title_words), 1) * c["relevance_score"]
            if score > best_score:
                best_score = score
                best_match = c
        if best_match and best_score > 0.3:
            best_match["is_main"] = True
            return

    # 3. Highest-relevance PERSON / COMPANY / ROLE
    for cat in ("PERSON", "COMPANY", "ROLE"):
        candidates = [c for c in concepts if c["category"] == cat]
        if candidates:
            candidates.sort(key=lambda c: c["relevance_score"], reverse=True)
            candidates[0]["is_main"] = True
            return

    # 4. Fallback: highest relevance
    concepts[0]["is_main"] = True
