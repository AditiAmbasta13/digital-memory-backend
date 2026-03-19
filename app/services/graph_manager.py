"""Knowledge graph manager for Neo4j operations."""
import json
import os
import logging
from typing import List, Dict, Optional
from app.db.neo4j_driver import get_neo4j_session

logger = logging.getLogger(__name__)

_GRAPH_STORE_PATH = "./graph_store.json"

# In-memory fallback graph — loaded from disk on first access
_fallback_graph: Dict = {"nodes": [], "edges": []}
_graph_loaded = False


def _load_fallback():
    global _fallback_graph, _graph_loaded
    if _graph_loaded:
        return
    _graph_loaded = True
    if os.path.exists(_GRAPH_STORE_PATH):
        try:
            with open(_GRAPH_STORE_PATH, "r", encoding="utf-8") as f:
                _fallback_graph = json.load(f)
            logger.info(f"Graph loaded from disk: {len(_fallback_graph['nodes'])} nodes, {len(_fallback_graph['edges'])} edges")
        except Exception as e:
            logger.warning(f"Could not load graph store: {e}. Starting fresh.")
            _fallback_graph = {"nodes": [], "edges": []}


def _save_fallback():
    try:
        with open(_GRAPH_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(_fallback_graph, f)
    except Exception as e:
        logger.warning(f"Could not save graph store: {e}")


def add_document_to_graph(doc_id: int, title: str, concepts: List[Dict], relationships: List[Dict]):
    """Add a document and its concepts to the knowledge graph."""
    session = get_neo4j_session()

    if session is None:
        # Fallback: store in memory
        _add_to_fallback(doc_id, title, concepts, relationships)
        return

    try:
        with session:
            # Create document node
            session.run(
                "MERGE (d:Document {doc_id: $doc_id}) SET d.title = $title",
                doc_id=doc_id, title=title,
            )

            # Create concept nodes and relationships
            for concept in concepts:
                session.run(
                    """
                    MERGE (c:Concept {name: $name})
                    SET c.category = $category
                    WITH c
                    MATCH (d:Document {doc_id: $doc_id})
                    MERGE (d)-[r:CONTAINS_CONCEPT]->(c)
                    SET r.relevance = $relevance
                    """,
                    name=concept["name"],
                    category=concept.get("category", "CONCEPT"),
                    doc_id=doc_id,
                    relevance=concept.get("relevance_score", 1.0),
                )

            # Create concept-concept relationships
            for rel in relationships:
                session.run(
                    """
                    MATCH (c1:Concept {name: $source})
                    MATCH (c2:Concept {name: $target})
                    MERGE (c1)-[r:RELATES_TO]->(c2)
                    SET r.weight = $weight
                    """,
                    source=rel["source"],
                    target=rel["target"],
                    weight=rel.get("weight", 1.0),
                )

            logger.info(f"Added document {doc_id} with {len(concepts)} concepts to graph")
    except Exception as e:
        logger.error(f"Neo4j operation failed: {e}")
        _add_to_fallback(doc_id, title, concepts, relationships)


def get_graph_data(limit: int = 100, doc_ids: Optional[List[int]] = None) -> Dict:
    """Get graph data for visualization. Optionally filter by document IDs."""
    _load_fallback()
    session = get_neo4j_session()

    if session is None:
        return _get_fallback_graph(doc_ids=doc_ids)

    try:
        with session:
            result = session.run(
                """
                MATCH (d:Document)-[r:CONTAINS_CONCEPT]->(c:Concept)
                RETURN d, r, c
                LIMIT $limit
                """,
                limit=limit,
            )

            nodes = {}
            edges = []

            for record in result:
                doc = record["d"]
                concept = record["c"]

                doc_id = f"doc_{doc['doc_id']}"
                concept_id = f"concept_{concept['name']}"

                if doc_id not in nodes:
                    nodes[doc_id] = {
                        "id": doc_id,
                        "label": doc["title"],
                        "type": "document",
                        "color": "#00d2ff",
                    }

                if concept_id not in nodes:
                    nodes[concept_id] = {
                        "id": concept_id,
                        "label": concept["name"],
                        "type": "concept",
                        "color": "#7b2ff7",
                    }

                edges.append({
                    "source": doc_id,
                    "target": concept_id,
                    "label": "CONTAINS",
                    "weight": record["r"].get("relevance", 1.0),
                })

            # Get concept-concept relationships
            rel_result = session.run(
                """
                MATCH (c1:Concept)-[r:RELATES_TO]->(c2:Concept)
                RETURN c1.name as source, c2.name as target, r.weight as weight
                LIMIT $limit
                """,
                limit=limit,
            )

            for record in rel_result:
                edges.append({
                    "source": f"concept_{record['source']}",
                    "target": f"concept_{record['target']}",
                    "label": "RELATES_TO",
                    "weight": record.get("weight", 1.0),
                })

            return {"nodes": list(nodes.values()), "edges": edges}
    except Exception as e:
        logger.error(f"Graph fetch failed: {e}")
        return _get_fallback_graph(doc_ids=doc_ids)


def get_node_details(node_id: str) -> Optional[Dict]:
    """Get details for a specific node."""
    session = get_neo4j_session()

    if session is None:
        return _get_fallback_node(node_id)

    try:
        with session:
            if node_id.startswith("doc_"):
                doc_id = int(node_id.replace("doc_", ""))
                result = session.run(
                    """
                    MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS_CONCEPT]->(c:Concept)
                    RETURN d.title as title, collect(c.name) as concepts
                    """,
                    doc_id=doc_id,
                )
                record = result.single()
                if record:
                    return {
                        "id": node_id,
                        "label": record["title"],
                        "type": "document",
                        "related_concepts": record["concepts"],
                    }
            elif node_id.startswith("concept_"):
                name = node_id.replace("concept_", "")
                result = session.run(
                    """
                    MATCH (c:Concept {name: $name})<-[:CONTAINS_CONCEPT]-(d:Document)
                    OPTIONAL MATCH (c)-[:RELATES_TO]-(c2:Concept)
                    RETURN c.name as name, c.category as category,
                           collect(DISTINCT d.title) as documents,
                           collect(DISTINCT c2.name) as related_concepts
                    """,
                    name=name,
                )
                record = result.single()
                if record:
                    return {
                        "id": node_id,
                        "label": record["name"],
                        "type": "concept",
                        "category": record["category"],
                        "related_documents": record["documents"],
                        "related_concepts": record["related_concepts"],
                    }
    except Exception as e:
        logger.error(f"Node detail fetch failed: {e}")

    return None


# ---- Fallback in-memory graph ----

def _add_to_fallback(doc_id: int, title: str, concepts: List[Dict], relationships: List[Dict]):
    """Store graph data on disk when Neo4j is unavailable."""
    _load_fallback()

    # We never add a separate document title node — the hub concept IS the
    # main subject and serves as the visual centre.
    # Every node carries a list of doc_ids so shared concepts are only removed
    # when ALL their source documents are deleted.

    CATEGORY_COLORS = {
        "PERSON":     "#FFD700",  # gold  — hub
        "EDUCATION":  "#4ade80",  # green
        "ROLE":       "#fb923c",  # orange
        "PROJECT":    "#c084fc",  # pink-purple
        "SKILL":      "#7b2ff7",  # purple
        "COMPANY":    "#38bdf8",  # sky blue
        "TECHNOLOGY": "#a78bfa",  # violet
        "LOCATION":   "#f472b6",  # pink
        "TOPIC":      "#34d399",  # emerald
        "CONCEPT":    "#fbbf24",  # amber
        "EVENT":      "#f87171",  # red
        "ENTITY":     "#64748b",  # slate
    }

    # Build a lookup for quicker updates
    existing_by_id: Dict[str, Dict] = {n["id"]: n for n in _fallback_graph["nodes"]}

    for concept in concepts:
        concept_id = f"concept_{concept['name']}"
        cat = concept.get("category", "ENTITY")
        is_main = concept.get("is_main", False)
        if concept_id in existing_by_id:
            # Node already exists — just tag it with this doc_id too
            node = existing_by_id[concept_id]
            doc_ids = node.get("doc_ids", [])
            # Migrate legacy single-value doc_id field
            if "doc_id" in node and node["doc_id"] not in doc_ids:
                doc_ids.append(node.pop("doc_id"))
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
            node["doc_ids"] = doc_ids
        else:
            new_node = {
                "id": concept_id,
                "label": concept["name"],
                "type": "main" if is_main else "concept",
                "category": cat,
                "color": "#FFD700" if is_main else CATEGORY_COLORS.get(cat, "#7b2ff7"),
                "is_main": is_main,
                # Store as list so shared concepts survive partial deletes
                "doc_ids": [doc_id],
            }
            _fallback_graph["nodes"].append(new_node)
            existing_by_id[concept_id] = new_node

    existing_edge_keys = {
        (e["source"], e["target"]) for e in _fallback_graph["edges"]
    }
    for rel in relationships:
        src = f"concept_{rel['source']}"
        tgt = f"concept_{rel['target']}"
        if src in existing_by_id and tgt in existing_by_id:
            key = (src, tgt)
            if key in existing_edge_keys:
                # Tag existing edge with this doc too
                for e in _fallback_graph["edges"]:
                    if e["source"] == src and e["target"] == tgt:
                        doc_ids = e.get("doc_ids", [])
                        if "doc_id" in e and e["doc_id"] not in doc_ids:
                            doc_ids.append(e.pop("doc_id"))
                        if doc_id not in doc_ids:
                            doc_ids.append(doc_id)
                        e["doc_ids"] = doc_ids
                        break
            else:
                rel_label = rel.get("label") or rel.get("type") or "RELATES TO"
                _fallback_graph["edges"].append({
                    "source": src,
                    "target": tgt,
                    "label": rel_label,
                    "weight": rel.get("weight", 1.0),
                    "doc_ids": [doc_id],
                })
                existing_edge_keys.add(key)

    logger.info(f"Added document {doc_id} to fallback graph ({len(_fallback_graph['nodes'])} nodes)")
    _save_fallback()


def remove_document_from_graph(doc_id: int):
    """Remove all nodes and edges that belong exclusively to this document."""
    _load_fallback()
    doc_node_id = f"doc_{doc_id}"

    def _node_survives(n: Dict) -> bool:
        # Always purge legacy doc_ title nodes for any document
        if n["id"].startswith("doc_"):
            return False
        # Migrate legacy single doc_id field
        if "doc_id" in n and "doc_ids" not in n:
            n["doc_ids"] = [n.pop("doc_id")]
        doc_ids: List = n.get("doc_ids", [])
        # Remove this doc from the node's owner list
        doc_ids = [d for d in doc_ids if d != doc_id]
        n["doc_ids"] = doc_ids
        # Keep node only if it still has other owners
        return len(doc_ids) > 0

    _fallback_graph["nodes"] = [n for n in _fallback_graph["nodes"] if _node_survives(n)]

    # Collect surviving node IDs
    remaining_ids = {n["id"] for n in _fallback_graph["nodes"]}

    def _edge_survives(e: Dict) -> bool:
        # Drop legacy doc_ source/target edges
        if e["source"] == doc_node_id or e["target"] == doc_node_id:
            return False
        # Migrate legacy single doc_id
        if "doc_id" in e and "doc_ids" not in e:
            e["doc_ids"] = [e.pop("doc_id")]
        doc_ids: List = e.get("doc_ids", [])
        doc_ids = [d for d in doc_ids if d != doc_id]
        e["doc_ids"] = doc_ids
        # Drop edge if both endpoints are gone OR if edge has no owners left
        if e["source"] not in remaining_ids or e["target"] not in remaining_ids:
            return False
        return len(doc_ids) > 0

    _fallback_graph["edges"] = [e for e in _fallback_graph["edges"] if _edge_survives(e)]

    _save_fallback()
    logger.info(f"Removed document {doc_id} from graph")


def _get_fallback_graph(doc_ids: Optional[List[int]] = None) -> Dict:
    _load_fallback()
    # Filter out any legacy document-type nodes
    concept_ids = {n["id"] for n in _fallback_graph["nodes"] if n.get("type") != "document"}
    
    nodes = [n for n in _fallback_graph["nodes"] if n.get("type") != "document"]
    edges = [
        e for e in _fallback_graph["edges"]
        if e["source"] in concept_ids and e["target"] in concept_ids
    ]

    # Filter by specific document IDs if requested
    if doc_ids is not None:
        doc_ids_set = set(doc_ids)
        
        # Keep nodes where at least one of their doc_ids overlaps with requested
        def node_has_doc(n):
            n_ids = n.get("doc_ids", [])
            # Also check old doc_id for safety
            if "doc_id" in n and n["doc_id"] not in n_ids:
                n_ids.append(n["doc_id"])
            return bool(set(n_ids).intersection(doc_ids_set))
            
        nodes = [n for n in nodes if node_has_doc(n)]
        
        valid_node_ids = {n["id"] for n in nodes}
        
        # Keep edges where at least one of their doc_ids overlaps with requested AND both nodes exist
        def edge_has_doc(e):
            e_ids = e.get("doc_ids", [])
            if "doc_id" in e and e["doc_id"] not in e_ids:
                e_ids.append(e["doc_id"])
            return bool(set(e_ids).intersection(doc_ids_set))
            
        edges = [e for e in edges if e["source"] in valid_node_ids and e["target"] in valid_node_ids and edge_has_doc(e)]

    return {"nodes": nodes, "edges": edges}


def _get_fallback_node(node_id: str) -> Optional[Dict]:
    _load_fallback()
    for node in _fallback_graph["nodes"]:
        if node["id"] == node_id:
            connections = [
                e for e in _fallback_graph["edges"]
                if e["source"] == node_id or e["target"] == node_id
            ]
            related = []
            for e in connections:
                other_id = e["target"] if e["source"] == node_id else e["source"]
                for n in _fallback_graph["nodes"]:
                    if n["id"] == other_id:
                        related.append(n["label"])
            return {**node, "connections": len(connections), "related": related}
    return None
