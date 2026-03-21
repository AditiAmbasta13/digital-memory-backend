"""One-time cleanup: migrate graph_store.json to new doc_ids format and remove stale doc_ nodes."""
import json

with open("graph_store.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 1. Remove all legacy doc_ title nodes (upload title nodes should never appear in graph)
data["nodes"] = [
    n for n in data["nodes"]
    if not n["id"].startswith("doc_") and n.get("type") != "document"
]

# 2. Migrate doc_id -> doc_ids on nodes
for n in data["nodes"]:
    if "doc_id" in n and "doc_ids" not in n:
        n["doc_ids"] = [n.pop("doc_id")]
    elif "doc_ids" not in n:
        # Legacy nodes with no tracking — assign 0 as a sentinel (won't be deleted)
        n["doc_ids"] = [0]

# 3. Migrate doc_id -> doc_ids on edges
for e in data["edges"]:
    if "doc_id" in e and "doc_ids" not in e:
        e["doc_ids"] = [e.pop("doc_id")]
    elif "doc_ids" not in e:
        e["doc_ids"] = [0]

# 4. Remove edges whose endpoints no longer exist
node_ids = {n["id"] for n in data["nodes"]}
data["edges"] = [
    e for e in data["edges"]
    if e["source"] in node_ids and e["target"] in node_ids
]

print("Nodes:", len(data["nodes"]), "| Edges:", len(data["edges"]))

with open("graph_store.json", "w", encoding="utf-8") as f:
    json.dump(data, f)

print("Done — graph_store.json cleaned and migrated!")
