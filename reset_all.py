"""Full reset: wipe DB records, graph store, vector store, and uploaded files."""
import json
import os
import shutil

# ── 1. Reset graph store ──────────────────────────────────────────────────────
with open("graph_store.json", "w", encoding="utf-8") as f:
    json.dump({"nodes": [], "edges": []}, f)
print("✓ graph_store.json cleared")

# ── 2. Reset vector store ─────────────────────────────────────────────────────
with open("vector_store.json", "w", encoding="utf-8") as f:
    json.dump([], f)
print("✓ vector_store.json cleared")

# ── 3. Clear uploaded files ───────────────────────────────────────────────────
uploads_dir = "./uploads"
if os.path.exists(uploads_dir):
    for fname in os.listdir(uploads_dir):
        fpath = os.path.join(uploads_dir, fname)
        try:
            if os.path.isfile(fpath):
                os.remove(fpath)
            elif os.path.isdir(fpath):
                shutil.rmtree(fpath)
        except Exception as e:
            print(f"  Warning: could not remove {fpath}: {e}")
    print("✓ uploads/ folder cleared")
else:
    print("  uploads/ folder not found - skipping")

# ── 4. Wipe DB tables ─────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.db.session import SessionLocal
from app.models.database import Document, Summary, DocumentConcept

db = SessionLocal()
try:
    deleted_concepts = db.query(DocumentConcept).delete()
    deleted_summaries = db.query(Summary).delete()
    deleted_docs = db.query(Document).delete()
    db.commit()
    print(f"✓ Database cleared — {deleted_docs} docs, {deleted_summaries} summaries, {deleted_concepts} concepts removed")
finally:
    db.close()

print("\n🎉 All data wiped. Ready for a fresh start!")
