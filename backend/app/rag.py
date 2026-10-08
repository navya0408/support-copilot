"""Retrieval over the knowledge base: markdown -> chunks -> embeddings -> ChromaDB.

Embeddings use Chroma's built-in default (all-MiniLM-L6-v2 via ONNX), which needs no
PyTorch and no API key, so it stays small enough for free hosting.
"""
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from .config import CHROMA_DIR, KB_DIR

COLLECTION = "support_kb"
_client = None
_collection = None


def load_chunks(kb_dir: Path = KB_DIR):
    """One chunk per '## section' of each markdown file, prefixed with the doc title."""
    chunks = []
    for path in sorted(Path(kb_dir).glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        text = path.read_text(encoding="utf-8")
        m = re.search(r"^# (.+)$", text, re.M)
        title = m.group(1).strip() if m else path.stem
        parts = re.split(r"^## ", text, flags=re.M)[1:]      # [0] is the title block
        for i, part in enumerate(parts):
            heading, _, body = part.partition("\n")
            heading, body = heading.strip(), body.strip()
            if not body:
                continue
            chunks.append({
                "id": f"{path.stem}#{i}",
                "source": path.stem,
                "title": title,
                "section": heading,
                "text": f"{title} - {heading}\n{body}",
            })
    return chunks


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


def _make_collection():
    return _get_client().get_or_create_collection(
        name=COLLECTION,
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},        # distance = 1 - cosine similarity
    )


def _get_collection():
    global _collection
    if _collection is None:
        _collection = _make_collection()
    return _collection


def ensure_index(force: bool = False) -> int:
    """(Re)build the vector index if missing or if the number of chunks changed."""
    global _collection
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError(f"No markdown files found in {KB_DIR}")
    col = _get_collection()
    if not force and col.count() == len(chunks):
        existing = col.get(include=["documents"])
        if dict(zip(existing["ids"], existing["documents"])) == {c["id"]: c["text"] for c in chunks}:
            return col.count()          # same ids AND same text: nothing to do
    _get_client().delete_collection(COLLECTION)
    _collection = _make_collection()
    _collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "title": c["title"], "section": c["section"]} for c in chunks],
    )
    return _collection.count()


def count() -> int:
    return _get_collection().count()


def retrieve(query: str, k: int = 3):
    """Top-k chunks with cosine distance (lower = more similar)."""
    col = _get_collection()
    res = col.query(query_texts=[query[:1500]], n_results=min(k, col.count()))
    out = []
    for id_, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                    res["metadatas"][0], res["distances"][0]):
        out.append({
            "id": id_,
            "source": meta["source"],
            "title": meta["title"],
            "section": meta["section"],
            "text": doc,
            "distance": round(float(dist), 4),
        })
    return out
