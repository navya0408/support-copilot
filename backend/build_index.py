"""Build (or rebuild) the vector index from ../knowledge_base/*.md

Run from the backend/ folder:   python build_index.py [--force]
"""
import sys

from app import rag

if __name__ == "__main__":
    n = rag.ensure_index(force="--force" in sys.argv)
    print(f"Indexed {n} chunks from {rag.KB_DIR}")
