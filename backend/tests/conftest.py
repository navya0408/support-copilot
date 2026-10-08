"""Lets the pure-Python tests run from the backend/ folder.

If chromadb is not installed (for example in a bare CI box) a tiny stub is used so that the
chunking tests can still import app.rag. Real retrieval is never exercised by these tests.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import chromadb  # noqa: F401
except ImportError:
    cd = types.ModuleType("chromadb")
    cu = types.ModuleType("chromadb.utils")
    ef = types.ModuleType("chromadb.utils.embedding_functions")
    cd.utils, cu.embedding_functions = cu, ef
    sys.modules.update({"chromadb": cd, "chromadb.utils": cu, "chromadb.utils.embedding_functions": ef})
