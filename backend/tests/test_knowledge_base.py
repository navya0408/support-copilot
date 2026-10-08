import json
from pathlib import Path

from app import rag

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"


def test_chunks_load_and_have_text():
    chunks = rag.load_chunks()
    assert len(chunks) >= 18
    assert all(c["text"].strip() and c["title"] for c in chunks)
    assert len({c["id"] for c in chunks}) == len(chunks)       # ids are unique


def test_readme_is_not_indexed():
    assert not any(c["source"].lower() == "readme" for c in rag.load_chunks())


def test_eval_questions_point_to_real_documents():
    stems = {c["source"] for c in rag.load_chunks()}
    for name in ("eval_set.json", "eval_holdout.json"):
        for item in json.loads((EVAL_DIR / name).read_text(encoding="utf-8")):
            if item["expected"] is not None:
                assert set(item["expected"]) <= stems, item["question"]
