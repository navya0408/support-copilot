"""Retrieval evaluation.  Run from the backend/ folder:   python -m eval.run_eval [eval_file.json]

Measures:
  hit@1 / hit@3      correct document ranked first / in the top 3 (in-scope questions)
  answered rate      in-scope questions that pass the distance threshold
  refusal rate       out-of-scope questions correctly refused by the threshold

Use the printed distances to choose MAX_DISTANCE in .env: it should sit between the
in-scope distances (want it above them) and the out-of-scope distances (want it below them).
"""
import json
import sys
from pathlib import Path
from statistics import mean

from app import rag
from app.config import MAX_DISTANCE

HERE = Path(__file__).parent


def main():
    rag.ensure_index()
    eval_name = sys.argv[1] if len(sys.argv) > 1 else "eval_set.json"
    items = json.loads((HERE / eval_name).read_text(encoding="utf-8"))
    rows, d_in, d_out = [], [], []
    hit1 = hit3 = answered = n_in = refused = n_out = 0

    for it in items:
        res = rag.retrieve(it["question"], k=3)
        best = res[0]["distance"]
        sources = [r["source"] for r in res]
        exp = it["expected"]
        if exp is None:
            n_out += 1
            d_out.append(best)
            ok = best > MAX_DISTANCE
            refused += ok
            tag = "REFUSED " if ok else "WRONGLY ANSWERED"
        else:
            n_in += 1
            d_in.append(best)
            h1, h3 = sources[0] in exp, any(s in exp for s in sources)
            hit1 += h1
            hit3 += h3
            answered += best <= MAX_DISTANCE
            tag = "hit@1" if h1 else ("hit@3" if h3 else "MISS")
        rows.append({"question": it["question"], "expected": exp, "retrieved": sources,
                     "best_distance": best, "result": tag})
        print(f"{tag:<17} d={best:.3f}  {it['question'][:70]}")
        if tag == "MISS":
            print(f"{'':<17} expected={exp}  got={sources}")

    summary = {
        "n_in_scope": n_in, "n_out_of_scope": n_out, "max_distance": MAX_DISTANCE,
        "hit_at_1": round(hit1 / n_in, 3), "hit_at_3": round(hit3 / n_in, 3),
        "in_scope_answered_rate": round(answered / n_in, 3),
        "out_of_scope_refusal_rate": round(refused / n_out, 3) if n_out else None,
        "mean_distance_in_scope": round(mean(d_in), 3),
        "mean_distance_out_of_scope": round(mean(d_out), 3) if d_out else None,
        "max_in_scope_distance": round(max(d_in), 3),
        "min_out_of_scope_distance": round(min(d_out), 3) if d_out else None,
    }
    print("\n" + json.dumps(summary, indent=2))
    out = HERE / f"results_{Path(eval_name).stem}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"saved {out.name}")


if __name__ == "__main__":
    main()
