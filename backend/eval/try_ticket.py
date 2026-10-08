"""Type any ticket and see what the system would do (NO Gemini call, so it uses no quota).

Run from the backend/ folder:
    python -m eval.try_ticket                      (interactive: type tickets, empty line to quit)
    python -m eval.try_ticket "my card was charged twice"
"""
import sys

from app import classify, rag, urgency
from app.config import LOW_CLASSIFIER_CONFIDENCE, MAX_DISTANCE, TOP_K


def show(text: str):
    label, conf, top3 = classify.predict(text)
    level, reasons = urgency.assess(text)
    chunks = rag.retrieve(text, TOP_K)
    best = chunks[0]["distance"]

    print(f"\n  category : {label}  ({conf:.0%})" + ("   <- LOW CONFIDENCE" if conf < LOW_CLASSIFIER_CONFIDENCE else ""))
    print("  top 3    : " + ", ".join(f"{t['label']} {t['probability']:.0%}" for t in top3))
    print(f"  urgency  : {level.upper()}" + (f"  ({'; '.join(reasons)})" if reasons else ""))
    verdict = "ANSWER (would call the LLM)" if best <= MAX_DISTANCE else "REFUSE -> escalate to a human"
    print(f"  retrieval: best distance {best:.3f} vs limit {MAX_DISTANCE}  =>  {verdict}")
    for i, c in enumerate(chunks, 1):
        print(f"    [{i}] d={c['distance']:.3f}  {c['title']} - {c['section']}")


def main():
    rag.ensure_index()
    if len(sys.argv) > 1:
        show(" ".join(sys.argv[1:]))
        return
    print("Type a ticket and press Enter. Empty line to quit.")
    while True:
        text = input("\nticket> ").strip()
        if not text:
            break
        show(text)


if __name__ == "__main__":
    main()
