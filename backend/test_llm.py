"""Time one Gemini call.  Usage:  python test_llm.py [model-name]"""
import sys
import time

from google import genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL

model = sys.argv[1] if len(sys.argv) > 1 else GEMINI_MODEL
c = genai.Client(api_key=GEMINI_API_KEY)
prompt = (
    "You are a support assistant. Write a polite 3-sentence reply to a customer who says: "
    "'There is a collection account on my credit report that I have never heard of.' "
    "Mention that they can ask the collector for written validation of the debt."
)
print("model:", model)
t = time.time()
try:
    r = c.models.generate_content(model=model, contents=prompt)
    print(f"OK in {time.time() - t:.1f}s:\n", r.text)
except Exception as e:
    print(f"FAILED after {time.time() - t:.1f}s: {type(e).__name__}: {str(e)[:300]}")
