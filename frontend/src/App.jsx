import { useState } from "react";

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

const SAMPLES = [
  "There is a collection account on my credit report that I have never heard of. I want it removed and nobody has answered my letters.",
  "I was charged twice for the same purchase on my credit card and the bank will not fix it.",
  "My mortgage servicer applied my payment to the wrong month and now I am getting late notices. I am afraid of foreclosure.",
  "What is the weather like today?",
];

const LABELS = {
  CREDIT_REPORTING: "Credit reporting",
  DEBT_COLLECTION: "Debt collection",
  BANK_ACCOUNT: "Bank account",
  CREDIT_CARD: "Credit card",
  LOANS: "Loans",
  MONEY_TRANSFER_PREPAID: "Money transfer / prepaid",
  MORTGAGE: "Mortgage",
};

export default function App() {
  const [ticket, setTicket] = useState("");
  const [result, setResult] = useState(null);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [rated, setRated] = useState(null);
  const [copied, setCopied] = useState(false);

  async function analyze() {
    setLoading(true);
    setError("");
    setResult(null);
    setRated(null);
    try {
      const res = await fetch(`${API}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: ticket }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${res.status})`);
      }
      const data = await res.json();
      setResult(data);
      setDraft(data.draft || "");
    } catch (e) {
      setError(e.message || "Could not reach the API. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  async function sendFeedback(rating) {
    setRated(rating);
    try {
      await fetch(`${API}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticket, category: result.category, draft, rating }),
      });
    } catch {
      /* feedback is best-effort */
    }
  }

  async function copyDraft() {
    try {
      await navigator.clipboard.writeText(draft);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard may be blocked */
    }
  }

  const tooShort = ticket.trim().length < 5;

  return (
    <div className="page">
      <header>
        <h1>Support Copilot</h1>
        <p className="sub">
          Classifies a financial complaint, finds the relevant guidance, and drafts a reply for a human agent to review.
        </p>
      </header>

      <section className="card">
        <label htmlFor="ticket">Customer ticket</label>
        <textarea
          id="ticket"
          rows={6}
          value={ticket}
          onChange={(e) => setTicket(e.target.value)}
          placeholder="Paste a customer complaint here..."
        />
        <div className="samples">
          <span>Try a sample:</span>
          {SAMPLES.map((s, i) => (
            <button key={i} className="chip" onClick={() => setTicket(s)}>
              Sample {i + 1}
            </button>
          ))}
        </div>
        <button className="primary" onClick={analyze} disabled={loading || tooShort}>
          {loading ? "Analyzing..." : "Analyze ticket"}
        </button>
        {error && <div className="banner error">{error}</div>}
      </section>

      {result && (
        <>
          <section className="grid">
            <div className="card">
              <h2>Category</h2>
              <div className="big">{LABELS[result.category] || result.category}</div>
              <div className="bar">
                <div style={{ width: `${Math.round(result.category_confidence * 100)}%` }} />
              </div>
              <div className="muted">Confidence {Math.round(result.category_confidence * 100)}%</div>
              {result.category_low_confidence && (
                <div className="banner warn">Low confidence: please check the category manually.</div>
              )}
              <ul className="muted small">
                {result.category_top3.slice(1).map((t) => (
                  <li key={t.label}>
                    {LABELS[t.label] || t.label}: {Math.round(t.probability * 100)}%
                  </li>
                ))}
              </ul>
            </div>

            <div className="card">
              <h2>Urgency</h2>
              <span className={`badge ${result.urgency}`}>{result.urgency.toUpperCase()}</span>
              {result.urgency_reasons.length > 0 ? (
                <ul className="muted small">
                  {result.urgency_reasons.map((r) => (
                    <li key={r}>{r}</li>
                  ))}
                </ul>
              ) : (
                <p className="muted small">No urgency signals found (rule-based).</p>
              )}
              <div className="muted small">Processed in {result.latency_ms} ms</div>
            </div>
          </section>

          <section className="card">
            {result.needs_human ? (
              <>
                <h2>Escalate to a human agent</h2>
                <div className="banner warn">{result.message}</div>
                <div className="row">
                  {rated ? (
                    <span className="muted">Thanks for the feedback</span>
                  ) : (
                    <>
                      <span className="muted">Was escalating the right call?</span>
                      <button onClick={() => sendFeedback("up")} aria-label="Yes">👍</button>
                      <button onClick={() => sendFeedback("down")} aria-label="No">👎</button>
                    </>
                  )}
                </div>
              </>
            ) : (
              <>
                <h2>Draft reply</h2>
                {result.message && <div className="banner info">{result.message}</div>}
                <textarea rows={8} value={draft} onChange={(e) => setDraft(e.target.value)} />
                <div className="row">
                  <button onClick={copyDraft}>{copied ? "Copied" : "Copy"}</button>
                  <span className="spacer" />
                  {rated ? (
                    <span className="muted">Thanks for the feedback ({rated === "up" ? "helpful" : "not helpful"})</span>
                  ) : (
                    <>
                      <span className="muted">Was this draft useful?</span>
                      <button onClick={() => sendFeedback("up")} aria-label="Helpful">👍</button>
                      <button onClick={() => sendFeedback("down")} aria-label="Not helpful">👎</button>
                    </>
                  )}
                </div>
              </>
            )}
          </section>

          <section className="card">
            <h2>{result.needs_human ? "Closest passages (not used)" : "Sources used"}</h2>
            {result.sources.map((s, i) => (
              <details key={s.id} open={i === 0}>
                <summary>
                  [{i + 1}] {s.title} - {s.section}
                  <span className="muted"> (similarity {s.similarity})</span>
                </summary>
                <p className="source">{s.text}</p>
              </details>
            ))}
          </section>
        </>
      )}

      <footer>General guidance for learning purposes, not legal advice. A human agent must review every draft.</footer>
    </div>
  );
}
