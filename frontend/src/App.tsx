import { useRef, useState } from "react";
import { streamQuery } from "./api";
import { SourceCard } from "./components/SourceCard";
import type { ChatMessage } from "./types";

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  function patch(id: string, change: Partial<ChatMessage>) {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, ...change } : m)),
    );
  }

  async function ask() {
    const question = input.trim();
    if (!question || busy) return;
    const id = crypto.randomUUID();
    setMessages((prev) => [
      ...prev,
      { id, question, answer: "", sources: [], streaming: true },
    ]);
    setInput("");
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      for await (const ev of streamQuery(question, controller.signal)) {
        if (ev.event === "token") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === id ? { ...m, answer: m.answer + ev.data.text } : m,
            ),
          );
        } else if (ev.event === "sources") {
          patch(id, { sources: ev.data.sources });
        } else if (ev.event === "error") {
          patch(id, { error: ev.data.detail });
        }
      }
    } catch (err) {
      patch(id, { error: err instanceof Error ? err.message : String(err) });
    } finally {
      patch(id, { streaming: false });
      setBusy(false);
      abortRef.current = null;
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Mneme</h1>
        <p className="tagline">Answers grounded in your notes.</p>
      </header>

      <main className="thread">
        {messages.length === 0 && (
          <p className="empty">Ask a question about your vault.</p>
        )}
        {messages.map((m) => (
          <article key={m.id} className="turn">
            <div className="question">{m.question}</div>
            <div className="answer">
              {m.answer}
              {m.streaming && <span className="cursor">▍</span>}
            </div>
            {m.error && <div className="error">{m.error}</div>}
            {m.sources.length > 0 && (
              <div className="sources">
                {m.sources.map((s, i) => (
                  <SourceCard key={`${m.id}-${i}`} source={s} />
                ))}
              </div>
            )}
          </article>
        ))}
      </main>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          void ask();
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask your notes…"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !input.trim()}>
          {busy ? "…" : "Ask"}
        </button>
      </form>
    </div>
  );
}
