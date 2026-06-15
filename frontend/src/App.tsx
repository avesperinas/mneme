import { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { streamQuery } from "./api";
import { CopyButton } from "./components/CopyButton";
import { SourceCard } from "./components/SourceCard";
import type { ChatMessage } from "./types";

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  function patch(id: string, change: Partial<ChatMessage>) {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, ...change } : m)),
    );
  }

  async function ask() {
    const question = input.trim();
    if (!question) return;

    // Claim ownership before aborting any in-flight stream, so the superseded
    // request's cleanup does not clobber this one's state.
    const controller = new AbortController();
    const previous = abortRef.current;
    abortRef.current = controller;
    if (previous) previous.abort();

    const id = crypto.randomUUID();
    setMessages((prev) => [
      ...prev,
      { id, question, answer: "", sources: [], streaming: true },
    ]);
    setInput("");

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
      // A stream aborted because a newer question superseded it is not an error.
      if (!controller.signal.aborted) {
        patch(id, { error: err instanceof Error ? err.message : String(err) });
      }
    } finally {
      patch(id, { streaming: false });
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Mneme</h1>
        <p className="tagline">Answers grounded in your notes.</p>
      </header>

      <main className="thread">
        {messages.length === 0 && (
          <p className="empty">Ask a question about your vault.</p>
        )}
        {messages.map((m) => (
          <article key={m.id} className="turn">
            <div className="bubble user">{m.question}</div>

            <div className="bubble assistant">
              <div className="assistant-head">
                <span className="role">Mneme</span>
                {!m.streaming && m.answer && <CopyButton text={m.answer} />}
              </div>

              <div className="markdown">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {m.answer}
                </ReactMarkdown>
                {m.streaming && <span className="cursor">▍</span>}
              </div>

              {m.error && <div className="error">{m.error}</div>}

              {m.sources.length > 0 && (
                <div className="sources">
                  <div className="sources-label">Sources</div>
                  <div className="sources-grid">
                    {m.sources.map((s, i) => (
                      <SourceCard key={`${m.id}-${i}`} source={s} />
                    ))}
                  </div>
                </div>
              )}
            </div>
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
        />
        <button type="submit" disabled={!input.trim()}>
          Ask
        </button>
      </form>
    </div>
  );
}
