// Thin client over the backend SSE endpoint. The UI only transports and
// renders; all retrieval and prompt logic lives in the backend.
import type { Source } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8001";

export type StreamEvent =
  | { event: "token"; data: { text: string } }
  | { event: "sources"; data: { sources: Source[]; mode: string } }
  | { event: "error"; data: { detail: string } };

function parseFrame(frame: string): StreamEvent | null {
  let event = "";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data += line.slice(5).trim();
  }
  if (!event) return null;
  try {
    return { event, data: JSON.parse(data) } as StreamEvent;
  } catch {
    return null;
  }
}

// Async-iterates the answer stream: token events, then a final sources event.
export async function* streamQuery(
  question: string,
  signal: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const url = `${API_BASE_URL}/query/stream?question=${encodeURIComponent(question)}`;
  const resp = await fetch(url, {
    signal,
    headers: { Accept: "text/event-stream" },
  });
  if (!resp.ok || !resp.body) {
    throw new Error(`API request failed (HTTP ${resp.status})`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let split: number;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      const parsed = parseFrame(frame);
      if (parsed) yield parsed;
    }
  }
}
