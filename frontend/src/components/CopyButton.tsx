import { useState } from "react";

// Copies the given text to the clipboard with brief "Copied" feedback.
export function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard may be unavailable (e.g. non-secure context); ignore
    }
  }

  return (
    <button type="button" className="copy-btn" onClick={copy} title="Copy answer">
      {copied ? "Copied ✓" : "Copy"}
    </button>
  );
}
