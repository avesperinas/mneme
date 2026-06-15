import { useState } from "react";
import type { Source } from "../types";

// A clickable citation card: click to expand the snippet. Shows the source's
// rel_path and heading_path breadcrumb so it points at the exact fragment.
export function SourceCard({ source }: { source: Source }) {
  const [open, setOpen] = useState(false);
  const breadcrumb = source.heading_path.join(" › ");
  return (
    <button
      type="button"
      className={`source-card${open ? " open" : ""}`}
      onClick={() => setOpen((v) => !v)}
      title="Click to expand the cited fragment"
    >
      <div className="source-head">
        <span className="source-path">{source.rel_path}</span>
        <span className="source-score">{source.score.toFixed(2)}</span>
      </div>
      {breadcrumb && <div className="source-breadcrumb">{breadcrumb}</div>}
      {open && <p className="source-snippet">{source.snippet}</p>}
    </button>
  );
}
