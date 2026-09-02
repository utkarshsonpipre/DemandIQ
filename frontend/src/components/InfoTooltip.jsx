import { useState } from "react";
import { METRIC_INFO } from "../metricInfo";

/** Small (i) button — hover or focus reveals a one-line plain-language
 * explanation. `text` overrides the METRIC_INFO lookup for `metricKey`. */
export default function InfoTooltip({ metricKey, text }) {
  const [open, setOpen] = useState(false);
  const content = text || METRIC_INFO[metricKey];
  if (!content) return null;

  return (
    <span style={{ position: "relative", display: "inline-flex" }}>
      <button
        type="button"
        className="info-icon"
        aria-label="What is this metric?"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
      >
        i
      </button>
      {open && <span className="info-tooltip" role="tooltip">{content}</span>}
    </span>
  );
}
