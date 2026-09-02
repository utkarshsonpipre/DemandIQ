import InfoTooltip from "./InfoTooltip";
import { METRIC_LABEL } from "../metricInfo";

/** A stat tile: label (+ optional info tooltip), big value, optional sub-line. */
export default function MetricCard({ metricKey, label, value, sub, tooltip }) {
  return (
    <div className="card metric-card">
      <span className="metric-label">
        {label || METRIC_LABEL[metricKey] || metricKey}
        <InfoTooltip metricKey={metricKey} text={tooltip} />
      </span>
      <span className="metric-value">{value ?? "—"}</span>
      {sub && <span className="metric-sub">{sub}</span>}
    </div>
  );
}
