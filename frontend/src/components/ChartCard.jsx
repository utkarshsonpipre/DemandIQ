import InfoTooltip from "./InfoTooltip";

/** Card wrapper for a chart, with a title and optional info tooltip. */
export default function ChartCard({ title, tooltip, children, height = 320 }) {
  return (
    <div className="card">
      {title && (
        <div className="section-title" style={{ marginBottom: 14 }}>
          {title}
          {tooltip && <InfoTooltip text={tooltip} />}
        </div>
      )}
      <div style={{ width: "100%", height }}>{children}</div>
    </div>
  );
}
