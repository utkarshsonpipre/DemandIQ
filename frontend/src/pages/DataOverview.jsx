import { useApi } from "../useApi";
import MetricCard from "../components/MetricCard";
import InfoTooltip from "../components/InfoTooltip";
import { Loading, ErrorBox, Empty } from "../components/Loading";

export default function DataOverview() {
  const { data, error, loading } = useApi("/data-overview");

  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;

  const report = data.validation_report;
  const missingEntries = Object.entries(data.missing_pct || {});

  return (
    <div>
      <h1 className="page-title">Data Overview</h1>
      <p className="page-subtitle">Dataset statistics, missing values, and the automated validation report</p>

      <div className="section grid grid-4">
        <MetricCard label="Rows" value={data.n_rows.toLocaleString()} />
        <MetricCard label="Series" value={data.n_series} />
        <MetricCard label="Date range" value={`${data.date_min} → ${data.date_max}`} />
        <MetricCard metricKey="quality_score" value={report ? `${report.quality_score}/100` : "—"} />
      </div>

      <div className="section grid grid-2">
        <div className="card">
          <div className="section-title" style={{ marginBottom: 12 }}>Missing values by column</div>
          {missingEntries.length === 0 ? (
            <Empty>No missing values after cleaning.</Empty>
          ) : (
            <table>
              <thead><tr><th>Column</th><th>Missing %</th></tr></thead>
              <tbody>{missingEntries.map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}%</td></tr>)}</tbody>
            </table>
          )}
        </div>

        <div className="card">
          <div className="section-title" style={{ marginBottom: 12 }}>Validation report</div>
          {!report ? <Empty /> : (
            <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 13 }}>
              <Row label="Schema" value={report.schema_errors.length ? report.schema_errors.join("; ") : "Valid ✓"} />
              <Row label="Duplicate rows" value={report.duplicate_rows} />
              <Row label="Date gaps" value={report.date_gaps} />
              <Row label="Negative sales" value={report.negative_sales} />
              <Row label={<>Outliers <InfoTooltip metricKey="outlier_pct" /></>}
                   value={`${report.outlier_count} rows (${report.outlier_pct}%)`} />
            </div>
          )}
        </div>
      </div>

      <div className="section">
        <div className="section-title">Sample</div>
        <div className="card table-wrap">
          <table>
            <thead>
              <tr>{data.sample[0] && Object.keys(data.sample[0]).map((k) => <th key={k}>{k}</th>)}</tr>
            </thead>
            <tbody>
              {data.sample.slice(0, 20).map((row, i) => (
                <tr key={i}>{Object.values(row).map((v, j) => <td key={j}>{String(v)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
      <span style={{ color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 4 }}>{label}</span>
      <span>{value}</span>
    </div>
  );
}
