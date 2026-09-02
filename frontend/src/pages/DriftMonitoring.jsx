import { useState } from "react";
import { api } from "../api";
import { useApi } from "../useApi";
import StatusBadge from "../components/StatusBadge";
import InfoTooltip from "../components/InfoTooltip";
import { Loading, ErrorBox, Empty } from "../components/Loading";

export default function DriftMonitoring() {
  const { data: drift, error, loading, reload } = useApi("/drift");
  const { data: log } = useApi("/retrain-log");
  const [perf, setPerf] = useState(null);
  const [checking, setChecking] = useState(false);
  const [showReport, setShowReport] = useState(false);

  const runPerfCheck = async () => {
    setChecking(true);
    try {
      const { data } = await api.get("/performance-check");
      setPerf(data);
    } catch (e) {
      setPerf({ status: "ERROR", message: e.response?.data?.detail || e.message });
    } finally {
      setChecking(false);
    }
  };

  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;
  if (!drift?.status) return <Empty>No drift report yet — run a drift check from the Overview page.</Empty>;

  return (
    <div>
      <h1 className="page-title">Drift Monitoring</h1>
      <p className="page-subtitle">Is today's data still shaped like the data the model was trained on?</p>

      <div className="section card">
        <div style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 15, fontWeight: 650, marginBottom: 4 }}>
          Overall drift status <StatusBadge status={drift.status} /> <InfoTooltip metricKey="drift_status" />
        </div>
        <p className="metric-sub" style={{ margin: 0 }}>
          {drift.recommendation} · checked {drift.checked_at?.slice(0, 19).replace("T", " ")} · {drift.n_drifted} feature(s) drifted
        </p>
      </div>

      <div className="section">
        <div className="section-title">Feature drift (KS test)</div>
        <div className="card table-wrap">
          <table>
            <thead>
              <tr>
                <th>Feature</th><th>Reference mean</th><th>Current mean</th>
                <th>KS statistic <InfoTooltip metricKey="ks_statistic" /></th>
                <th>p-value <InfoTooltip metricKey="p_value" /></th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {drift.features.map((f) => (
                <tr key={f.feature}>
                  <td>{f.feature}</td><td>{f.reference_mean}</td><td>{f.current_mean}</td>
                  <td>{f.ks_statistic}</td><td>{f.p_value}</td>
                  <td>{f.drift ? <span className="badge badge-warning">⚠ drift</span> : <span className="badge badge-good">✓ stable</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="section">
        <button className="btn" onClick={() => setShowReport((s) => !s)}>
          {showReport ? "Hide" : "Show"} Evidently full report
        </button>
        {showReport && (
          <div className="card" style={{ marginTop: 12, padding: 0, overflow: "hidden" }}>
            <iframe title="evidently" src="http://localhost:8000/api/drift/html" style={{ width: "100%", height: 600, border: "none" }} />
          </div>
        )}
      </div>

      <div className="section">
        <div className="section-title">Model performance on recent data</div>
        <div className="card">
          <button className="btn" onClick={runPerfCheck} disabled={checking}>
            {checking && <span className="spinner spinner-dark" />} Run performance check
          </button>
          {perf && (
            <div style={{ marginTop: 14, fontSize: 13 }}>
              {perf.status === "DEGRADED" && (
                <span className="badge badge-critical">
                  ⚠ MAPE degraded {perf.degradation_pct}%: recent {perf.recent_mape?.toFixed(2)}% vs baseline {perf.baseline_mape?.toFixed(2)}%
                </span>
              )}
              {perf.status === "OK" && (
                <span className="badge badge-good">
                  ✓ OK — recent MAPE {perf.recent_mape?.toFixed(2)}% vs baseline {perf.baseline_mape?.toFixed(2)}% ({perf.degradation_pct > 0 ? "+" : ""}{perf.degradation_pct}%)
                </span>
              )}
              {perf.status !== "OK" && perf.status !== "DEGRADED" && <span className="badge badge-neutral">{perf.status}</span>}
            </div>
          )}
        </div>
      </div>

      <div className="section">
        <div className="section-title">Retraining history</div>
        <div className="card table-wrap">
          {!log?.length ? <Empty>No retraining events yet.</Empty> : (
            <table>
              <thead><tr><th>Timestamp</th><th>Trigger</th><th>Best model</th><th>Test MAPE</th></tr></thead>
              <tbody>
                {[...log].reverse().map((r, i) => (
                  <tr key={i}><td>{r.timestamp?.slice(0, 19).replace("T", " ")}</td><td>{r.trigger}</td><td>{r.best_model}</td><td>{r.test_mape?.toFixed(2)}%</td></tr>
                ))}
              </tbody>
            </table>
          )}
          {log?.length > 0 && <p className="metric-sub">Last retrained: {log[log.length - 1].timestamp?.slice(0, 19).replace("T", " ")}</p>}
        </div>
      </div>
    </div>
  );
}
