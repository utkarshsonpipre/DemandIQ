import { useRef, useState } from "react";
import { api } from "../api";
import { useApi } from "../useApi";
import MetricCard from "../components/MetricCard";
import StatusBadge from "../components/StatusBadge";
import { Loading, ErrorBox, Empty } from "../components/Loading";

export default function Overview() {
  const { data: status, error, loading, reload: reloadStatus } = useApi("/status");
  const { data: log, reload: reloadLog } = useApi("/retrain-log");
  const [retraining, setRetraining] = useState(false);
  const [checking, setChecking] = useState(false);
  const [toast, setToast] = useState(null);
  const fileRef = useRef(null);

  const showToast = (msg) => { setToast(msg); setTimeout(() => setToast(null), 6000); };

  const retrain = async () => {
    setRetraining(true);
    try {
      const { data } = await api.post("/retrain");
      showToast(`Retrained: ${data.best_model} · test MAPE ${data.test_mape.toFixed(2)}% · ${data.duration_seconds}s`);
      reloadStatus(); reloadLog();
    } catch (e) {
      showToast(`Retraining failed: ${e.response?.data?.detail || e.message}`);
    } finally {
      setRetraining(false);
    }
  };

  const checkDrift = async () => {
    setChecking(true);
    try {
      const { data } = await api.post("/drift-check");
      showToast(`Drift status: ${data.status} — ${data.recommendation}`);
      reloadStatus();
    } catch (e) {
      showToast(`Drift check failed: ${e.response?.data?.detail || e.message}`);
    } finally {
      setChecking(false);
    }
  };

  const upload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    try {
      const { data } = await api.post("/upload", form);
      showToast(data.message);
    } catch (e2) {
      showToast(`Upload failed: ${e2.response?.data?.detail || e2.message}`);
    }
  };

  const downloadForecast = () => window.open("http://localhost:8000/api/forecast", "_blank");

  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;

  return (
    <div>
      <h1 className="page-title">DemandIQ — Demand Forecasting Platform</h1>
      <p className="page-subtitle">Automated forecasting, experiment tracking, drift monitoring, retraining</p>

      <div className="section grid grid-4">
        <MetricCard metricKey="production_model" label="Production model" value={status.best_model || "—"} />
        <MetricCard metricKey="test_mape" value={status.test_mape != null ? `${status.test_mape.toFixed(2)}%` : "—"} />
        <div className="card metric-card">
          <span className="metric-label">Drift status</span>
          <span className="metric-value" style={{ fontSize: 20 }}><StatusBadge status={status.drift_status} /></span>
        </div>
        <MetricCard metricKey="series_tracked" label="Series tracked" value={status.n_series ?? "—"} />
      </div>

      {status.trained_at && (
        <div className="section card" style={{ fontSize: 12.5, color: "var(--text-secondary)" }}>
          Last trained: {status.trained_at.slice(0, 19).replace("T", " ")} · {status.n_features} features
          · avg CV MAPE {status.avg_cv_mape?.toFixed(2)}%
        </div>
      )}

      <div className="section">
        <div className="section-title">Actions & Controls</div>
        <div className="grid grid-3">
          <div className="card">
            <strong style={{ fontSize: 13 }}>Retrain model</strong>
            <p className="metric-sub" style={{ margin: "6px 0 12px" }}>Re-run the full pipeline on all available data.</p>
            <button className="btn btn-primary" onClick={retrain} disabled={retraining}>
              {retraining && <span className="spinner" />} 🔁 {retraining ? "Retraining…" : "Retrain now"}
            </button>
          </div>
          <div className="card">
            <strong style={{ fontSize: 13 }}>Run drift check</strong>
            <p className="metric-sub" style={{ margin: "6px 0 12px" }}>Compare recent data against the training baseline.</p>
            <button className="btn" onClick={checkDrift} disabled={checking}>
              {checking && <span className="spinner spinner-dark" />} 🔍 {checking ? "Checking…" : "Check drift"}
            </button>
          </div>
          <div className="card">
            <strong style={{ fontSize: 13 }}>Download forecast</strong>
            <p className="metric-sub" style={{ margin: "6px 0 12px" }}>Export the latest 30-day forecast as CSV.</p>
            <button className="btn" onClick={downloadForecast}>⬇️ forecast.csv</button>
          </div>
        </div>
      </div>

      <div className="section">
        <div className="section-title">Upload data</div>
        <div className="card">
          <p className="metric-sub" style={{ marginTop: 0 }}>
            Long-format CSV with columns: date, sales (+ optional series/store/item ids).
          </p>
          <input ref={fileRef} type="file" accept=".csv" onChange={upload} />
        </div>
      </div>

      <div className="section">
        <div className="section-title">Retraining history</div>
        <div className="card table-wrap">
          {!log || log.length === 0 ? <Empty>No retraining events yet.</Empty> : (
            <table>
              <thead><tr><th>Timestamp</th><th>Trigger</th><th>Best model</th><th>Avg CV MAPE</th><th>Test MAPE</th><th>Duration (s)</th></tr></thead>
              <tbody>
                {[...log].reverse().map((r, i) => (
                  <tr key={i}>
                    <td>{r.timestamp?.slice(0, 19).replace("T", " ")}</td>
                    <td>{r.trigger}</td>
                    <td>{r.best_model}</td>
                    <td>{r.avg_cv_mape?.toFixed(2)}%</td>
                    <td>{r.test_mape?.toFixed(2)}%</td>
                    <td>{r.duration_seconds}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
