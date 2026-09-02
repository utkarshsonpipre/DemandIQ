import { useApi } from "../useApi";
import MetricCard from "../components/MetricCard";
import StatusBadge from "../components/StatusBadge";
import { Loading, ErrorBox, Empty } from "../components/Loading";

export default function MlflowTracking() {
  const { data, error, loading } = useApi("/mlflow");
  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;

  const best = data.best_run;

  return (
    <div>
      <h1 className="page-title">MLflow Experiment Tracking</h1>
      <p className="page-subtitle">Every training run's parameters and metrics, plus the model registry's version history</p>

      {best && (
        <>
          <div className="section-title">Current best run</div>
          <div className="section grid grid-3">
            <MetricCard label="Model" value={best.best_model} />
            <MetricCard metricKey="avg_cv_mape" value={`${best.avg_cv_mape.toFixed(2)}%`} />
            <MetricCard metricKey="test_mape" value={`${best.test_metrics.mape.toFixed(2)}%`} />
          </div>
          {best.registry?.version && (
            <div className="card" style={{ marginBottom: 24, fontSize: 13 }}>
              Registered as version {best.registry.version} — stage: <StatusBadge status={best.registry.stage} />
            </div>
          )}
        </>
      )}

      <div className="section">
        <div className="section-title">Model registry versions</div>
        <div className="card table-wrap">
          {!data.versions?.length ? <Empty>No registered versions yet.</Empty> : (
            <table>
              <thead><tr><th>Version</th><th>Model type</th><th>Test MAPE</th><th>Stage</th><th>Created</th></tr></thead>
              <tbody>
                {[...data.versions].sort((a, b) => b.version - a.version).map((v) => (
                  <tr key={v.version}>
                    <td>{v.version}</td><td>{v.model_type}</td><td>{v.test_mape}%</td>
                    <td><StatusBadge status={v.stage} /></td>
                    <td>{new Date(v.created).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <div className="section">
        <div className="section-title">Experiment runs</div>
        <div className="card table-wrap">
          {!data.runs?.length ? <Empty>No runs found — run the pipeline first.</Empty> : (
            <table>
              <thead>
                <tr>{Object.keys(data.runs[0]).map((k) => <th key={k}>{k}</th>)}</tr>
              </thead>
              <tbody>
                {data.runs.map((r, i) => (
                  <tr key={i}>{Object.values(r).map((v, j) => (
                    <td key={j}>{typeof v === "number" ? v.toFixed(3) : (v ?? "—")}</td>
                  ))}</tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      <p className="metric-sub">Full UI: <code>mlflow ui --backend-store-uri sqlite:///mlflow.db</code> → http://localhost:5000</p>
    </div>
  );
}
