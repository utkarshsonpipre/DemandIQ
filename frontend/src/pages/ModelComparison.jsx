import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LabelList } from "recharts";
import { useApi } from "../useApi";
import ChartCard from "../components/ChartCard";
import InfoTooltip from "../components/InfoTooltip";
import { Loading, ErrorBox } from "../components/Loading";
import { METRIC_INFO, METRIC_LABEL } from "../metricInfo";
import { seriesColors } from "../colors";

const METRIC_COLS = ["mape", "rmse", "mae", "r2_score", "smape", "training_time_seconds", "inference_time_ms"];

export default function ModelComparison() {
  const { data, error, loading } = useApi("/model-comparison");
  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;

  const avg = data.average;
  const colors = seriesColors(avg.map((r) => r.model));

  const bar = (metricKey, title) => (
    <ChartCard title={title} tooltip={METRIC_INFO[metricKey]} height={340}>
      <ResponsiveContainer>
        <BarChart data={avg} margin={{ top: 20 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="model" tick={{ fontSize: 11 }} />
          <YAxis tick={{ fontSize: 11 }} width={45} />
          <Tooltip formatter={(v) => v.toFixed(3)} />
          <Bar dataKey={metricKey} radius={[4, 4, 0, 0]}>
            <LabelList dataKey={metricKey} position="top" formatter={(v) => v.toFixed(2)} style={{ fontSize: 11, fill: "var(--text-secondary)" }} />
            {avg.map((r) => <Cell key={r.model} fill={colors[r.model]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );

  return (
    <div>
      <h1 className="page-title">Model Comparison</h1>
      <p className="page-subtitle">Every model trained on identical walk-forward CV folds — lowest MAPE wins</p>

      <div className="section">
        <div className="section-title">Average metrics across CV folds</div>
        <div className="card table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                {METRIC_COLS.map((c) => (
                  <th key={c}>{METRIC_LABEL[c] || c} <InfoTooltip metricKey={c} /></th>
                ))}
                <th>Rank</th>
              </tr>
            </thead>
            <tbody>
              {avg.map((r) => (
                <tr key={r.model}>
                  <td><span className="legend-swatch" style={{ background: colors[r.model], display: "inline-block", marginRight: 6 }} />{r.model}</td>
                  {METRIC_COLS.map((c) => <td key={c}>{r[c]?.toFixed(3)}</td>)}
                  <td>{r.rank}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="section grid grid-2">
        {bar("mape", "MAPE (%) — lower is better")}
        {bar("rmse", "RMSE — lower is better")}
      </div>
      <div className="section grid grid-2">
        {bar("training_time_seconds", "Training time (s)")}
        {bar("inference_time_ms", "Inference time per row (ms)")}
      </div>
    </div>
  );
}
