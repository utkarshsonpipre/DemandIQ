import { useState } from "react";
import {
  ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Legend,
} from "recharts";
import { useApi } from "../useApi";
import ChartCard from "../components/ChartCard";
import MetricCard from "../components/MetricCard";
import { Loading, ErrorBox, Empty } from "../components/Loading";

export default function Forecasting() {
  const { data: fc, error, loading } = useApi("/forecast");
  const [sid, setSid] = useState(null);
  const active = sid || fc?.series?.[0];
  const seriesFetch = useApi(active && fc && active !== fc.series[0] ? `/forecast/${active}` : null, [active]);

  if (loading) return <Loading />;
  if (error) return <ErrorBox error={error} />;

  const view = active === fc.series[0] ? { history: fc.default_series.history, forecast: fc.default_series.forecast } : seriesFetch.data;

  const merged = view ? [
    ...view.history.map((h) => ({ date: h.date, actual: h.sales })),
    ...view.forecast.map((f) => ({ date: f.date, forecast: f.forecast, band: [f.lower, f.upper] })),
  ] : [];

  return (
    <div>
      <h1 className="page-title">Forecasting Results</h1>
      <p className="page-subtitle">The winning model's accuracy and its next 30-day forecast, per series</p>

      {fc.best_model && (
        <>
          <div className="section-title">Best model: {fc.best_model}</div>
          <div className="section grid grid-4">
            <MetricCard metricKey="test_mape" value={`${fc.test_metrics.mape.toFixed(2)}%`} />
            <MetricCard metricKey="rmse" value={fc.test_metrics.rmse.toFixed(2)} />
            <MetricCard metricKey="mae" value={fc.test_metrics.mae.toFixed(2)} />
            <MetricCard metricKey="r2_score" value={fc.test_metrics.r2_score.toFixed(3)} />
          </div>
        </>
      )}

      {fc.feature_importance?.length > 0 && (
        <div className="section">
          <ChartCard title="Feature importance (top 15)" tooltip="How much each input feature influenced the model's predictions, relative to the others." height={420}>
            <ResponsiveContainer>
              <BarChart data={[...fc.feature_importance].reverse()} layout="vertical" margin={{ left: 40 }}>
                <CartesianGrid horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={140} />
                <Tooltip />
                <Bar dataKey="importance" fill="var(--series-1)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      )}

      <div className="section">
        <select value={active || ""} onChange={(e) => setSid(e.target.value)}>
          {fc.series.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>

      {!view ? <Loading /> : (
        <>
          <div className="section">
            <ChartCard title={`History + 30-day forecast — ${active}`} tooltip="Shaded band = 95% confidence range, from recent forecast volatility." height={380}>
              <ResponsiveContainer>
                <ComposedChart data={merged}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={50} />
                  <YAxis tick={{ fontSize: 11 }} width={45} />
                  <Tooltip />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <Area dataKey="band" stroke="none" fill="var(--series-2)" fillOpacity={0.15} name="95% band" />
                  <Line dataKey="actual" stroke="var(--series-1)" strokeWidth={2} dot={false} name="Actual" connectNulls={false} />
                  <Line dataKey="forecast" stroke="var(--series-2)" strokeWidth={2} strokeDasharray="5 4" dot={false} name="Forecast" />
                </ComposedChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          <div className="section">
            <div className="section-title">Forecast table</div>
            <div className="card table-wrap">
              <table>
                <thead><tr><th>Date</th><th>Forecast</th><th>Lower 95%</th><th>Upper 95%</th></tr></thead>
                <tbody>
                  {view.forecast.map((f) => (
                    <tr key={f.date}><td>{f.date}</td><td>{f.forecast}</td><td>{f.lower}</td><td>{f.upper}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div style={{ marginTop: 12 }}>
              <button className="btn" onClick={() => window.open("http://localhost:8000/api/forecast", "_blank")}>
                ⬇️ Download forecast CSV
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
