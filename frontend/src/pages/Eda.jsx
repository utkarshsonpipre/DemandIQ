import { useState } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { useApi } from "../useApi";
import ChartCard from "../components/ChartCard";
import { Loading, ErrorBox } from "../components/Loading";

export default function Eda() {
  const { data: series } = useApi("/series");
  const [sid, setSid] = useState(null);
  const active = sid || series?.[0];
  const { data, error, loading } = useApi(active ? `/eda/${active}` : null, [active]);

  return (
    <div>
      <h1 className="page-title">Exploratory Analysis</h1>
      <p className="page-subtitle">Historical trend, seasonal decomposition, and autocorrelation for one series</p>

      {series && (
        <div className="section">
          <select value={active || ""} onChange={(e) => setSid(e.target.value)}>
            {series.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      )}

      {loading && <Loading />}
      {error && <ErrorBox error={error} />}
      {data && (
        <>
          <div className="section">
            <ChartCard title={`Historical sales — ${active}`}>
              <ResponsiveContainer>
                <LineChart data={data.history}>
                  <CartesianGrid vertical={false} />
                  <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={40} />
                  <YAxis tick={{ fontSize: 11 }} width={45} />
                  <Tooltip />
                  <Line type="monotone" dataKey="sales" stroke="var(--series-1)" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          {data.decomposition && (
            <div className="section grid grid-3">
              {["trend", "seasonal", "residual"].map((k) => (
                <ChartCard key={k} title={k[0].toUpperCase() + k.slice(1)} height={220}>
                  <ResponsiveContainer>
                    <LineChart data={data.decomposition[k]}>
                      <CartesianGrid vertical={false} />
                      <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={60} />
                      <YAxis tick={{ fontSize: 10 }} width={40} />
                      <Tooltip />
                      <Line type="monotone" dataKey="value" stroke="var(--series-1)" strokeWidth={1.5} dot={false} connectNulls />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>
              ))}
            </div>
          )}

          {data.acf_pacf && (
            <div className="section grid grid-2">
              {["acf", "pacf"].map((k) => (
                <ChartCard key={k} title={k.toUpperCase()}
                          tooltip={k === "acf"
                            ? "Autocorrelation — how strongly sales correlate with themselves N days ago."
                            : "Partial autocorrelation — the same, with the effect of shorter lags removed."}>
                  <ResponsiveContainer>
                    <BarChart data={data.acf_pacf[k].map((v, i) => ({ lag: i, value: v }))}>
                      <CartesianGrid vertical={false} />
                      <XAxis dataKey="lag" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} width={40} />
                      <Tooltip />
                      <Bar dataKey="value" fill="var(--series-1)" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartCard>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
