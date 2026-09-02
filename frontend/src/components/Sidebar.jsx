import { NavLink } from "react-router-dom";

const LINKS = [
  { to: "/", label: "Overview", icon: "🏠", end: true },
  { to: "/data", label: "Data Overview", icon: "📋" },
  { to: "/eda", label: "Exploratory Analysis", icon: "📉" },
  { to: "/models", label: "Model Comparison", icon: "🏆" },
  { to: "/forecast", label: "Forecasting", icon: "🔮" },
  { to: "/mlflow", label: "MLflow Tracking", icon: "🧪" },
  { to: "/drift", label: "Drift Monitoring", icon: "🛰️" },
];

export default function Sidebar() {
  return (
    <aside className="sidebar">
      <h1>📈 DemandIQ</h1>
      <p className="tagline">Demand forecasting platform</p>
      <nav>
        {LINKS.map((l) => (
          <NavLink key={l.to} to={l.to} end={l.end} className={({ isActive }) => (isActive ? "active" : "")}>
            <span>{l.icon}</span>{l.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
