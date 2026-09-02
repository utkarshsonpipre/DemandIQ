import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Overview from "./pages/Overview";
import DataOverview from "./pages/DataOverview";
import Eda from "./pages/Eda";
import ModelComparison from "./pages/ModelComparison";
import Forecasting from "./pages/Forecasting";
import MlflowTracking from "./pages/MlflowTracking";
import DriftMonitoring from "./pages/DriftMonitoring";

export default function App() {
  return (
    <div className="app-shell">
      <Sidebar />
      <main className="main">
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/data" element={<DataOverview />} />
          <Route path="/eda" element={<Eda />} />
          <Route path="/models" element={<ModelComparison />} />
          <Route path="/forecast" element={<Forecasting />} />
          <Route path="/mlflow" element={<MlflowTracking />} />
          <Route path="/drift" element={<DriftMonitoring />} />
        </Routes>
      </main>
    </div>
  );
}
