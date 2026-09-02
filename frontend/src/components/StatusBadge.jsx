const MAP = {
  LOW: { cls: "badge-good", icon: "✓" },
  OK: { cls: "badge-good", icon: "✓" },
  MEDIUM: { cls: "badge-warning", icon: "⚠" },
  HIGH: { cls: "badge-critical", icon: "⚠" },
  DEGRADED: { cls: "badge-critical", icon: "⚠" },
  Production: { cls: "badge-good", icon: "●" },
  Staging: { cls: "badge-warning", icon: "●" },
  Archived: { cls: "badge-neutral", icon: "●" },
};

export default function StatusBadge({ status }) {
  const m = MAP[status] || { cls: "badge-neutral", icon: "•" };
  return <span className={`badge ${m.cls}`}>{m.icon} {status ?? "unknown"}</span>;
}
