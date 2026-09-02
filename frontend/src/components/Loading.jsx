export function Loading() {
  return (
    <div className="empty-state">
      <span className="spinner spinner-dark" style={{ display: "inline-block", verticalAlign: "middle", marginRight: 8 }} />
      Loading…
    </div>
  );
}

export function ErrorBox({ error }) {
  return <div className="error-state">Couldn't load this: {String(error)}</div>;
}

export function Empty({ children = "Nothing here yet — run the pipeline first." }) {
  return <div className="empty-state">{children}</div>;
}
