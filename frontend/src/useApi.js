import { useEffect, useState, useCallback } from "react";
import { api } from "./api";

/** GET a JSON endpoint; re-fetches when `deps` change. */
export function useApi(path, deps = []) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const reload = useCallback(() => {
    if (!path) { setLoading(false); return; }
    setLoading(true);
    setError(null);
    api.get(path)
      .then((res) => setData(res.data))
      .catch((err) => setError(err.response?.data?.detail || err.message))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path]);

  useEffect(reload, [reload, ...deps]);

  return { data, error, loading, reload };
}
