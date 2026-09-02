export const PALETTE = [
  "var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)",
  "var(--series-5)", "var(--series-6)", "var(--series-7)", "var(--series-8)",
];

/** Stable name -> color: sorted names get fixed slots so a filtered chart
 * never repaints survivors. */
export function seriesColors(names) {
  const sorted = [...names].sort();
  const map = {};
  sorted.forEach((n, i) => { map[n] = PALETTE[i % PALETTE.length]; });
  return map;
}
