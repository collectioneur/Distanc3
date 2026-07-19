/** Display-only rounding for UI inputs — state keeps full precision. */
export function roundTo(n: number, decimals = 2): number {
  const f = 10 ** decimals;
  return Math.round(n * f) / f;
}
