/** Pure scale-gizmo math — no store/GPU imports so it stays unit-checkable. */

export const MIN_SCALE = 0.01;

/**
 * Multiplicative factor from a drag along the handle axis: pulling the tip
 * from arrow length to 2× arrow length doubles the scale.
 */
export function scaleFactorFromAxisDelta(refWorldLen: number, delta: number): number {
  return (refWorldLen + delta) / refWorldLen;
}

/**
 * Apply factor to start scale. handleIndex 0/1/2 = single axis, null = uniform.
 * Clamped to MIN_SCALE: zero would divide by zero in the shader.
 */
export function applyFactorToScale(
  start: [number, number, number],
  handleIndex: 0 | 1 | 2 | null,
  factor: number,
): [number, number, number] {
  const next: [number, number, number] = [...start];
  if (handleIndex === null) {
    for (let i = 0; i < 3; i++) next[i] = Math.max(MIN_SCALE, start[i] * factor);
  } else {
    next[handleIndex] = Math.max(MIN_SCALE, start[handleIndex] * factor);
  }
  return next;
}
