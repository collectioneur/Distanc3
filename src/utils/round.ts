export function roundTo(n: number, decimals = 2): number {
  const f = 10 ** decimals;
  return Math.round(n * f) / f;
}

export function roundVec3(v: [number, number, number]): [number, number, number] {
  return [roundTo(v[0]), roundTo(v[1]), roundTo(v[2])];
}

export function roundLayerParams(
  params: [number, number, number, number],
): [number, number, number, number] {
  return [roundTo(params[0]), roundTo(params[1]), roundTo(params[2]), params[3]];
}
