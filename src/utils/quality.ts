export const QUALITY_PRESET_COUNT = 5;
export const DEFAULT_QUALITY = 3;

export type QualityPreset = {
  id: string;
  label: string;
  dprCap: number;
  maxSteps: number;
  reflSteps: number;
  outlineSteps: number;
  fpsCap: number;
};

export const QUALITY_PRESETS: readonly QualityPreset[] = [
  {
    id: "potato",
    label: "Potato",
    dprCap: 0.6,
    maxSteps: 24,
    reflSteps: 0,
    outlineSteps: 12,
    fpsCap: 30,
  },
  {
    id: "low",
    label: "Low",
    dprCap: 0.9,
    maxSteps: 32,
    reflSteps: 12,
    outlineSteps: 16,
    fpsCap: 30,
  },
  {
    id: "medium",
    label: "Medium",
    dprCap: 1.2,
    maxSteps: 40,
    reflSteps: 18,
    outlineSteps: 20,
    fpsCap: 60,
  },
  {
    id: "high",
    label: "High",
    dprCap: 1.5,
    maxSteps: 48,
    reflSteps: 24,
    outlineSteps: 24,
    fpsCap: 60,
  },
  {
    id: "ultra",
    label: "Ultra",
    dprCap: Infinity,
    maxSteps: 96,
    reflSteps: 32,
    outlineSteps: 32,
    fpsCap: 60,
  },
] as const;

export function stopFraction(
  index: number,
  count: number = QUALITY_PRESET_COUNT,
): number {
  if (count <= 1) return 0;
  return index / (count - 1);
}

export function nearestStop(
  fraction: number,
  count: number = QUALITY_PRESET_COUNT,
): number {
  const clamped = Math.max(0, Math.min(1, fraction));
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < count; i++) {
    const dist = Math.abs(clamped - stopFraction(i, count));
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  }
  return best;
}

export function dprForPreset(preset: QualityPreset): number {
  const native = window.devicePixelRatio ?? 1;
  if (preset.dprCap === Infinity) return native;
  return Math.min(native, preset.dprCap);
}

export function assertQualityInvariants(): void {
  for (let i = 1; i < QUALITY_PRESETS.length; i++) {
    const prev = QUALITY_PRESETS[i - 1];
    const cur = QUALITY_PRESETS[i];
    if (cur.maxSteps <= prev.maxSteps) {
      throw new Error(`maxSteps not monotonic at ${cur.id}`);
    }
    if (cur.reflSteps < prev.reflSteps) {
      throw new Error(`reflSteps not monotonic at ${cur.id}`);
    }
    if (cur.outlineSteps <= prev.outlineSteps) {
      throw new Error(`outlineSteps not monotonic at ${cur.id}`);
    }
    if (cur.fpsCap < prev.fpsCap) {
      throw new Error(`fpsCap not monotonic at ${cur.id}`);
    }
    if (
      prev.dprCap !== Infinity &&
      cur.dprCap !== Infinity &&
      cur.dprCap <= prev.dprCap
    ) {
      throw new Error(`dprCap not monotonic at ${cur.id}`);
    }
  }

  if (nearestStop(0) !== 0) throw new Error("nearestStop(0) !== 0");
  if (nearestStop(1) !== QUALITY_PRESET_COUNT - 1) {
    throw new Error("nearestStop(1) !== last");
  }
  if (nearestStop(0.12) !== 0) throw new Error("nearestStop(0.12) !== 0");
  if (nearestStop(0.3) !== 1) throw new Error("nearestStop(0.3) !== 1");
  if (nearestStop(0.38) !== 2) throw new Error("nearestStop(0.38) !== 2");
  if (nearestStop(0.62) !== 2) throw new Error("nearestStop(0.62) !== 2");
  if (nearestStop(0.88) !== 4) throw new Error("nearestStop(0.88) !== 4");
}

assertQualityInvariants();
