/**
 * CPU-side transform baking for the shader's CSG instruction stream.
 *
 * The shader used to keep a 16-deep transform stack and re-derive every
 * rotation from Euler angles (6 sin/cos per shape per SDF eval). All of that
 * is constant across pixels, so we bake the full inverse affine map per shape
 * here instead: local = rows · p + offset, evaluated once per scene change.
 *
 * Must reproduce the old GPU math exactly (same composition order), so the
 * rendered image is unchanged:
 *   pEval  = T_{n-1}(...T_0(p))          T_k(x) = D(1/s_k)·R_k⁻¹·(x − p_k)
 *   lp     = D(1/acc)·R_shape⁻¹·D(acc)·(pEval − shapePos)   (unsheared trick)
 *   q      = D(1/shapeScale)·lp
 *   dist   = sd(q) · min(shapeScale) · min(acc)
 * where acc = componentwise product of ancestor scales.
 */

export type Vec3 = readonly [number, number, number];
/** Row-major 3×3: [r0x, r0y, r0z, r1x, r1y, r1z, r2x, r2y, r2z]. */
export type Mat3 = number[];

export interface BakeCtx {
  /** Rows of A in `local = A·p + b`. */
  m: Mat3;
  b: Vec3;
  /** Componentwise product of ancestor group scales. */
  accScl: Vec3;
}

export interface BakedShape {
  row0: Vec3;
  row1: Vec3;
  row2: Vec3;
  offset: Vec3;
  /** Conservative distance factor: min(shapeScale) · min(accScl). */
  factor: number;
}

const DEG_TO_RAD = Math.PI / 180;

export function identityCtx(): BakeCtx {
  return { m: [1, 0, 0, 0, 1, 0, 0, 0, 1], b: [0, 0, 0], accScl: [1, 1, 1] };
}

/**
 * Inverse of Euler XYZ rotation as a matrix: Rx(−x)·Ry(−y)·Rz(−z).
 * Matches the shader's old applyInvRotXYZ application order exactly.
 */
export function invRotXYZ(rotDeg: Vec3): Mat3 {
  const rx = -rotDeg[0] * DEG_TO_RAD;
  const ry = -rotDeg[1] * DEG_TO_RAD;
  const rz = -rotDeg[2] * DEG_TO_RAD;
  const cx = Math.cos(rx), sx = Math.sin(rx);
  const cy = Math.cos(ry), sy = Math.sin(ry);
  const cz = Math.cos(rz), sz = Math.sin(rz);
  // Rz(rz) rows
  const rzM = [cz, -sz, 0, sz, cz, 0, 0, 0, 1];
  // Ry(ry) rows
  const ryM = [cy, 0, sy, 0, 1, 0, -sy, 0, cy];
  // Rx(rx) rows
  const rxM = [1, 0, 0, 0, cx, -sx, 0, sx, cx];
  return mulMat3(rxM, mulMat3(ryM, rzM));
}

export function mulMat3(a: Mat3, b: Mat3): Mat3 {
  const out = new Array<number>(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      out[r * 3 + c] =
        a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
    }
  }
  return out;
}

export function mulMat3Vec3(m: Mat3, v: Vec3): Vec3 {
  return [
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ];
}

/** D(s)·M — scale rows. */
function scaleRows(m: Mat3, s: Vec3): Mat3 {
  return [
    m[0] * s[0], m[1] * s[0], m[2] * s[0],
    m[3] * s[1], m[4] * s[1], m[5] * s[1],
    m[6] * s[2], m[7] * s[2], m[8] * s[2],
  ];
}

/** M·D(s) — scale columns. */
function scaleCols(m: Mat3, s: Vec3): Mat3 {
  return [
    m[0] * s[0], m[1] * s[1], m[2] * s[2],
    m[3] * s[0], m[4] * s[1], m[5] * s[2],
    m[6] * s[0], m[7] * s[1], m[8] * s[2],
  ];
}

/** Enter a group: T(x) = D(1/scale)·R⁻¹·(x − position), composed onto ctx. */
export function composeGroupCtx(
  ctx: BakeCtx,
  position: Vec3,
  rotationDeg: Vec3,
  scale: Vec3,
): BakeCtx {
  const invS: Vec3 = [1 / scale[0], 1 / scale[1], 1 / scale[2]];
  const t = scaleRows(invRotXYZ(rotationDeg), invS);
  return {
    m: mulMat3(t, ctx.m),
    b: mulMat3Vec3(t, [
      ctx.b[0] - position[0],
      ctx.b[1] - position[1],
      ctx.b[2] - position[2],
    ]),
    accScl: [
      ctx.accScl[0] * scale[0],
      ctx.accScl[1] * scale[1],
      ctx.accScl[2] * scale[2],
    ],
  };
}

/**
 * Bake a shape's full inverse map:
 *   q = D(1/shapeScale)·D(1/acc)·R_shape⁻¹·D(acc)·(A·p + b − shapePos)
 */
export function bakeShape(
  ctx: BakeCtx,
  position: Vec3,
  rotationDeg: Vec3,
  scale: Vec3,
): BakedShape {
  const acc = ctx.accScl;
  let m = scaleCols(invRotXYZ(rotationDeg), acc);
  m = scaleRows(m, [
    1 / (scale[0] * acc[0]),
    1 / (scale[1] * acc[1]),
    1 / (scale[2] * acc[2]),
  ]);
  const rows = mulMat3(m, ctx.m);
  const offset = mulMat3Vec3(m, [
    ctx.b[0] - position[0],
    ctx.b[1] - position[1],
    ctx.b[2] - position[2],
  ]);
  return {
    row0: [rows[0], rows[1], rows[2]],
    row1: [rows[3], rows[4], rows[5]],
    row2: [rows[6], rows[7], rows[8]],
    offset,
    factor:
      Math.min(scale[0], scale[1], scale[2]) *
      Math.min(acc[0], acc[1], acc[2]),
  };
}

// ── Self-check ───────────────────────────────────────────────────────────────
// Reference implementation of the OLD sequential GPU math; asserts the baked
// matrix produces the same local point. Runs once at import (same convention
// as quality.ts assertQualityInvariants). Fails loud if composition order or
// sign conventions ever drift.

function refApplyInvRotXYZ(p: Vec3, rotDeg: Vec3): Vec3 {
  const rx = rotDeg[0] * DEG_TO_RAD;
  const ry = rotDeg[1] * DEG_TO_RAD;
  const rz = rotDeg[2] * DEG_TO_RAD;
  const czn = Math.cos(-rz), szn = Math.sin(-rz);
  const p1: Vec3 = [czn * p[0] - szn * p[1], szn * p[0] + czn * p[1], p[2]];
  const cyn = Math.cos(-ry), syn = Math.sin(-ry);
  const p2: Vec3 = [cyn * p1[0] + syn * p1[2], p1[1], -syn * p1[0] + cyn * p1[2]];
  const cxn = Math.cos(-rx), sxn = Math.sin(-rx);
  return [p2[0], cxn * p2[1] - sxn * p2[2], sxn * p2[1] + cxn * p2[2]];
}

type RefGroup = { position: Vec3; rotationDeg: Vec3; scale: Vec3 };

function refLocalPoint(
  p: Vec3,
  groups: RefGroup[],
  shapePos: Vec3,
  shapeRotDeg: Vec3,
  shapeScale: Vec3,
): { q: Vec3; factor: number } {
  let pEval = p;
  const acc: [number, number, number] = [1, 1, 1];
  for (const g of groups) {
    const d0: Vec3 = [
      pEval[0] - g.position[0],
      pEval[1] - g.position[1],
      pEval[2] - g.position[2],
    ];
    const r = refApplyInvRotXYZ(d0, g.rotationDeg);
    pEval = [r[0] / g.scale[0], r[1] / g.scale[1], r[2] / g.scale[2]];
    acc[0] *= g.scale[0];
    acc[1] *= g.scale[1];
    acc[2] *= g.scale[2];
  }
  const diff: Vec3 = [
    pEval[0] - shapePos[0],
    pEval[1] - shapePos[1],
    pEval[2] - shapePos[2],
  ];
  const scaledUp: Vec3 = [diff[0] * acc[0], diff[1] * acc[1], diff[2] * acc[2]];
  const rot = refApplyInvRotXYZ(scaledUp, shapeRotDeg);
  const lp: Vec3 = [rot[0] / acc[0], rot[1] / acc[1], rot[2] / acc[2]];
  const q: Vec3 = [
    lp[0] / shapeScale[0],
    lp[1] / shapeScale[1],
    lp[2] / shapeScale[2],
  ];
  return {
    q,
    factor: Math.min(...shapeScale) * Math.min(...acc),
  };
}

export function assertBakeParity(): void {
  // Deterministic pseudo-random (mulberry32).
  let seed = 0x1234abcd;
  const rnd = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return (((t ^ (t >>> 14)) >>> 0) / 4294967296) * 2 - 1; // [-1, 1)
  };
  const rv = (span: number): Vec3 => [rnd() * span, rnd() * span, rnd() * span];
  const rs = (): Vec3 => [
    0.2 + Math.abs(rnd()) * 2,
    0.2 + Math.abs(rnd()) * 2,
    0.2 + Math.abs(rnd()) * 2,
  ];

  for (let depth = 0; depth <= 4; depth++) {
    for (let iter = 0; iter < 8; iter++) {
      const groups: RefGroup[] = Array.from({ length: depth }, () => ({
        position: rv(2),
        rotationDeg: rv(180),
        scale: rs(),
      }));
      const shapePos = rv(2);
      const shapeRot = rv(180);
      const shapeScale = rs();

      let ctx = identityCtx();
      for (const g of groups) {
        ctx = composeGroupCtx(ctx, g.position, g.rotationDeg, g.scale);
      }
      const baked = bakeShape(ctx, shapePos, shapeRot, shapeScale);

      for (let s = 0; s < 4; s++) {
        const p = rv(3);
        const ref = refLocalPoint(p, groups, shapePos, shapeRot, shapeScale);
        const q: Vec3 = [
          baked.row0[0] * p[0] + baked.row0[1] * p[1] + baked.row0[2] * p[2] + baked.offset[0],
          baked.row1[0] * p[0] + baked.row1[1] * p[1] + baked.row1[2] * p[2] + baked.offset[1],
          baked.row2[0] * p[0] + baked.row2[1] * p[1] + baked.row2[2] * p[2] + baked.offset[2],
        ];
        for (let c = 0; c < 3; c++) {
          const scaleRef = Math.max(1, Math.abs(ref.q[c]));
          if (Math.abs(q[c] - ref.q[c]) / scaleRef > 1e-9) {
            throw new Error(
              `bake parity broken: depth=${depth} iter=${iter} comp=${c} baked=${q[c]} ref=${ref.q[c]}`,
            );
          }
        }
        if (Math.abs(baked.factor - ref.factor) > 1e-12) {
          throw new Error(
            `bake factor broken: depth=${depth} baked=${baked.factor} ref=${ref.factor}`,
          );
        }
      }
    }
  }
}

assertBakeParity();
