import {
  ALL_SHAPE_TYPES,
  type ObjectGroup,
  type OpType,
  type SceneItem,
  type SceneRoot,
  type ShapeLayer,
  type ShapeType,
} from "../store/sceneStore";

export type CodeLang = "typegpu" | "wgsl" | "glsl";

export interface GeneratedSdf {
  fnName: string;
  typegpu: string;
  wgsl: string;
  glsl: string;
}

export const PLACEHOLDER = "// Nothing to export — add shapes to the scene";

const DEG_TO_RAD = Math.PI / 180;

type V3 = readonly [number, number, number];

const isZero = (v: V3) => v.every((c) => Math.abs(c) < 1e-9);
const isOne = (v: V3) => v.every((c) => Math.abs(c - 1) < 1e-9);
const isUniform = (v: V3) => v[0] === v[1] && v[1] === v[2];
const minV = (v: V3) => Math.min(v[0], v[1], v[2]);
const mulV = (a: V3, b: V3): V3 => [a[0] * b[0], a[1] * b[1], a[2] * b[2]];
const toRad = (deg: V3): V3 => [deg[0] * DEG_TO_RAD, deg[1] * DEG_TO_RAD, deg[2] * DEG_TO_RAD];

function fmt(n: number): string {
  const s = parseFloat(n.toFixed(6)).toString();
  return /[.e]/.test(s) ? s : s + ".0";
}

// ponytail: reciprocal-multiply instead of vec division for TypeGPU (no
// documented vec/vec `/` overload); full-precision literal keeps f32 parity.
function fmtRecip(n: number): string {
  const s = parseFloat((1 / n).toPrecision(9)).toString();
  return /[.e]/.test(s) ? s : s + ".0";
}

// ── IR ────────────────────────────────────────────────────────────────────────

interface ShapeRhs {
  kind: "shape";
  shape: ShapeType;
  point: string;
  pos: V3 | null;
  rotRad: V3 | null;
  /** Non-uniform accumulated ancestor scale — triggers unsheared-rotation trick. */
  accScl: V3 | null;
  scale: V3 | null;
  /** min(scale) * min(accScl) conservative distance factor. */
  factor: number;
  params: [number, number, number, number];
}
interface VarRhs {
  kind: "var";
  name: string;
}
type Rhs = ShapeRhs | VarRhs;

type Stmt =
  | { t: "point"; name: string; from: string; pos: V3 | null; rotRad: V3 | null; scale: V3 | null }
  | { t: "def"; name: string; rhs: Rhs }
  | { t: "combine"; name: string; op: OpType; k: number; rhs: Rhs };

interface Ir {
  stmts: Stmt[];
  result: string | null;
  usedShapes: Set<ShapeType>;
  usesRotation: boolean;
  usesSmin: boolean;
}

const SMOOTH_OPS = new Set<OpType>(["sUnion", "sSubtract", "sIntersect"]);

function findItemIn(items: SceneItem[], itemId: string): SceneItem | null {
  for (const item of items) {
    if (item.id === itemId) return item;
    if (item.kind === "group") {
      const found = findItemIn(item.items, itemId);
      if (found) return found;
    }
  }
  return null;
}

/**
 * Build language-neutral statements mirroring the viewport shader math
 * (see src/gpu/shader.ts): per-group applyParentTransform, unsheared shape
 * rotation under non-uniform ancestor scale, min-scale distance factors.
 * Selected node's own position is dropped so the model sits at the origin.
 */
function buildIr(root: SceneRoot, selectedItemId: string | null): Ir {
  const stmts: Stmt[] = [];
  const usedShapes = new Set<ShapeType>();
  let usesRotation = false;
  let usesSmin = false;
  let accN = 0;
  let ptN = 0;

  function shapeRhs(layer: ShapeLayer, point: string, accScl: V3, dropPos: boolean): ShapeRhs {
    usedShapes.add(layer.shapeType);
    const rot = isZero(layer.rotation) ? null : toRad(layer.rotation);
    if (rot) usesRotation = true;
    return {
      kind: "shape",
      shape: layer.shapeType,
      point,
      pos: dropPos || isZero(layer.position) ? null : layer.position,
      rotRad: rot,
      accScl: rot && !isUniform(accScl) ? accScl : null,
      scale: isOne(layer.scale) ? null : layer.scale,
      factor: minV(layer.scale) * minV(accScl),
      params: layer.params,
    };
  }

  function groupPoint(group: ObjectGroup, point: string, dropPos: boolean): string {
    const pos = dropPos || isZero(group.position) ? null : group.position;
    const rot = isZero(group.rotation) ? null : toRad(group.rotation);
    const scale = isOne(group.scale) ? null : group.scale;
    if (!pos && !rot && !scale) return point;
    if (rot) usesRotation = true;
    const name = `q${++ptN}`;
    stmts.push({ t: "point", name, from: point, pos, rotRad: rot, scale });
    return name;
  }

  function walkItems(items: SceneItem[], point: string, accScl: V3): string | null {
    let acc: string | null = null;
    for (const item of items) {
      let rhs: Rhs | null = null;
      if (item.kind === "layer") {
        rhs = shapeRhs(item, point, accScl, false);
      } else if (item.items.length > 0) {
        const childPoint = groupPoint(item, point, false);
        const childAcc = walkItems(item.items, childPoint, mulV(accScl, item.scale));
        if (childAcc) rhs = { kind: "var", name: childAcc };
      }
      if (!rhs) continue;
      if (!acc) {
        acc = `d${accN++}`;
        stmts.push({ t: "def", name: acc, rhs });
      } else {
        if (SMOOTH_OPS.has(item.op)) usesSmin = true;
        stmts.push({ t: "combine", name: acc, op: item.op, k: item.smoothK, rhs });
      }
    }
    return acc;
  }

  const selected = selectedItemId ? findItemIn(root.items, selectedItemId) : null;
  let result: string | null;
  if (selected?.kind === "layer") {
    const acc = `d${accN++}`;
    stmts.push({ t: "def", name: acc, rhs: shapeRhs(selected, "p", [1, 1, 1], true) });
    result = acc;
  } else if (selected?.kind === "group") {
    const point = groupPoint(selected, "p", true);
    result = walkItems(selected.items, point, selected.scale);
  } else {
    result = walkItems(root.items, "p", [1, 1, 1]);
  }

  return { stmts, result, usedShapes, usesRotation, usesSmin };
}

// ── Helper sources ────────────────────────────────────────────────────────────

const TG_HELPERS: Record<ShapeType, string> = {
  sphere: `const sdSphere = (p: d.v3f, R: number): number => {
  "use gpu";
  return std.length(p) - R;
};`,
  box: `const sdBox = (p: d.v3f, b: d.v3f): number => {
  "use gpu";
  const q = std.abs(p) - b;
  return (
    std.length(std.max(q, d.vec3f(0.0))) +
    std.min(std.max(q.x, std.max(q.y, q.z)), 0.0)
  );
};`,
  torus: `const sdTorus = (p: d.v3f, t: d.v2f): number => {
  "use gpu";
  const q = d.vec2f(std.length(d.vec2f(p.x, p.z)) - t.x, p.y);
  return std.length(q) - t.y;
};`,
  cylinder: `const sdCylinder = (p: d.v3f, r: number, h: number): number => {
  "use gpu";
  const d2 = d.vec2f(std.length(d.vec2f(p.x, p.z)) - r, std.abs(p.y) - h);
  return std.min(std.max(d2.x, d2.y), 0.0) + std.length(std.max(d2, d.vec2f(0.0)));
};`,
  capsule: `const sdCapsule = (p: d.v3f, r: number, h: number): number => {
  "use gpu";
  const py = std.clamp(p.y, -h, h);
  return std.length(p - d.vec3f(0.0, py, 0.0)) - r;
};`,
  cone: `const sdCone = (p: d.v3f, r: number, h: number): number => {
  "use gpu";
  const q = d.vec2f(std.length(d.vec2f(p.x, p.z)), p.y);
  const k1 = d.vec2f(0.0, h);
  const k2 = d.vec2f(-r, 2.0 * h);
  let capR = d.f32(0.0);
  if (p.y < 0.0) { capR = d.f32(r); }
  const ca = d.vec2f(q.x - std.min(q.x, capR), std.abs(p.y) - h);
  const t = std.clamp(std.dot(k1 - q, k2) / std.dot(k2, k2), 0.0, 1.0);
  const cb = q - k1 + k2 * t;
  let s = d.f32(1.0);
  if (cb.x < 0.0) { if (ca.y < 0.0) { s = d.f32(-1.0); } }
  return s * std.sqrt(std.min(std.dot(ca, ca), std.dot(cb, cb)));
};`,
  roundedBox: `const sdRoundedBox = (p: d.v3f, b: d.v3f, r: number): number => {
  "use gpu";
  const q = std.abs(p) - b;
  return std.length(std.max(q, d.vec3f(0.0))) + std.min(std.max(q.x, std.max(q.y, q.z)), 0.0) - r;
};`,
  boxFrame: `const sdBoxFrame = (pIn: d.v3f, b: d.v3f, e: number): number => {
  "use gpu";
  const p = std.abs(pIn) - b;
  const q = std.abs(p + d.vec3f(e)) - d.vec3f(e);
  const dA = std.length(std.max(d.vec3f(p.x, q.y, q.z), d.vec3f(0.0))) + std.min(std.max(p.x, std.max(q.y, q.z)), 0.0);
  const dB = std.length(std.max(d.vec3f(q.x, p.y, q.z), d.vec3f(0.0))) + std.min(std.max(q.x, std.max(p.y, q.z)), 0.0);
  const dC = std.length(std.max(d.vec3f(q.x, q.y, p.z), d.vec3f(0.0))) + std.min(std.max(q.x, std.max(q.y, p.z)), 0.0);
  return std.min(dA, std.min(dB, dC));
};`,
  cappedTorus: `const sdCappedTorus = (pIn: d.v3f, ra: number, rb: number, angleDeg: number): number => {
  "use gpu";
  const an = angleDeg * 0.017453292519943295;
  const sc = d.vec2f(std.sin(an), std.cos(an));
  const p = d.vec3f(std.abs(pIn.x), pIn.z, pIn.y);
  let k = std.length(d.vec2f(p.x, p.y));
  if (sc.y * p.x > sc.x * p.y) { k = std.dot(d.vec2f(p.x, p.y), sc); }
  return std.sqrt(std.dot(p, p) + ra * ra - 2.0 * ra * k) - rb;
};`,
  link: `const sdLink = (p: d.v3f, le: number, r1: number, r2: number): number => {
  "use gpu";
  const q = d.vec3f(p.x, std.max(std.abs(p.y) - le, 0.0), p.z);
  return std.length(d.vec2f(std.length(d.vec2f(q.x, q.y)) - r1, q.z)) - r2;
};`,
  hexPrism: `const sdHexPrism = (pIn: d.v3f, r: number, h: number): number => {
  "use gpu";
  const k = d.vec3f(-0.8660254, 0.5, 0.57735);
  const q0 = std.abs(d.vec3f(pIn.x, pIn.z, pIn.y));
  const kd = 2.0 * std.min(std.dot(d.vec2f(k.x, k.y), d.vec2f(q0.x, q0.y)), 0.0);
  const qx = q0.x - kd * k.x;
  const qy = q0.y - kd * k.y;
  const d1 = std.length(d.vec2f(qx - std.clamp(qx, -k.z * r, k.z * r), qy - r)) * std.sign(qy - r);
  const d2 = q0.z - h;
  return std.min(std.max(d1, d2), 0.0) + std.length(std.max(d.vec2f(d1, d2), d.vec2f(0.0)));
};`,
  triPrism: `const sdTriPrism = (pIn: d.v3f, r: number, h: number): number => {
  "use gpu";
  const p = d.vec3f(pIn.x, pIn.z, pIn.y);
  const q = std.abs(p);
  return std.max(q.z - h, std.max(q.x * 0.866025 + p.y * 0.5, -p.y) - r * 0.5);
};`,
  roundedCylinder: `const sdRoundedCylinder = (p: d.v3f, ra: number, rb: number, h: number): number => {
  "use gpu";
  const d2 = d.vec2f(std.length(d.vec2f(p.x, p.z)) - ra + rb, std.abs(p.y) - h);
  return std.min(std.max(d2.x, d2.y), 0.0) + std.length(std.max(d2, d.vec2f(0.0))) - rb;
};`,
  roundCone: `const sdRoundCone = (p: d.v3f, r1: number, r2: number, h: number): number => {
  "use gpu";
  const b = std.clamp((r1 - r2) / std.max(h, 0.0001), -0.999, 0.999);
  const a = std.sqrt(1.0 - b * b);
  const q = d.vec2f(std.length(d.vec2f(p.x, p.z)), p.y);
  const k = std.dot(q, d.vec2f(-b, a));
  let result = std.dot(q, d.vec2f(a, b)) - r1;
  if (k < 0.0) { result = std.length(q) - r1; }
  else if (k > a * h) { result = std.length(q - d.vec2f(0.0, h)) - r2; }
  return result;
};`,
  solidAngle: `const sdSolidAngle = (p: d.v3f, angleDeg: number, ra: number): number => {
  "use gpu";
  const an = angleDeg * 0.017453292519943295;
  const sc = d.vec2f(std.sin(an), std.cos(an));
  const q = d.vec2f(std.length(d.vec2f(p.x, p.z)), p.y);
  const l = std.length(q) - ra;
  const m = std.length(q - sc * std.clamp(std.dot(q, sc), 0.0, ra));
  return std.max(l, m * std.sign(sc.y * q.x - sc.x * q.y));
};`,
  cutSphere: `const sdCutSphere = (p: d.v3f, r: number, hIn: number): number => {
  "use gpu";
  const h = std.clamp(hIn, -r * 0.999, r * 0.999);
  const w = std.sqrt(r * r - h * h);
  const q = d.vec2f(std.length(d.vec2f(p.x, p.z)), p.y);
  const s = std.max((h - r) * q.x * q.x + w * w * (h + r - 2.0 * q.y), h * q.x - w * q.y);
  let result = std.length(q - d.vec2f(w, h));
  if (s < 0.0) { result = std.length(q) - r; }
  else if (q.x < w) { result = h - q.y; }
  return result;
};`,
  cutHollowSphere: `const sdCutHollowSphere = (p: d.v3f, r: number, hIn: number, t: number): number => {
  "use gpu";
  const h = std.clamp(hIn, -r * 0.999, r * 0.999);
  const w = std.sqrt(r * r - h * h);
  const q = d.vec2f(std.length(d.vec2f(p.x, p.z)), p.y);
  let result = std.abs(std.length(q) - r);
  if (h * q.x < w * q.y) { result = std.length(q - d.vec2f(w, h)); }
  return result - t;
};`,
  deathStar: `const sdDeathStar = (p: d.v3f, ra: number, rb: number, dOff: number): number => {
  "use gpu";
  const dd = std.max(dOff, 0.0001);
  const a = (ra * ra - rb * rb + dd * dd) / (2.0 * dd);
  const b = std.sqrt(std.max(ra * ra - a * a, 0.0));
  const q = d.vec2f(p.y, std.length(d.vec2f(p.x, p.z)));
  let result = std.max(std.length(q) - ra, -(std.length(q - d.vec2f(dd, 0.0)) - rb));
  if (q.x * b - q.y * a > dd * std.max(b - q.y, 0.0)) { result = std.length(q - d.vec2f(a, b)); }
  return result;
};`,
  rhombus: `const ndot2 = (a: d.v2f, b: d.v2f): number => {
  "use gpu";
  return a.x * b.x - a.y * b.y;
};

const sdRhombus = (p: d.v3f, la: number, lb: number, h: number, ra: number): number => {
  "use gpu";
  const q = std.abs(p);
  const b = d.vec2f(la, lb);
  const f = std.clamp(ndot2(b, b - d.vec2f(2.0 * q.x, 2.0 * q.z)) / std.dot(b, b), -1.0, 1.0);
  const qx = std.length(d.vec2f(q.x, q.z) - b * d.vec2f(1.0 - f, 1.0 + f) * 0.5) * std.sign(q.x * b.y + q.z * b.x - b.x * b.y) - ra;
  const qy = q.y - h;
  return std.min(std.max(qx, qy), 0.0) + std.length(std.max(d.vec2f(qx, qy), d.vec2f(0.0)));
};`,
  octahedron: `const sdOctahedron = (pIn: d.v3f, s: number): number => {
  "use gpu";
  const p = std.abs(pIn);
  const m = p.x + p.y + p.z - s;
  let result = m * 0.57735027;
  let q = d.vec3f(0.0);
  let onEdge = d.f32(0.0);
  if (3.0 * p.x < m) { q = d.vec3f(p.x, p.y, p.z); onEdge = d.f32(1.0); }
  else if (3.0 * p.y < m) { q = d.vec3f(p.y, p.z, p.x); onEdge = d.f32(1.0); }
  else if (3.0 * p.z < m) { q = d.vec3f(p.z, p.x, p.y); onEdge = d.f32(1.0); }
  if (onEdge > 0.5) {
    const k = std.clamp(0.5 * (q.z - q.y + s), 0.0, s);
    result = std.length(d.vec3f(q.x, q.y - s + k, q.z - k));
  }
  return result;
};`,
  pyramid: `const sdPyramid = (pIn: d.v3f, base: number, height: number): number => {
  "use gpu";
  const invB = 1.0 / std.max(base, 0.0001);
  const p0 = d.vec3f(pIn.x * invB, pIn.y * invB, pIn.z * invB);
  const h = height * invB;
  const m2 = h * h + 0.25;
  let px = std.abs(p0.x);
  let pz = std.abs(p0.z);
  if (pz > px) { const tmp = px; px = pz; pz = tmp; }
  px = px - 0.5;
  pz = pz - 0.5;
  const q = d.vec3f(pz, h * p0.y - 0.5 * px, h * px + 0.5 * p0.y);
  if (q.z < 0.0 && p0.y > 0.0) {
    return -std.min(-q.z / std.sqrt(m2), p0.y) * base;
  }
  let dSlant = d.f32(1e10);
  if (q.z > 0.0) {
    const s = std.max(-q.x, 0.0);
    const t = std.clamp((q.y - 0.5 * pz) / (m2 + 0.25), 0.0, 1.0);
    const a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    const b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);
    let d2 = std.min(a, b);
    if (std.min(q.y, -q.x * m2 - q.y * 0.5) > 0.0) { d2 = d.f32(0.0); }
    dSlant = std.sqrt((d2 + q.z * q.z) / m2);
  }
  let dBase = d.f32(1e10);
  if (p0.y < 0.0) {
    dBase = std.length(d.vec3f(std.max(px, 0.0), p0.y, std.max(pz, 0.0)));
  }
  return std.min(dSlant, dBase) * base;
};`,
  vesica: `const sdVesica = (p: d.v3f, r: number, dIn: number): number => {
  "use gpu";
  const dd = std.clamp(dIn, 0.0001, r * 0.999);
  const b = std.sqrt(r * r - dd * dd);
  const q = d.vec2f(std.length(d.vec2f(p.x, p.z)), std.abs(p.y));
  let result = std.length(q - d.vec2f(-dd, 0.0)) - r;
  if ((q.y - b) * dd > q.x * b) { result = std.length(q - d.vec2f(0.0, b)); }
  return result;
};`,
};

const WGSL_HELPERS: Record<ShapeType, string> = {
  sphere: `fn sdSphere(p: vec3f, R: f32) -> f32 {
  return length(p) - R;
}`,
  box: `fn sdBox(p: vec3f, b: vec3f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}`,
  torus: `fn sdTorus(p: vec3f, t: vec2f) -> f32 {
  let q = vec2f(length(p.xz) - t.x, p.y);
  return length(q) - t.y;
}`,
  cylinder: `fn sdCylinder(p: vec3f, r: f32, h: f32) -> f32 {
  let d2 = vec2f(length(p.xz) - r, abs(p.y) - h);
  return min(max(d2.x, d2.y), 0.0) + length(max(d2, vec2f(0.0)));
}`,
  capsule: `fn sdCapsule(p: vec3f, r: f32, h: f32) -> f32 {
  let py = clamp(p.y, -h, h);
  return length(p - vec3f(0.0, py, 0.0)) - r;
}`,
  cone: `fn sdCone(p: vec3f, r: f32, h: f32) -> f32 {
  let q = vec2f(length(p.xz), p.y);
  let k1 = vec2f(0.0, h);
  let k2 = vec2f(-r, 2.0 * h);
  var capR = 0.0;
  if (p.y < 0.0) { capR = r; }
  let ca = vec2f(q.x - min(q.x, capR), abs(p.y) - h);
  let t = clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
  let cb = q - k1 + k2 * t;
  var s = 1.0;
  if (cb.x < 0.0 && ca.y < 0.0) { s = -1.0; }
  return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}`,
  roundedBox: `fn sdRoundedBox(p: vec3f, b: vec3f, r: f32) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}`,
  boxFrame: `fn sdBoxFrame(pIn: vec3f, b: vec3f, e: f32) -> f32 {
  let p = abs(pIn) - b;
  let q = abs(p + vec3f(e)) - vec3f(e);
  let dA = length(max(vec3f(p.x, q.y, q.z), vec3f(0.0))) + min(max(p.x, max(q.y, q.z)), 0.0);
  let dB = length(max(vec3f(q.x, p.y, q.z), vec3f(0.0))) + min(max(q.x, max(p.y, q.z)), 0.0);
  let dC = length(max(vec3f(q.x, q.y, p.z), vec3f(0.0))) + min(max(q.x, max(q.y, p.z)), 0.0);
  return min(dA, min(dB, dC));
}`,
  cappedTorus: `fn sdCappedTorus(pIn: vec3f, ra: f32, rb: f32, angleDeg: f32) -> f32 {
  let an = angleDeg * 0.017453292519943295;
  let sc = vec2f(sin(an), cos(an));
  let p = vec3f(abs(pIn.x), pIn.z, pIn.y);
  var k = length(p.xy);
  if (sc.y * p.x > sc.x * p.y) { k = dot(p.xy, sc); }
  return sqrt(dot(p, p) + ra * ra - 2.0 * ra * k) - rb;
}`,
  link: `fn sdLink(p: vec3f, le: f32, r1: f32, r2: f32) -> f32 {
  let q = vec3f(p.x, max(abs(p.y) - le, 0.0), p.z);
  return length(vec2f(length(q.xy) - r1, q.z)) - r2;
}`,
  hexPrism: `fn sdHexPrism(pIn: vec3f, r: f32, h: f32) -> f32 {
  let k = vec3f(-0.8660254, 0.5, 0.57735);
  let q0 = abs(vec3f(pIn.x, pIn.z, pIn.y));
  let kd = 2.0 * min(dot(k.xy, q0.xy), 0.0);
  let qx = q0.x - kd * k.x;
  let qy = q0.y - kd * k.y;
  let d1 = length(vec2f(qx - clamp(qx, -k.z * r, k.z * r), qy - r)) * sign(qy - r);
  let d2 = q0.z - h;
  return min(max(d1, d2), 0.0) + length(max(vec2f(d1, d2), vec2f(0.0)));
}`,
  triPrism: `fn sdTriPrism(pIn: vec3f, r: f32, h: f32) -> f32 {
  let p = vec3f(pIn.x, pIn.z, pIn.y);
  let q = abs(p);
  return max(q.z - h, max(q.x * 0.866025 + p.y * 0.5, -p.y) - r * 0.5);
}`,
  roundedCylinder: `fn sdRoundedCylinder(p: vec3f, ra: f32, rb: f32, h: f32) -> f32 {
  let d2 = vec2f(length(p.xz) - ra + rb, abs(p.y) - h);
  return min(max(d2.x, d2.y), 0.0) + length(max(d2, vec2f(0.0))) - rb;
}`,
  roundCone: `fn sdRoundCone(p: vec3f, r1: f32, r2: f32, h: f32) -> f32 {
  let b = clamp((r1 - r2) / max(h, 0.0001), -0.999, 0.999);
  let a = sqrt(1.0 - b * b);
  let q = vec2f(length(p.xz), p.y);
  let k = dot(q, vec2f(-b, a));
  if (k < 0.0) { return length(q) - r1; }
  if (k > a * h) { return length(q - vec2f(0.0, h)) - r2; }
  return dot(q, vec2f(a, b)) - r1;
}`,
  solidAngle: `fn sdSolidAngle(p: vec3f, angleDeg: f32, ra: f32) -> f32 {
  let an = angleDeg * 0.017453292519943295;
  let sc = vec2f(sin(an), cos(an));
  let q = vec2f(length(p.xz), p.y);
  let l = length(q) - ra;
  let m = length(q - sc * clamp(dot(q, sc), 0.0, ra));
  return max(l, m * sign(sc.y * q.x - sc.x * q.y));
}`,
  cutSphere: `fn sdCutSphere(p: vec3f, r: f32, hIn: f32) -> f32 {
  let h = clamp(hIn, -r * 0.999, r * 0.999);
  let w = sqrt(r * r - h * h);
  let q = vec2f(length(p.xz), p.y);
  let s = max((h - r) * q.x * q.x + w * w * (h + r - 2.0 * q.y), h * q.x - w * q.y);
  if (s < 0.0) { return length(q) - r; }
  if (q.x < w) { return h - q.y; }
  return length(q - vec2f(w, h));
}`,
  cutHollowSphere: `fn sdCutHollowSphere(p: vec3f, r: f32, hIn: f32, t: f32) -> f32 {
  let h = clamp(hIn, -r * 0.999, r * 0.999);
  let w = sqrt(r * r - h * h);
  let q = vec2f(length(p.xz), p.y);
  if (h * q.x < w * q.y) { return length(q - vec2f(w, h)) - t; }
  return abs(length(q) - r) - t;
}`,
  deathStar: `fn sdDeathStar(p: vec3f, ra: f32, rb: f32, dOff: f32) -> f32 {
  let dd = max(dOff, 0.0001);
  let a = (ra * ra - rb * rb + dd * dd) / (2.0 * dd);
  let b = sqrt(max(ra * ra - a * a, 0.0));
  let q = vec2f(p.y, length(p.xz));
  if (q.x * b - q.y * a > dd * max(b - q.y, 0.0)) { return length(q - vec2f(a, b)); }
  return max(length(q) - ra, -(length(q - vec2f(dd, 0.0)) - rb));
}`,
  rhombus: `fn ndot2(a: vec2f, b: vec2f) -> f32 {
  return a.x * b.x - a.y * b.y;
}

fn sdRhombus(p: vec3f, la: f32, lb: f32, h: f32, ra: f32) -> f32 {
  let q = abs(p);
  let b = vec2f(la, lb);
  let f = clamp(ndot2(b, b - 2.0 * q.xz) / dot(b, b), -1.0, 1.0);
  let qx = length(q.xz - 0.5 * b * vec2f(1.0 - f, 1.0 + f)) * sign(q.x * b.y + q.z * b.x - b.x * b.y) - ra;
  let qy = q.y - h;
  return min(max(qx, qy), 0.0) + length(max(vec2f(qx, qy), vec2f(0.0)));
}`,
  octahedron: `fn sdOctahedron(pIn: vec3f, s: f32) -> f32 {
  let p = abs(pIn);
  let m = p.x + p.y + p.z - s;
  var q = vec3f(0.0);
  if (3.0 * p.x < m) { q = p.xyz; }
  else if (3.0 * p.y < m) { q = p.yzx; }
  else if (3.0 * p.z < m) { q = p.zxy; }
  else { return m * 0.57735027; }
  let k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
  return length(vec3f(q.x, q.y - s + k, q.z - k));
}`,
  pyramid: `fn sdPyramid(pIn: vec3f, base: f32, height: f32) -> f32 {
  let invB = 1.0 / max(base, 0.0001);
  let p0 = pIn * invB;
  let h = height * invB;
  let m2 = h * h + 0.25;
  var px = abs(p0.x);
  var pz = abs(p0.z);
  if (pz > px) { let tmp = px; px = pz; pz = tmp; }
  px -= 0.5;
  pz -= 0.5;
  let q = vec3f(pz, h * p0.y - 0.5 * px, h * px + 0.5 * p0.y);
  if (q.z < 0.0 && p0.y > 0.0) {
    return -min(-q.z / sqrt(m2), p0.y) * base;
  }
  var dSlant = 1e10;
  if (q.z > 0.0) {
    let s = max(-q.x, 0.0);
    let t = clamp((q.y - 0.5 * pz) / (m2 + 0.25), 0.0, 1.0);
    let a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    let b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);
    var d2 = min(a, b);
    if (min(q.y, -q.x * m2 - q.y * 0.5) > 0.0) { d2 = 0.0; }
    dSlant = sqrt((d2 + q.z * q.z) / m2);
  }
  var dBase = 1e10;
  if (p0.y < 0.0) {
    dBase = length(vec3f(max(px, 0.0), p0.y, max(pz, 0.0)));
  }
  return min(dSlant, dBase) * base;
}`,
  vesica: `fn sdVesica(p: vec3f, r: f32, dIn: f32) -> f32 {
  let dd = clamp(dIn, 0.0001, r * 0.999);
  let b = sqrt(r * r - dd * dd);
  let q = vec2f(length(p.xz), abs(p.y));
  if ((q.y - b) * dd > q.x * b) { return length(q - vec2f(0.0, b)); }
  return length(q - vec2f(-dd, 0.0)) - r;
}`,
};

const GLSL_HELPERS: Record<ShapeType, string> = {
  sphere: `float sdSphere(vec3 p, float R) {
  return length(p) - R;
}`,
  box: `float sdBox(vec3 p, vec3 b) {
  vec3 q = abs(p) - b;
  return length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}`,
  torus: `float sdTorus(vec3 p, vec2 t) {
  vec2 q = vec2(length(p.xz) - t.x, p.y);
  return length(q) - t.y;
}`,
  cylinder: `float sdCylinder(vec3 p, float r, float h) {
  vec2 d2 = vec2(length(p.xz) - r, abs(p.y) - h);
  return min(max(d2.x, d2.y), 0.0) + length(max(d2, vec2(0.0)));
}`,
  capsule: `float sdCapsule(vec3 p, float r, float h) {
  float py = clamp(p.y, -h, h);
  return length(p - vec3(0.0, py, 0.0)) - r;
}`,
  cone: `float sdCone(vec3 p, float r, float h) {
  vec2 q = vec2(length(p.xz), p.y);
  vec2 k1 = vec2(0.0, h);
  vec2 k2 = vec2(-r, 2.0 * h);
  float capR = 0.0;
  if (p.y < 0.0) { capR = r; }
  vec2 ca = vec2(q.x - min(q.x, capR), abs(p.y) - h);
  float t = clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
  vec2 cb = q - k1 + k2 * t;
  float s = 1.0;
  if (cb.x < 0.0 && ca.y < 0.0) { s = -1.0; }
  return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}`,
  roundedBox: `float sdRoundedBox(vec3 p, vec3 b, float r) {
  vec3 q = abs(p) - b;
  return length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}`,
  boxFrame: `float sdBoxFrame(vec3 pIn, vec3 b, float e) {
  vec3 p = abs(pIn) - b;
  vec3 q = abs(p + vec3(e)) - vec3(e);
  float dA = length(max(vec3(p.x, q.y, q.z), vec3(0.0))) + min(max(p.x, max(q.y, q.z)), 0.0);
  float dB = length(max(vec3(q.x, p.y, q.z), vec3(0.0))) + min(max(q.x, max(p.y, q.z)), 0.0);
  float dC = length(max(vec3(q.x, q.y, p.z), vec3(0.0))) + min(max(q.x, max(q.y, p.z)), 0.0);
  return min(dA, min(dB, dC));
}`,
  cappedTorus: `float sdCappedTorus(vec3 pIn, float ra, float rb, float angleDeg) {
  float an = angleDeg * 0.017453292519943295;
  vec2 sc = vec2(sin(an), cos(an));
  vec3 p = vec3(abs(pIn.x), pIn.z, pIn.y);
  float k = (sc.y * p.x > sc.x * p.y) ? dot(p.xy, sc) : length(p.xy);
  return sqrt(dot(p, p) + ra * ra - 2.0 * ra * k) - rb;
}`,
  link: `float sdLink(vec3 p, float le, float r1, float r2) {
  vec3 q = vec3(p.x, max(abs(p.y) - le, 0.0), p.z);
  return length(vec2(length(q.xy) - r1, q.z)) - r2;
}`,
  hexPrism: `float sdHexPrism(vec3 pIn, float r, float h) {
  const vec3 k = vec3(-0.8660254, 0.5, 0.57735);
  vec3 q0 = abs(vec3(pIn.x, pIn.z, pIn.y));
  float kd = 2.0 * min(dot(k.xy, q0.xy), 0.0);
  float qx = q0.x - kd * k.x;
  float qy = q0.y - kd * k.y;
  float d1 = length(vec2(qx - clamp(qx, -k.z * r, k.z * r), qy - r)) * sign(qy - r);
  float d2 = q0.z - h;
  return min(max(d1, d2), 0.0) + length(max(vec2(d1, d2), vec2(0.0)));
}`,
  triPrism: `float sdTriPrism(vec3 pIn, float r, float h) {
  vec3 p = vec3(pIn.x, pIn.z, pIn.y);
  vec3 q = abs(p);
  return max(q.z - h, max(q.x * 0.866025 + p.y * 0.5, -p.y) - r * 0.5);
}`,
  roundedCylinder: `float sdRoundedCylinder(vec3 p, float ra, float rb, float h) {
  vec2 d2 = vec2(length(p.xz) - ra + rb, abs(p.y) - h);
  return min(max(d2.x, d2.y), 0.0) + length(max(d2, vec2(0.0))) - rb;
}`,
  roundCone: `float sdRoundCone(vec3 p, float r1, float r2, float h) {
  float b = clamp((r1 - r2) / max(h, 0.0001), -0.999, 0.999);
  float a = sqrt(1.0 - b * b);
  vec2 q = vec2(length(p.xz), p.y);
  float k = dot(q, vec2(-b, a));
  if (k < 0.0) { return length(q) - r1; }
  if (k > a * h) { return length(q - vec2(0.0, h)) - r2; }
  return dot(q, vec2(a, b)) - r1;
}`,
  solidAngle: `float sdSolidAngle(vec3 p, float angleDeg, float ra) {
  float an = angleDeg * 0.017453292519943295;
  vec2 sc = vec2(sin(an), cos(an));
  vec2 q = vec2(length(p.xz), p.y);
  float l = length(q) - ra;
  float m = length(q - sc * clamp(dot(q, sc), 0.0, ra));
  return max(l, m * sign(sc.y * q.x - sc.x * q.y));
}`,
  cutSphere: `float sdCutSphere(vec3 p, float r, float hIn) {
  float h = clamp(hIn, -r * 0.999, r * 0.999);
  float w = sqrt(r * r - h * h);
  vec2 q = vec2(length(p.xz), p.y);
  float s = max((h - r) * q.x * q.x + w * w * (h + r - 2.0 * q.y), h * q.x - w * q.y);
  if (s < 0.0) { return length(q) - r; }
  if (q.x < w) { return h - q.y; }
  return length(q - vec2(w, h));
}`,
  cutHollowSphere: `float sdCutHollowSphere(vec3 p, float r, float hIn, float t) {
  float h = clamp(hIn, -r * 0.999, r * 0.999);
  float w = sqrt(r * r - h * h);
  vec2 q = vec2(length(p.xz), p.y);
  if (h * q.x < w * q.y) { return length(q - vec2(w, h)) - t; }
  return abs(length(q) - r) - t;
}`,
  deathStar: `float sdDeathStar(vec3 p, float ra, float rb, float dOff) {
  float dd = max(dOff, 0.0001);
  float a = (ra * ra - rb * rb + dd * dd) / (2.0 * dd);
  float b = sqrt(max(ra * ra - a * a, 0.0));
  vec2 q = vec2(p.y, length(p.xz));
  if (q.x * b - q.y * a > dd * max(b - q.y, 0.0)) { return length(q - vec2(a, b)); }
  return max(length(q) - ra, -(length(q - vec2(dd, 0.0)) - rb));
}`,
  rhombus: `float ndot2(vec2 a, vec2 b) {
  return a.x * b.x - a.y * b.y;
}

float sdRhombus(vec3 p, float la, float lb, float h, float ra) {
  vec3 q = abs(p);
  vec2 b = vec2(la, lb);
  float f = clamp(ndot2(b, b - 2.0 * q.xz) / dot(b, b), -1.0, 1.0);
  float qx = length(q.xz - 0.5 * b * vec2(1.0 - f, 1.0 + f)) * sign(q.x * b.y + q.z * b.x - b.x * b.y) - ra;
  float qy = q.y - h;
  return min(max(qx, qy), 0.0) + length(max(vec2(qx, qy), vec2(0.0)));
}`,
  octahedron: `float sdOctahedron(vec3 pIn, float s) {
  vec3 p = abs(pIn);
  float m = p.x + p.y + p.z - s;
  vec3 q;
  if (3.0 * p.x < m) { q = p.xyz; }
  else if (3.0 * p.y < m) { q = p.yzx; }
  else if (3.0 * p.z < m) { q = p.zxy; }
  else { return m * 0.57735027; }
  float k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
  return length(vec3(q.x, q.y - s + k, q.z - k));
}`,
  pyramid: `float sdPyramid(vec3 pIn, float base, float height) {
  float invB = 1.0 / max(base, 0.0001);
  vec3 p0 = pIn * invB;
  float h = height * invB;
  float m2 = h * h + 0.25;
  float px = abs(p0.x);
  float pz = abs(p0.z);
  if (pz > px) { float tmp = px; px = pz; pz = tmp; }
  px -= 0.5;
  pz -= 0.5;
  vec3 q = vec3(pz, h * p0.y - 0.5 * px, h * px + 0.5 * p0.y);
  if (q.z < 0.0 && p0.y > 0.0) {
    return -min(-q.z / sqrt(m2), p0.y) * base;
  }
  float dSlant = 1e10;
  if (q.z > 0.0) {
    float s = max(-q.x, 0.0);
    float t = clamp((q.y - 0.5 * pz) / (m2 + 0.25), 0.0, 1.0);
    float a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    float b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);
    float d2 = min(a, b);
    if (min(q.y, -q.x * m2 - q.y * 0.5) > 0.0) { d2 = 0.0; }
    dSlant = sqrt((d2 + q.z * q.z) / m2);
  }
  float dBase = 1e10;
  if (p0.y < 0.0) {
    dBase = length(vec3(max(px, 0.0), p0.y, max(pz, 0.0)));
  }
  return min(dSlant, dBase) * base;
}`,
  vesica: `float sdVesica(vec3 p, float r, float dIn) {
  float dd = clamp(dIn, 0.0001, r * 0.999);
  float b = sqrt(r * r - dd * dd);
  vec2 q = vec2(length(p.xz), abs(p.y));
  if ((q.y - b) * dd > q.x * b) { return length(q - vec2(0.0, b)); }
  return length(q - vec2(-dd, 0.0)) - r;
}`,
};

const TG_ROT = `const applyInvRotXYZ = (lp: d.v3f, rot: d.v3f): d.v3f => {
  "use gpu";
  const czn = std.cos(-rot.z); const szn = std.sin(-rot.z);
  const p1 = d.vec3f(czn * lp.x - szn * lp.y, szn * lp.x + czn * lp.y, lp.z);
  const cyn = std.cos(-rot.y); const syn = std.sin(-rot.y);
  const p2 = d.vec3f(cyn * p1.x + syn * p1.z, p1.y, -syn * p1.x + cyn * p1.z);
  const cxn = std.cos(-rot.x); const sxn = std.sin(-rot.x);
  return d.vec3f(p2.x, cxn * p2.y - sxn * p2.z, sxn * p2.y + cxn * p2.z);
};`;

const WGSL_ROT = `fn applyInvRotXYZ(lp: vec3f, rot: vec3f) -> vec3f {
  let czn = cos(-rot.z); let szn = sin(-rot.z);
  let p1 = vec3f(czn * lp.x - szn * lp.y, szn * lp.x + czn * lp.y, lp.z);
  let cyn = cos(-rot.y); let syn = sin(-rot.y);
  let p2 = vec3f(cyn * p1.x + syn * p1.z, p1.y, -syn * p1.x + cyn * p1.z);
  let cxn = cos(-rot.x); let sxn = sin(-rot.x);
  return vec3f(p2.x, cxn * p2.y - sxn * p2.z, sxn * p2.y + cxn * p2.z);
}`;

const GLSL_ROT = `vec3 applyInvRotXYZ(vec3 lp, vec3 rot) {
  float czn = cos(-rot.z); float szn = sin(-rot.z);
  vec3 p1 = vec3(czn * lp.x - szn * lp.y, szn * lp.x + czn * lp.y, lp.z);
  float cyn = cos(-rot.y); float syn = sin(-rot.y);
  vec3 p2 = vec3(cyn * p1.x + syn * p1.z, p1.y, -syn * p1.x + cyn * p1.z);
  float cxn = cos(-rot.x); float sxn = sin(-rot.x);
  return vec3(p2.x, cxn * p2.y - sxn * p2.z, sxn * p2.y + cxn * p2.z);
}`;

const TG_SMIN = `const smin = (a: number, b: number, k: number): number => {
  "use gpu";
  const h = std.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return std.mix(b, a, h) - k * h * (1.0 - h);
};`;

const WGSL_SMIN = `fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}`;

const GLSL_SMIN = `float smin(float a, float b, float k) {
  float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}`;

// ── Renderers ─────────────────────────────────────────────────────────────────

interface Syntax {
  vec3(v: V3): string;
  vec2(x: number, y: number): string;
  minFn: string;
  maxFn: string;
  divVec(e: string, v: V3): string;
  pointStmt(name: string, expr: string): string;
  defStmt(name: string, expr: string): string;
  assignStmt(name: string, expr: string): string;
  fnOpen(fn: string): string;
  fnClose: string;
  helpers: { shapes: Record<ShapeType, string>; rot: string; smin: string };
}

const IDENT = /^[A-Za-z_][A-Za-z0-9_.]*$/;
const paren = (e: string) => (IDENT.test(e) ? e : `(${e})`);
const neg = (e: string) => (IDENT.test(e) ? `-${e}` : `-(${e})`);

function shapeCall(syn: Syntax, rhs: ShapeRhs): string {
  let lp = rhs.point;
  if (rhs.pos) lp = `${lp} - ${syn.vec3(rhs.pos)}`;
  if (rhs.rotRad) {
    if (rhs.accScl) {
      lp = syn.divVec(`applyInvRotXYZ(${paren(lp)} * ${syn.vec3(rhs.accScl)}, ${syn.vec3(rhs.rotRad)})`, rhs.accScl);
    } else {
      lp = `applyInvRotXYZ(${lp}, ${syn.vec3(rhs.rotRad)})`;
    }
  }
  if (rhs.scale) lp = syn.divVec(lp, rhs.scale);

  const [p0, p1, p2, p3] = rhs.params;
  let call: string;
  switch (rhs.shape) {
    case "sphere":
      call = `sdSphere(${lp}, ${fmt(p0)})`;
      break;
    case "box":
      call = `sdBox(${lp}, ${syn.vec3([p0, p1, p2])})`;
      break;
    case "torus":
      call = `sdTorus(${lp}, ${syn.vec2(p0, p1)})`;
      break;
    case "roundedBox":
      call = `sdRoundedBox(${lp}, ${syn.vec3([p0, p1, p2])}, ${fmt(p3)})`;
      break;
    case "boxFrame":
      call = `sdBoxFrame(${lp}, ${syn.vec3([p0, p1, p2])}, ${fmt(p3)})`;
      break;
    case "rhombus":
      call = `sdRhombus(${lp}, ${fmt(p0)}, ${fmt(p1)}, ${fmt(p2)}, ${fmt(p3)})`;
      break;
    case "cappedTorus":
      call = `sdCappedTorus(${lp}, ${fmt(p0)}, ${fmt(p1)}, ${fmt(p2)})`;
      break;
    case "link":
      call = `sdLink(${lp}, ${fmt(p0)}, ${fmt(p1)}, ${fmt(p2)})`;
      break;
    case "roundedCylinder":
      call = `sdRoundedCylinder(${lp}, ${fmt(p0)}, ${fmt(p1)}, ${fmt(p2)})`;
      break;
    case "roundCone":
      call = `sdRoundCone(${lp}, ${fmt(p0)}, ${fmt(p1)}, ${fmt(p2)})`;
      break;
    case "cutHollowSphere":
      call = `sdCutHollowSphere(${lp}, ${fmt(p0)}, ${fmt(p1)}, ${fmt(p2)})`;
      break;
    case "deathStar":
      call = `sdDeathStar(${lp}, ${fmt(p0)}, ${fmt(p1)}, ${fmt(p2)})`;
      break;
    case "octahedron":
      call = `sdOctahedron(${lp}, ${fmt(p0)})`;
      break;
    case "cylinder":
    case "capsule":
    case "cone":
    case "hexPrism":
    case "triPrism":
    case "solidAngle":
    case "cutSphere":
    case "pyramid":
    case "vesica": {
      const fn = `sd${rhs.shape[0].toUpperCase()}${rhs.shape.slice(1)}`;
      call = `${fn}(${lp}, ${fmt(p0)}, ${fmt(p1)})`;
      break;
    }
  }
  return Math.abs(rhs.factor - 1) > 1e-9 ? `${call} * ${fmt(rhs.factor)}` : call;
}

function rhsExpr(syn: Syntax, rhs: Rhs): string {
  return rhs.kind === "var" ? rhs.name : shapeCall(syn, rhs);
}

function combineExpr(syn: Syntax, acc: string, op: OpType, k: number, b: string): string {
  switch (op) {
    case "union":
      return `${syn.minFn}(${acc}, ${b})`;
    case "subtract":
      return `${syn.maxFn}(${acc}, ${neg(b)})`;
    case "intersect":
      return `${syn.maxFn}(${acc}, ${b})`;
    case "sUnion":
      return `smin(${acc}, ${b}, ${fmt(k)})`;
    case "sSubtract":
      return `-smin(${neg(acc)}, ${b}, ${fmt(k)})`;
    case "sIntersect":
      return `-smin(${neg(acc)}, ${neg(b)}, ${fmt(k)})`;
  }
}

function pointExpr(syn: Syntax, stmt: Extract<Stmt, { t: "point" }>): string {
  let e = stmt.from;
  if (stmt.pos) e = `${e} - ${syn.vec3(stmt.pos)}`;
  if (stmt.rotRad) e = `applyInvRotXYZ(${e}, ${syn.vec3(stmt.rotRad)})`;
  if (stmt.scale) e = syn.divVec(e, stmt.scale);
  return e;
}

function render(ir: Ir, fnName: string, syn: Syntax): string {
  if (!ir.result) return PLACEHOLDER;

  const body: string[] = [];
  // Single shape → return the expression directly, no accumulator.
  if (ir.stmts.length === 1 && ir.stmts[0].t === "def") {
    body.push(`  return ${rhsExpr(syn, ir.stmts[0].rhs)};`);
  } else {
    for (const stmt of ir.stmts) {
      if (stmt.t === "point") body.push(syn.pointStmt(stmt.name, pointExpr(syn, stmt)));
      else if (stmt.t === "def") body.push(syn.defStmt(stmt.name, rhsExpr(syn, stmt.rhs)));
      else body.push(syn.assignStmt(stmt.name, combineExpr(syn, stmt.name, stmt.op, stmt.k, rhsExpr(syn, stmt.rhs))));
    }
    body.push(`  return ${ir.result};`);
  }

  const helpers: string[] = [];
  for (const shape of ALL_SHAPE_TYPES) {
    if (ir.usedShapes.has(shape)) helpers.push(syn.helpers.shapes[shape]);
  }
  if (ir.usesRotation) helpers.push(syn.helpers.rot);
  if (ir.usesSmin) helpers.push(syn.helpers.smin);

  return [...helpers, `${syn.fnOpen(fnName)}\n${body.join("\n")}\n${syn.fnClose}`].join("\n\n");
}

const TG_SYNTAX: Syntax = {
  vec3: (v) => `d.vec3f(${fmt(v[0])}, ${fmt(v[1])}, ${fmt(v[2])})`,
  vec2: (x, y) => `d.vec2f(${fmt(x)}, ${fmt(y)})`,
  minFn: "std.min",
  maxFn: "std.max",
  divVec: (e, v) => `${paren(e)} * d.vec3f(${fmtRecip(v[0])}, ${fmtRecip(v[1])}, ${fmtRecip(v[2])})`,
  pointStmt: (n, e) => `  const ${n} = ${e};`,
  defStmt: (n, e) => `  let ${n} = ${e};`,
  assignStmt: (n, e) => `  ${n} = ${e};`,
  fnOpen: (fn) => `const ${fn} = (p: d.v3f): number => {\n  "use gpu";`,
  fnClose: "};",
  helpers: { shapes: TG_HELPERS, rot: TG_ROT, smin: TG_SMIN },
};

const WGSL_SYNTAX: Syntax = {
  vec3: (v) => `vec3f(${fmt(v[0])}, ${fmt(v[1])}, ${fmt(v[2])})`,
  vec2: (x, y) => `vec2f(${fmt(x)}, ${fmt(y)})`,
  minFn: "min",
  maxFn: "max",
  divVec: (e, v) => `${paren(e)} / vec3f(${fmt(v[0])}, ${fmt(v[1])}, ${fmt(v[2])})`,
  pointStmt: (n, e) => `  let ${n} = ${e};`,
  defStmt: (n, e) => `  var ${n} = ${e};`,
  assignStmt: (n, e) => `  ${n} = ${e};`,
  fnOpen: (fn) => `fn ${fn}(p: vec3f) -> f32 {`,
  fnClose: "}",
  helpers: { shapes: WGSL_HELPERS, rot: WGSL_ROT, smin: WGSL_SMIN },
};

const GLSL_SYNTAX: Syntax = {
  vec3: (v) => `vec3(${fmt(v[0])}, ${fmt(v[1])}, ${fmt(v[2])})`,
  vec2: (x, y) => `vec2(${fmt(x)}, ${fmt(y)})`,
  minFn: "min",
  maxFn: "max",
  divVec: (e, v) => `${paren(e)} / vec3(${fmt(v[0])}, ${fmt(v[1])}, ${fmt(v[2])})`,
  pointStmt: (n, e) => `  vec3 ${n} = ${e};`,
  defStmt: (n, e) => `  float ${n} = ${e};`,
  assignStmt: (n, e) => `  ${n} = ${e};`,
  fnOpen: (fn) => `float ${fn}(vec3 p) {`,
  fnClose: "}",
  helpers: { shapes: GLSL_HELPERS, rot: GLSL_ROT, smin: GLSL_SMIN },
};

// ── Public API ────────────────────────────────────────────────────────────────

const RESERVED = new Set([
  ...ALL_SHAPE_TYPES.map((t) => `sd${t[0].toUpperCase()}${t.slice(1)}`),
  "smin",
  "applyInvRotXYZ",
  "ndot2",
]);

function sdfFnName(nodeName: string): string {
  const base = nodeName
    .replace(/[^a-zA-Z0-9]+/g, " ")
    .trim()
    .split(" ")
    .filter(Boolean)
    .map((w) => w[0].toUpperCase() + w.slice(1))
    .join("");
  let fn = `sd${base || "Model"}`;
  if (RESERVED.has(fn)) fn += "Model";
  return fn;
}

export function generateSdfCode(root: SceneRoot, selectedItemId: string | null): GeneratedSdf {
  const selected = selectedItemId ? findItemIn(root.items, selectedItemId) : null;
  const fnName = sdfFnName(selected ? selected.name : root.name);
  const ir = buildIr(root, selectedItemId);
  return {
    fnName,
    typegpu: render(ir, fnName, TG_SYNTAX),
    wgsl: render(ir, fnName, WGSL_SYNTAX),
    glsl: render(ir, fnName, GLSL_SYNTAX),
  };
}
