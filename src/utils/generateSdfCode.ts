import type {
  ObjectGroup,
  OpType,
  SceneItem,
  SceneRoot,
  ShapeLayer,
  ShapeType,
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

  const [p0, p1, p2] = rhs.params;
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
    case "cylinder":
      call = `sdCylinder(${lp}, ${fmt(p0)}, ${fmt(p1)})`;
      break;
    case "capsule":
      call = `sdCapsule(${lp}, ${fmt(p0)}, ${fmt(p1)})`;
      break;
    case "cone":
      call = `sdCone(${lp}, ${fmt(p0)}, ${fmt(p1)})`;
      break;
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
  for (const shape of ["sphere", "box", "torus", "cylinder", "capsule", "cone"] as ShapeType[]) {
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
  "sdSphere",
  "sdBox",
  "sdTorus",
  "sdCylinder",
  "sdCapsule",
  "sdCone",
  "smin",
  "applyInvRotXYZ",
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
