import type { CsgNode, SceneObject, ShapeType, OpType } from "../store/sceneStore";

function fmtFloat(n: number): string {
  const s = parseFloat(n.toFixed(6)).toString();
  return s.includes(".") ? s : s + ".0";
}

function vec3(x: number, y: number, z: number, prefix: "d." | ""): string {
  const f = fmtFloat;
  return `${prefix}vec3f(${f(x)}, ${f(y)}, ${f(z)})`;
}

function vec2(x: number, y: number, prefix: "d." | ""): string {
  const f = fmtFloat;
  return `${prefix}vec2f(${f(x)}, ${f(y)})`;
}

// ── Shape SDF helper sources ──────────────────────────────────────────────────

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
  let q = vec2f(length(vec2f(p.x, p.z)) - t.x, p.y);
  return length(q) - t.y;
}`,
  cylinder: `fn sdCylinder(p: vec3f, r: f32, h: f32) -> f32 {
  let d2 = vec2f(length(vec2f(p.x, p.z)) - r, abs(p.y) - h);
  return min(max(d2.x, d2.y), 0.0) + length(max(d2, vec2f(0.0)));
}`,
  capsule: `fn sdCapsule(p: vec3f, r: f32, h: f32) -> f32 {
  let py = clamp(p.y, -h, h);
  return length(p - vec3f(0.0, py, 0.0)) - r;
}`,
  cone: `fn sdCone(p: vec3f, r: f32, h: f32) -> f32 {
  let q = vec2f(length(vec2f(p.x, p.z)), p.y);
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

// ── Rotation helpers ──────────────────────────────────────────────────────────

const TG_ROTATION_HELPER = `const applyInvRotXYZ = (lp: d.v3f, rot: d.v3f): d.v3f => {
  "use gpu";
  const czn = std.cos(-rot.z); const szn = std.sin(-rot.z);
  const p1 = d.vec3f(czn * lp.x - szn * lp.y, szn * lp.x + czn * lp.y, lp.z);
  const cyn = std.cos(-rot.y); const syn = std.sin(-rot.y);
  const p2 = d.vec3f(cyn * p1.x + syn * p1.z, p1.y, -syn * p1.x + cyn * p1.z);
  const cxn = std.cos(-rot.x); const sxn = std.sin(-rot.x);
  return d.vec3f(p2.x, cxn * p2.y - sxn * p2.z, sxn * p2.y + cxn * p2.z);
};`;

const WGSL_ROTATION_HELPER = `fn applyInvRotXYZ(lp: vec3f, rot: vec3f) -> vec3f {
  let czn = cos(-rot.z); let szn = sin(-rot.z);
  let p1 = vec3f(czn * lp.x - szn * lp.y, szn * lp.x + czn * lp.y, lp.z);
  let cyn = cos(-rot.y); let syn = sin(-rot.y);
  let p2 = vec3f(cyn * p1.x + syn * p1.z, p1.y, -syn * p1.x + cyn * p1.z);
  let cxn = cos(-rot.x); let sxn = sin(-rot.x);
  return vec3f(p2.x, cxn * p2.y - sxn * p2.z, sxn * p2.y + cxn * p2.z);
}`;

// ── CSG operation helpers ─────────────────────────────────────────────────────

const TG_SMIN = `const smin = (a: number, b: number, k: number): number => {
  "use gpu";
  const h = std.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return std.mix(b, a, h) - k * h * (1.0 - h);
};`;

const WGSL_SMIN = `fn smin(a: f32, b: f32, k: f32) -> f32 {
  let h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
  return mix(b, a, h) - k * h * (1.0 - h);
}`;

// ── Recursive expression generators ──────────────────────────────────────────

function collectShapeTypes(node: CsgNode, out: Set<ShapeType>): void {
  if (node.kind === "shape") {
    out.add(node.shapeType);
    return;
  }
  collectShapeTypes(node.left, out);
  collectShapeTypes(node.right, out);
}

function collectUsesRotation(node: CsgNode): boolean {
  if (node.kind === "shape") {
    return node.rotation.some((a) => Math.abs(a) > 1e-9);
  }
  return collectUsesRotation(node.left) || collectUsesRotation(node.right);
}

function needsSmin(node: CsgNode): boolean {
  if (node.kind === "shape") return false;
  if (["sUnion", "sSubtract", "sIntersect"].includes(node.op)) return true;
  return needsSmin(node.left) || needsSmin(node.right);
}

function tgShapeExpr(node: CsgNode & { kind: "shape" }, prefix: string): string {
  const pos = vec3(...node.position, "d.");
  const [p0, p1, p2] = node.params;
  const DEG_TO_RAD = Math.PI / 180;
  const hasRotation = node.rotation.some((a) => Math.abs(a) > 1e-9);
  const lp = hasRotation
    ? `applyInvRotXYZ(p - ${pos}, ${vec3(node.rotation[0] * DEG_TO_RAD, node.rotation[1] * DEG_TO_RAD, node.rotation[2] * DEG_TO_RAD, "d.")})`
    : `p - ${pos}`;
  switch (node.shapeType) {
    case "sphere":
      return `sdSphere(${lp}, ${fmtFloat(p0)})`;
    case "box":
      return `sdBox(${lp}, ${vec3(p0, p1, p2, "d.")})`;
    case "torus":
      return `sdTorus(${lp}, ${vec2(p0, p1, "d.")})`;
    case "cylinder":
      return `sdCylinder(${lp}, ${fmtFloat(p0)}, ${fmtFloat(p1)})`;
    case "capsule":
      return `sdCapsule(${lp}, ${fmtFloat(p0)}, ${fmtFloat(p1)})`;
    case "cone":
      return `sdCone(${lp}, ${fmtFloat(p0)}, ${fmtFloat(p1)})`;
  }
  return prefix; // unreachable
}

function tgOpExpr(op: OpType, a: string, b: string, k: number): string {
  switch (op) {
    case "union":     return `std.min(${a}, ${b})`;
    case "subtract":  return `std.max(${a}, -(${b}))`;
    case "intersect": return `std.max(${a}, ${b})`;
    case "sUnion":    return `smin(${a}, ${b}, ${fmtFloat(k)})`;
    case "sSubtract": return `-smin(-(${a}), ${b}, ${fmtFloat(k)})`;
    case "sIntersect":return `-smin(-(${a}), -(${b}), ${fmtFloat(k)})`;
  }
}

function tgNodeExpr(node: CsgNode): string {
  if (node.kind === "shape") return tgShapeExpr(node, "");
  const a = tgNodeExpr(node.left);
  const b = tgNodeExpr(node.right);
  return tgOpExpr(node.op, a, b, node.smoothK);
}

function wgslShapeExpr(node: CsgNode & { kind: "shape" }): string {
  const pos = vec3(...node.position, "");
  const [p0, p1, p2] = node.params;
  const DEG_TO_RAD = Math.PI / 180;
  const hasRotation = node.rotation.some((a) => Math.abs(a) > 1e-9);
  const lp = hasRotation
    ? `applyInvRotXYZ(p - ${pos}, ${vec3(node.rotation[0] * DEG_TO_RAD, node.rotation[1] * DEG_TO_RAD, node.rotation[2] * DEG_TO_RAD, "")})`
    : `p - ${pos}`;
  switch (node.shapeType) {
    case "sphere":
      return `sdSphere(${lp}, ${fmtFloat(p0)})`;
    case "box":
      return `sdBox(${lp}, ${vec3(p0, p1, p2, "")})`;
    case "torus":
      return `sdTorus(${lp}, ${vec2(p0, p1, "")})`;
    case "cylinder":
      return `sdCylinder(${lp}, ${fmtFloat(p0)}, ${fmtFloat(p1)})`;
    case "capsule":
      return `sdCapsule(${lp}, ${fmtFloat(p0)}, ${fmtFloat(p1)})`;
    case "cone":
      return `sdCone(${lp}, ${fmtFloat(p0)}, ${fmtFloat(p1)})`;
  }
  return "";
}

function wgslOpExpr(op: OpType, a: string, b: string, k: number): string {
  switch (op) {
    case "union":     return `min(${a}, ${b})`;
    case "subtract":  return `max(${a}, -(${b}))`;
    case "intersect": return `max(${a}, ${b})`;
    case "sUnion":    return `smin(${a}, ${b}, ${fmtFloat(k)})`;
    case "sSubtract": return `-smin(-(${a}), ${b}, ${fmtFloat(k)})`;
    case "sIntersect":return `-smin(-(${a}), -(${b}), ${fmtFloat(k)})`;
  }
}

function wgslNodeExpr(node: CsgNode): string {
  if (node.kind === "shape") return wgslShapeExpr(node);
  const a = wgslNodeExpr(node.left);
  const b = wgslNodeExpr(node.right);
  return wgslOpExpr(node.op, a, b, node.smoothK);
}

// ── Main export ───────────────────────────────────────────────────────────────

export interface GeneratedCode {
  typegpu: string;
  wgsl: string;
}

const PLACEHOLDER = "// Add objects with shapes to generate code";

export function generateSdScene(objects: SceneObject[]): GeneratedCode {
  const activeObjects = objects.filter((o) => o.root !== null);
  if (activeObjects.length === 0) {
    return { typegpu: PLACEHOLDER, wgsl: PLACEHOLDER };
  }

  // Collect used shape types and whether smooth ops / rotation are used
  const usedTypes = new Set<ShapeType>();
  let usesSmin = false;
  let usesRotation = false;
  for (const obj of activeObjects) {
    if (obj.root) {
      collectShapeTypes(obj.root, usedTypes);
      if (needsSmin(obj.root)) usesSmin = true;
      if (collectUsesRotation(obj.root)) usesRotation = true;
    }
  }

  // TypeGPU output
  const tgHelpers = [...usedTypes].map((t) => TG_HELPERS[t]).join("\n\n");
  const tgSmin = usesSmin ? "\n\n" + TG_SMIN : "";
  const tgRotation = usesRotation ? "\n\n" + TG_ROTATION_HELPER : "";

  // One function per object, then a combined sdScene
  const tgObjectFns = activeObjects
    .map((obj, i) => {
      const expr = tgNodeExpr(obj.root!);
      return `const sdObject${i} = (p: d.v3f): number => {\n  "use gpu";\n  return ${expr};\n};`;
    })
    .join("\n\n");

  const tgSceneCalls = activeObjects
    .map((_, i) => `  dist = std.min(dist, sdObject${i}(p));`)
    .join("\n");

  const typegpu =
    `${tgHelpers}${tgSmin}${tgRotation}\n\n${tgObjectFns}\n\n` +
    `const sdScene = (p: d.v3f): number => {\n  "use gpu";\n  let dist = d.f32(1e10);\n${tgSceneCalls}\n  return dist;\n};`;

  // WGSL output
  const wgslHelpers = [...usedTypes].map((t) => WGSL_HELPERS[t]).join("\n\n");
  const wgslSmin = usesSmin ? "\n\n" + WGSL_SMIN : "";
  const wgslRotation = usesRotation ? "\n\n" + WGSL_ROTATION_HELPER : "";

  const wgslObjectFns = activeObjects
    .map((obj, i) => {
      const expr = wgslNodeExpr(obj.root!);
      return `fn sdObject${i}(p: vec3f) -> f32 {\n  return ${expr};\n}`;
    })
    .join("\n\n");

  const wgslSceneCalls = activeObjects
    .map((_, i) => `  dist = min(dist, sdObject${i}(p));`)
    .join("\n");

  const wgsl =
    `${wgslHelpers}${wgslSmin}${wgslRotation}\n\n${wgslObjectFns}\n\n` +
    `fn sdScene(p: vec3f) -> f32 {\n  var dist: f32 = 1e10;\n${wgslSceneCalls}\n  return dist;\n}`;

  return { typegpu, wgsl };
}
