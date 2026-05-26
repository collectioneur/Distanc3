import type { ShapeInstance, ShapeType } from "../store/sceneStore";

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

// ── TypeGPU helper sources ────────────────────────────────────────────────────

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

// ── WGSL helper sources ───────────────────────────────────────────────────────

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

// ── Per-shape call-site generators ────────────────────────────────────────────

function tgCall(shape: ShapeInstance): string {
  const [px, py, pz] = shape.position;
  const [p0, p1, p2] = shape.params;
  const pos = vec3(px, py, pz, "d.");
  switch (shape.type) {
    case "sphere":
      return `  dist = std.min(dist, sdSphere(p - ${pos}, ${fmtFloat(p0)}));`;
    case "box":
      return `  dist = std.min(dist, sdBox(p - ${pos}, ${vec3(p0, p1, p2, "d.")}));`;
    case "torus":
      return `  dist = std.min(dist, sdTorus(p - ${pos}, ${vec2(p0, p1, "d.")}));`;
    case "cylinder":
      return `  dist = std.min(dist, sdCylinder(p - ${pos}, ${fmtFloat(p0)}, ${fmtFloat(p1)}));`;
    case "capsule":
      return `  dist = std.min(dist, sdCapsule(p - ${pos}, ${fmtFloat(p0)}, ${fmtFloat(p1)}));`;
    case "cone":
      return `  dist = std.min(dist, sdCone(p - ${pos}, ${fmtFloat(p0)}, ${fmtFloat(p1)}));`;
  }
}

function wgslCall(shape: ShapeInstance): string {
  const [px, py, pz] = shape.position;
  const [p0, p1, p2] = shape.params;
  const pos = vec3(px, py, pz, "");
  switch (shape.type) {
    case "sphere":
      return `  dist = min(dist, sdSphere(p - ${pos}, ${fmtFloat(p0)}));`;
    case "box":
      return `  dist = min(dist, sdBox(p - ${pos}, ${vec3(p0, p1, p2, "")}));`;
    case "torus":
      return `  dist = min(dist, sdTorus(p - ${pos}, ${vec2(p0, p1, "")}));`;
    case "cylinder":
      return `  dist = min(dist, sdCylinder(p - ${pos}, ${fmtFloat(p0)}, ${fmtFloat(p1)}));`;
    case "capsule":
      return `  dist = min(dist, sdCapsule(p - ${pos}, ${fmtFloat(p0)}, ${fmtFloat(p1)}));`;
    case "cone":
      return `  dist = min(dist, sdCone(p - ${pos}, ${fmtFloat(p0)}, ${fmtFloat(p1)}));`;
  }
}

// ── Main export ───────────────────────────────────────────────────────────────

export interface GeneratedCode {
  typegpu: string;
  wgsl: string;
}

const PLACEHOLDER = "// Add shapes to generate code";

export function generateSdScene(shapes: ShapeInstance[]): GeneratedCode {
  if (shapes.length === 0) {
    return { typegpu: PLACEHOLDER, wgsl: PLACEHOLDER };
  }

  const usedTypes = [...new Set(shapes.map((s) => s.type))] as ShapeType[];

  const tgHelpers = usedTypes.map((t) => TG_HELPERS[t]).join("\n\n");
  const tgCalls = shapes.map(tgCall).join("\n");
  const typegpu = `${tgHelpers}\n\nconst sdScene = (p: d.v3f): number => {\n  "use gpu";\n  let dist = d.f32(1e10);\n${tgCalls}\n  return dist;\n};`;

  const wgslHelpers = usedTypes.map((t) => WGSL_HELPERS[t]).join("\n\n");
  const wgslCalls = shapes.map(wgslCall).join("\n");
  const wgsl = `${wgslHelpers}\n\nfn sdScene(p: vec3f) -> f32 {\n  var dist: f32 = 1e10;\n${wgslCalls}\n  return dist;\n}`;

  return { typegpu, wgsl };
}
