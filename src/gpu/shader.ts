import tgpu, { d, std } from "typegpu";
import { fullScreenTriangle } from "typegpu/common";
import {
  MAX_GPU_OBJECTS,
  MAX_INSTRUCTIONS,
  MAX_PICK_INSTRUCTIONS,
  MAX_PICK_OBJECTS,
} from "../store/sceneStore";

type TgpuRoot = Awaited<ReturnType<typeof tgpu.init>>;

export {
  MAX_GPU_OBJECTS,
  MAX_INSTRUCTIONS,
  MAX_PICK_INSTRUCTIONS,
  MAX_PICK_OBJECTS,
};

export const OPCODE_PUSH_SHAPE = 0;
export const OPCODE_OP = 1;

export const SHAPE_TYPE_INT = {
  sphere: 0,
  box: 1,
  torus: 2,
  cylinder: 3,
  capsule: 4,
  cone: 5,
  roundedBox: 6,
  boxFrame: 7,
  cappedTorus: 8,
  link: 9,
  hexPrism: 10,
  triPrism: 11,
  roundedCylinder: 12,
  roundCone: 13,
  solidAngle: 14,
  cutSphere: 15,
  cutHollowSphere: 16,
  deathStar: 17,
  rhombus: 18,
  octahedron: 19,
  pyramid: 20,
  vesica: 21,
} as const;

export const OP_TYPE_INT = {
  union: 0,
  subtract: 1,
  intersect: 2,
  sUnion: 3,
  sSubtract: 4,
  sIntersect: 5,
} as const;

// Each instruction is PUSH_SHAPE (0) or OP (1). Group/layer transforms are
// baked on the CPU into one inverse affine map per shape (see bake.ts):
//   q = (row0·p, row1·p, row2·p) + offset;  dist = sd(q, params) * factor
// Layout: 4×u32/f32 header (16 B) + vec4f (16 B) + 4×(vec3f + f32) (64 B)
//         + boundCenter/Radius (16 B) = 112 B
const Instruction = d.struct({
  opcode: d.u32, // 0=PUSH_SHAPE, 1=OP
  shapeType: d.u32, // for PUSH_SHAPE: SHAPE_TYPE_INT value
  opType: d.u32, // for OP: 0=union,1=subtract,2=intersect,3=sUnion,4=sSubtract,5=sIntersect
  smoothK: d.f32, // for smooth OPs
  params: d.vec4f, // for PUSH_SHAPE: shape params
  row0: d.vec3f, // rows of the baked inverse affine matrix
  factor: d.f32, // conservative distance factor: min(scale)·min(accScl)
  row1: d.vec3f,
  unionOnly: d.u32, // 1 = every CSG op above this shape is hard union
  row2: d.vec3f,
  _pad2: d.f32,
  offset: d.vec3f, // baked translation: M·(b − shapePos)
  _pad3: d.f32,
  boundCenter: d.vec3f, // world-space bounding sphere of this shape
  boundRadius: d.f32,
});

const ObjectInfo = d.struct({
  start: d.u32, // index into flat instruction buffer
  count: d.u32, // number of instructions for this object
});

const CameraUniforms = d.struct({
  time: d.f32,
  aspect: d.f32,
  mouse: d.vec2f,
  distance: d.f32,
});

const SceneUniforms = d.struct({
  objectCount: d.u32,
  boundsRadius: d.f32, // conservative world-space bounding sphere of scene
  renderMode: d.u32, // RENDER_MODE_CLASSIC | RENDER_MODE_CHROME
  _pad1: d.f32,
  boundsCenter: d.vec3f,
  _pad2: d.f32,
});

const SelectionUniforms = d.struct({
  enabled: d.u32,
  usesSceneSdf: d.u32,
  count: d.u32,
  boundsRadius: d.f32, // conservative world bounding sphere of selected subtree
  boundsCenter: d.vec3f,
  _pad: d.f32,
});

const GizmoUniforms = d.struct({
  enabled: d.u32,
  activeAxis: d.u32,
  mode: d.u32,
  _pad0: d.u32,
  position: d.vec3f,
  _pad: d.f32,
  scale: d.f32,
  axisX: d.vec3f,
  _padX: d.f32,
  axisY: d.vec3f,
  _padY: d.f32,
  axisZ: d.vec3f,
  _padZ: d.f32,
});

const PickUniforms = d.struct({
  objectCount: d.u32,
  pickPass: d.u32,
});

const QualityUniforms = d.struct({
  maxSteps: d.f32,
  reflSteps: d.f32,
  outlineSteps: d.f32,
  _pad: d.f32,
});

const TOTAL_INSTRUCTIONS = MAX_INSTRUCTIONS;

const emptyInstruction = {
  opcode: 0,
  shapeType: 0,
  opType: 0,
  smoothK: 0,
  params: d.vec4f(0, 0, 0, 0),
  row0: d.vec3f(1, 0, 0),
  factor: 0,
  row1: d.vec3f(0, 1, 0),
  unionOnly: 0,
  row2: d.vec3f(0, 0, 1),
  _pad2: 0,
  offset: d.vec3f(0, 0, 0),
  _pad3: 0,
  boundCenter: d.vec3f(0, 0, 0),
  boundRadius: 0,
};

const emptyObjectInfo = { start: 0, count: 0 };

export const OUTLINE_OFFSET = 0.01;
export const OUTLINE_BAND = 0.01;
const OUTLINE_STRENGTH = 0.5;
const OUTLINE_RIM_POWER = 1.0;
const OUTLINE_GRAD_LO = 1.06;
const OUTLINE_GRAD_HI = 1.45;
const OUTLINE_EDGE_LO = 0.25;
const OUTLINE_EDGE_HI = 0.85;
const RAY_MISS_T = 50.0;
/**
 * Sphere-skip only above this. March hits at `0.0001*t` / reflection `0.001*t`;
 * returning a smaller sphereDist paints the bounding sphere as the surface.
 */
const SHAPE_BOUND_SKIP = 0.15;
/** Surface tie epsilon for pick — prefer smaller CSG subtree (more specific item). */
const PICK_TIE_EPS = 0.002;

/** Pick pass: ray-march translate gizmo, output axis id 1/2/3 in R. */
export const PICK_PASS_GIZMO = 2;

/** Cheap lit shading (no reflection bounce, iteration-based AO). */
export const RENDER_MODE_CLASSIC = 0;
/** Reflective chrome shading (reflection march + AO + tonemap). */
export const RENDER_MODE_CHROME = 1;

export function createShader(root: TgpuRoot) {
  const cameraUniforms = root.createUniform(CameraUniforms, {
    time: 0,
    aspect: 1,
    mouse: d.vec2f(0.3, -0.4),
    distance: 2.5,
  });
  const sceneUniforms = root.createUniform(SceneUniforms, {
    objectCount: 0,
    boundsRadius: 0,
    renderMode: RENDER_MODE_CHROME,
    _pad1: 0,
    boundsCenter: d.vec3f(0, 0, 0),
    _pad2: 0,
  });
  const selectionUniforms = root.createUniform(SelectionUniforms, {
    enabled: 0,
    usesSceneSdf: 0,
    count: 0,
    boundsRadius: 0,
    boundsCenter: d.vec3f(0, 0, 0),
    _pad: 0,
  });
  const gizmoUniforms = root.createUniform(GizmoUniforms, {
    enabled: 0,
    activeAxis: 0,
    mode: 0,
    _pad0: 0,
    position: d.vec3f(0.0, 0.0, 0.0),
    _pad: 0,
    scale: 0.3,
    axisX: d.vec3f(1.0, 0.0, 0.0),
    _padX: 0,
    axisY: d.vec3f(0.0, 1.0, 0.0),
    _padY: 0,
    axisZ: d.vec3f(0.0, 0.0, 1.0),
    _padZ: 0,
  });

  const instructionsBuffer = root.createReadonly(
    d.arrayOf(Instruction, TOTAL_INSTRUCTIONS),
    Array.from({ length: TOTAL_INSTRUCTIONS }, () => ({ ...emptyInstruction })),
  );

  const objectInfoBuffer = root.createReadonly(
    d.arrayOf(ObjectInfo, MAX_GPU_OBJECTS),
    Array.from({ length: MAX_GPU_OBJECTS }, () => ({ ...emptyObjectInfo })),
  );

  const selectionInstructionsBuffer = root.createReadonly(
    d.arrayOf(Instruction, MAX_INSTRUCTIONS),
    Array.from({ length: MAX_INSTRUCTIONS }, () => ({
      ...emptyInstruction,
    })),
  );

  const pickUniforms = root.createUniform(PickUniforms, {
    objectCount: 0,
    pickPass: 0,
  });
  const qualityUniforms = root.createUniform(QualityUniforms, {
    maxSteps: 48,
    reflSteps: 24,
    outlineSteps: 24,
    _pad: 0,
  });

  const pickInstructionsBuffer = root.createReadonly(
    d.arrayOf(Instruction, MAX_PICK_INSTRUCTIONS),
    Array.from({ length: MAX_PICK_INSTRUCTIONS }, () => ({
      ...emptyInstruction,
    })),
  );

  const pickObjectInfoBuffer = root.createReadonly(
    d.arrayOf(ObjectInfo, MAX_PICK_OBJECTS),
    Array.from({ length: MAX_PICK_OBJECTS }, () => ({ ...emptyObjectInfo })),
  );

  // ── SDF primitives ────────────────────────────────────────────────────────

  const sdSphere = (p: d.v3f, R: number): number => {
    "use gpu";
    return std.length(p) - R;
  };

  const sdBox = (p: d.v3f, b: d.v3f): number => {
    "use gpu";
    const q = std.abs(p) - b;
    return (
      std.length(std.max(q, d.vec3f(0.0))) +
      std.min(std.max(q.x, std.max(q.y, q.z)), 0.0)
    );
  };

  const sdTorus = (p: d.v3f, t: d.v2f): number => {
    "use gpu";
    const q = d.vec2f(std.length(d.vec2f(p.x, p.z)) - t.x, p.y);
    return std.length(q) - t.y;
  };

  const sdCylinder = (p: d.v3f, r: number, h: number): number => {
    "use gpu";
    const d2 = d.vec2f(std.length(d.vec2f(p.x, p.z)) - r, std.abs(p.y) - h);
    return (
      std.min(std.max(d2.x, d2.y), 0.0) + std.length(std.max(d2, d.vec2f(0.0)))
    );
  };

  const sdCapsule = (p: d.v3f, r: number, h: number): number => {
    "use gpu";
    const py = std.clamp(p.y, -h, h);
    return std.length(p - d.vec3f(0.0, py, 0.0)) - r;
  };

  const sdCone = (p: d.v3f, r: number, h: number): number => {
    "use gpu";
    const q = d.vec2f(std.length(d.vec2f(p.x, p.z)), p.y);
    const k1 = d.vec2f(0.0, h);
    const k2 = d.vec2f(-r, 2.0 * h);
    let capR = d.f32(0.0);
    if (p.y < 0.0) {
      capR = d.f32(r);
    }
    const ca = d.vec2f(q.x - std.min(q.x, capR), std.abs(p.y) - h);
    const t = std.clamp(std.dot(k1 - q, k2) / std.dot(k2, k2), 0.0, 1.0);
    const cb = q - k1 + k2 * t;
    let s = d.f32(1.0);
    if (cb.x < 0.0) {
      if (ca.y < 0.0) {
        s = d.f32(-1.0);
      }
    }
    return s * std.sqrt(std.min(std.dot(ca, ca), std.dot(cb, cb)));
  };

  const sdRoundedBox = (p: d.v3f, b: d.v3f, r: number): number => {
    "use gpu";
    const q = std.abs(p) - b;
    return (
      std.length(std.max(q, d.vec3f(0.0))) +
      std.min(std.max(q.x, std.max(q.y, q.z)), 0.0) -
      r
    );
  };

  const sdBoxFrame = (pIn: d.v3f, b: d.v3f, e: number): number => {
    "use gpu";
    const p = std.abs(pIn) - b;
    const q = std.abs(p + d.vec3f(e)) - d.vec3f(e);
    const dA =
      std.length(std.max(d.vec3f(p.x, q.y, q.z), d.vec3f(0.0))) +
      std.min(std.max(p.x, std.max(q.y, q.z)), 0.0);
    const dB =
      std.length(std.max(d.vec3f(q.x, p.y, q.z), d.vec3f(0.0))) +
      std.min(std.max(q.x, std.max(p.y, q.z)), 0.0);
    const dC =
      std.length(std.max(d.vec3f(q.x, q.y, p.z), d.vec3f(0.0))) +
      std.min(std.max(q.x, std.max(q.y, p.z)), 0.0);
    return std.min(dA, std.min(dB, dC));
  };

  // Angle params arrive in degrees (UI convention, same as rotation).
  const sdCappedTorus = (
    pIn: d.v3f,
    ra: number,
    rb: number,
    angleDeg: number,
  ): number => {
    "use gpu";
    const an = angleDeg * 0.017453292519943295;
    const sc = d.vec2f(std.sin(an), std.cos(an));
    // IQ's version is z-up; remap to this project's y-up torus orientation.
    const p = d.vec3f(std.abs(pIn.x), pIn.z, pIn.y);
    let k = std.length(d.vec2f(p.x, p.y));
    if (sc.y * p.x > sc.x * p.y) {
      k = std.dot(d.vec2f(p.x, p.y), sc);
    }
    return std.sqrt(std.dot(p, p) + ra * ra - 2.0 * ra * k) - rb;
  };

  const sdLink = (p: d.v3f, le: number, r1: number, r2: number): number => {
    "use gpu";
    const q = d.vec3f(p.x, std.max(std.abs(p.y) - le, 0.0), p.z);
    return (
      std.length(d.vec2f(std.length(d.vec2f(q.x, q.y)) - r1, q.z)) - r2
    );
  };

  const sdHexPrism = (pIn: d.v3f, r: number, h: number): number => {
    "use gpu";
    const k = d.vec3f(-0.8660254, 0.5, 0.57735);
    // y-up: prism axis along y (IQ's is along z).
    const q0 = std.abs(d.vec3f(pIn.x, pIn.z, pIn.y));
    const kd = 2.0 * std.min(std.dot(d.vec2f(k.x, k.y), d.vec2f(q0.x, q0.y)), 0.0);
    const qx = q0.x - kd * k.x;
    const qy = q0.y - kd * k.y;
    const d1 =
      std.length(d.vec2f(qx - std.clamp(qx, -k.z * r, k.z * r), qy - r)) *
      std.sign(qy - r);
    const d2 = q0.z - h;
    return (
      std.min(std.max(d1, d2), 0.0) +
      std.length(std.max(d.vec2f(d1, d2), d.vec2f(0.0)))
    );
  };

  const sdTriPrism = (pIn: d.v3f, r: number, h: number): number => {
    "use gpu";
    const p = d.vec3f(pIn.x, pIn.z, pIn.y);
    const q = std.abs(p);
    return std.max(
      q.z - h,
      std.max(q.x * 0.866025 + p.y * 0.5, -p.y) - r * 0.5,
    );
  };

  const sdRoundedCylinder = (
    p: d.v3f,
    ra: number,
    rb: number,
    h: number,
  ): number => {
    "use gpu";
    const d2 = d.vec2f(
      std.length(d.vec2f(p.x, p.z)) - ra + rb,
      std.abs(p.y) - h,
    );
    return (
      std.min(std.max(d2.x, d2.y), 0.0) +
      std.length(std.max(d2, d.vec2f(0.0))) -
      rb
    );
  };

  const sdRoundCone = (p: d.v3f, r1: number, r2: number, h: number): number => {
    "use gpu";
    // Clamp keeps sqrt real when |r1 - r2| >= h (user-editable params).
    const b = std.clamp((r1 - r2) / std.max(h, 0.0001), -0.999, 0.999);
    const a = std.sqrt(1.0 - b * b);
    const q = d.vec2f(std.length(d.vec2f(p.x, p.z)), p.y);
    const k = std.dot(q, d.vec2f(-b, a));
    let result = std.dot(q, d.vec2f(a, b)) - r1;
    if (k < 0.0) {
      result = std.length(q) - r1;
    } else if (k > a * h) {
      result = std.length(q - d.vec2f(0.0, h)) - r2;
    }
    return result;
  };

  const sdSolidAngle = (p: d.v3f, angleDeg: number, ra: number): number => {
    "use gpu";
    const an = angleDeg * 0.017453292519943295;
    const sc = d.vec2f(std.sin(an), std.cos(an));
    const q = d.vec2f(std.length(d.vec2f(p.x, p.z)), p.y);
    const l = std.length(q) - ra;
    const m = std.length(q - sc * std.clamp(std.dot(q, sc), 0.0, ra));
    return std.max(l, m * std.sign(sc.y * q.x - sc.x * q.y));
  };

  const sdCutSphere = (p: d.v3f, r: number, hIn: number): number => {
    "use gpu";
    const h = std.clamp(hIn, -r * 0.999, r * 0.999);
    const w = std.sqrt(r * r - h * h);
    const q = d.vec2f(std.length(d.vec2f(p.x, p.z)), p.y);
    const s = std.max(
      (h - r) * q.x * q.x + w * w * (h + r - 2.0 * q.y),
      h * q.x - w * q.y,
    );
    let result = std.length(q - d.vec2f(w, h));
    if (s < 0.0) {
      result = std.length(q) - r;
    } else if (q.x < w) {
      result = h - q.y;
    }
    return result;
  };

  const sdCutHollowSphere = (
    p: d.v3f,
    r: number,
    hIn: number,
    t: number,
  ): number => {
    "use gpu";
    const h = std.clamp(hIn, -r * 0.999, r * 0.999);
    const w = std.sqrt(r * r - h * h);
    const q = d.vec2f(std.length(d.vec2f(p.x, p.z)), p.y);
    let result = std.abs(std.length(q) - r);
    if (h * q.x < w * q.y) {
      result = std.length(q - d.vec2f(w, h));
    }
    return result - t;
  };

  const sdDeathStar = (p: d.v3f, ra: number, rb: number, dOff: number): number => {
    "use gpu";
    const dd = std.max(dOff, 0.0001);
    const a = (ra * ra - rb * rb + dd * dd) / (2.0 * dd);
    const b = std.sqrt(std.max(ra * ra - a * a, 0.0));
    // Cut opens along +y (IQ's is along +x).
    const q = d.vec2f(p.y, std.length(d.vec2f(p.x, p.z)));
    let result = std.max(
      std.length(q) - ra,
      -(std.length(q - d.vec2f(dd, 0.0)) - rb),
    );
    if (q.x * b - q.y * a > dd * std.max(b - q.y, 0.0)) {
      result = std.length(q - d.vec2f(a, b));
    }
    return result;
  };

  const ndot2 = (a: d.v2f, b: d.v2f): number => {
    "use gpu";
    return a.x * b.x - a.y * b.y;
  };

  const sdRhombus = (
    p: d.v3f,
    la: number,
    lb: number,
    h: number,
    ra: number,
  ): number => {
    "use gpu";
    const q = std.abs(p);
    const b = d.vec2f(la, lb);
    const f = std.clamp(
      ndot2(b, b - d.vec2f(2.0 * q.x, 2.0 * q.z)) / std.dot(b, b),
      -1.0,
      1.0,
    );
    const qx =
      std.length(
        d.vec2f(q.x, q.z) - b * d.vec2f(1.0 - f, 1.0 + f) * 0.5,
      ) *
        std.sign(q.x * b.y + q.z * b.x - b.x * b.y) -
      ra;
    const qy = q.y - h;
    return (
      std.min(std.max(qx, qy), 0.0) +
      std.length(std.max(d.vec2f(qx, qy), d.vec2f(0.0)))
    );
  };

  const sdOctahedron = (pIn: d.v3f, s: number): number => {
    "use gpu";
    const p = std.abs(pIn);
    const m = p.x + p.y + p.z - s;
    let result = m * 0.57735027;
    let q = d.vec3f(0.0);
    let onEdge = d.f32(0.0);
    if (3.0 * p.x < m) {
      q = d.vec3f(p.x, p.y, p.z);
      onEdge = d.f32(1.0);
    } else if (3.0 * p.y < m) {
      q = d.vec3f(p.y, p.z, p.x);
      onEdge = d.f32(1.0);
    } else if (3.0 * p.z < m) {
      q = d.vec3f(p.z, p.x, p.y);
      onEdge = d.f32(1.0);
    }
    if (onEdge > 0.5) {
      const k = std.clamp(0.5 * (q.z - q.y + s), 0.0, s);
      result = std.length(d.vec3f(q.x, q.y - s + k, q.z - k));
    }
    return result;
  };

  // IQ's unit-base pyramid, uniformly scaled so `base` is the footprint width.
  // IQ's original overestimates distance below the base plane (it only
  // measures to the slant faces), which makes rays tunnel through the bottom
  // and corrupts normals. Split: exact base-face distance for p.y < 0, slant
  // machinery only when outside the slant half-space, analytic interior.
  const sdPyramid = (pIn: d.v3f, base: number, height: number): number => {
    "use gpu";
    const invB = 1.0 / std.max(base, 0.0001);
    const p0 = d.vec3f(pIn.x * invB, pIn.y * invB, pIn.z * invB);
    const h = height * invB;
    const m2 = h * h + 0.25;
    let px = std.abs(p0.x);
    let pz = std.abs(p0.z);
    if (pz > px) {
      const tmp = px;
      px = pz;
      pz = tmp;
    }
    px = px - 0.5;
    pz = pz - 0.5;
    const q = d.vec3f(pz, h * p0.y - 0.5 * px, h * px + 0.5 * p0.y);

    // Fully inside: below the (folded) closest slant plane and above base.
    if (q.z < 0.0 && p0.y > 0.0) {
      return -std.min(-q.z / std.sqrt(m2), p0.y) * base;
    }

    let dSlant = d.f32(1e10);
    if (q.z > 0.0) {
      const s = std.max(-q.x, 0.0);
      const t = std.clamp((q.y - 0.5 * pz) / (m2 + 0.25), 0.0, 1.0);
      const a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
      const b =
        m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) +
        (q.y - m2 * t) * (q.y - m2 * t);
      let d2 = std.min(a, b);
      if (std.min(q.y, -q.x * m2 - q.y * 0.5) > 0.0) {
        d2 = d.f32(0.0);
      }
      dSlant = std.sqrt((d2 + q.z * q.z) / m2);
    }
    let dBase = d.f32(1e10);
    if (p0.y < 0.0) {
      dBase = std.length(
        d.vec3f(std.max(px, 0.0), p0.y, std.max(pz, 0.0)),
      );
    }
    return std.min(dSlant, dBase) * base;
  };

  // 2D vesica revolved around y — a lens/spindle solid.
  const sdVesica = (p: d.v3f, r: number, dIn: number): number => {
    "use gpu";
    const dd = std.clamp(dIn, 0.0001, r * 0.999);
    const b = std.sqrt(r * r - dd * dd);
    const q = d.vec2f(std.length(d.vec2f(p.x, p.z)), std.abs(p.y));
    let result = std.length(q - d.vec2f(-dd, 0.0)) - r;
    if ((q.y - b) * dd > q.x * b) {
      result = std.length(q - d.vec2f(0.0, b));
    }
    return result;
  };

  const divAccSclBy = (acc: d.v3f, s: d.v3f): d.v3f => {
    "use gpu";
    return d.vec3f(acc.x / s.x, acc.y / s.y, acc.z / s.z);
  };

  const mulVec3 = (a: d.v3f, b: d.v3f): d.v3f => {
    "use gpu";
    return d.vec3f(a.x * b.x, a.y * b.y, a.z * b.z);
  };

  // Apply a shape's baked inverse affine map: q = M·p + offset.
  // M composes all ancestor group transforms + the shape's own rotation/scale
  // (including the unsheared-rotation trick for non-uniform parent scale);
  // computed once per scene change on the CPU (bake.ts).
  const bakedLocalPoint = (
    p: d.v3f,
    row0: d.v3f,
    row1: d.v3f,
    row2: d.v3f,
    offset: d.v3f,
  ): d.v3f => {
    "use gpu";
    return (
      d.vec3f(std.dot(row0, p), std.dot(row1, p), std.dot(row2, p)) + offset
    );
  };

  const evalShape = (lp: d.v3f, shapeType: number, params: d.v4f): number => {
    "use gpu";
    let result = d.f32(1e10);
    if (shapeType === d.u32(0)) {
      result = sdSphere(lp, params.x);
    } else if (shapeType === d.u32(1)) {
      result = sdBox(lp, d.vec3f(params.x, params.y, params.z));
    } else if (shapeType === d.u32(2)) {
      result = sdTorus(lp, d.vec2f(params.x, params.y));
    } else if (shapeType === d.u32(3)) {
      result = sdCylinder(lp, params.x, params.y);
    } else if (shapeType === d.u32(4)) {
      result = sdCapsule(lp, params.x, params.y);
    } else if (shapeType === d.u32(5)) {
      result = sdCone(lp, params.x, params.y);
    } else if (shapeType === d.u32(6)) {
      result = sdRoundedBox(lp, d.vec3f(params.x, params.y, params.z), params.w);
    } else if (shapeType === d.u32(7)) {
      result = sdBoxFrame(lp, d.vec3f(params.x, params.y, params.z), params.w);
    } else if (shapeType === d.u32(8)) {
      result = sdCappedTorus(lp, params.x, params.y, params.z);
    } else if (shapeType === d.u32(9)) {
      result = sdLink(lp, params.x, params.y, params.z);
    } else if (shapeType === d.u32(10)) {
      result = sdHexPrism(lp, params.x, params.y);
    } else if (shapeType === d.u32(11)) {
      result = sdTriPrism(lp, params.x, params.y);
    } else if (shapeType === d.u32(12)) {
      result = sdRoundedCylinder(lp, params.x, params.y, params.z);
    } else if (shapeType === d.u32(13)) {
      result = sdRoundCone(lp, params.x, params.y, params.z);
    } else if (shapeType === d.u32(14)) {
      result = sdSolidAngle(lp, params.x, params.y);
    } else if (shapeType === d.u32(15)) {
      result = sdCutSphere(lp, params.x, params.y);
    } else if (shapeType === d.u32(16)) {
      result = sdCutHollowSphere(lp, params.x, params.y, params.z);
    } else if (shapeType === d.u32(17)) {
      result = sdDeathStar(lp, params.x, params.y, params.z);
    } else if (shapeType === d.u32(18)) {
      result = sdRhombus(lp, params.x, params.y, params.z, params.w);
    } else if (shapeType === d.u32(19)) {
      result = sdOctahedron(lp, params.x);
    } else if (shapeType === d.u32(20)) {
      result = sdPyramid(lp, params.x, params.y);
    } else if (shapeType === d.u32(21)) {
      result = sdVesica(lp, params.x, params.y);
    }
    return result;
  };

  // Polynomial smooth-min (IQ)
  const smin = (a: number, b: number, k: number): number => {
    "use gpu";
    const h = std.clamp(0.5 + (0.5 * (b - a)) / k, 0.0, 1.0);
    return std.mix(b, a, h) - k * h * (1.0 - h);
  };

  // Apply a CSG operation: a = left operand, b = right operand
  const applyOp = (a: number, b: number, opType: number, k: number): number => {
    "use gpu";
    let result = a;
    if (opType === d.u32(0)) {
      result = std.min(a, b); // union
    } else if (opType === d.u32(1)) {
      result = std.max(a, -b); // subtract: a minus b
    } else if (opType === d.u32(2)) {
      result = std.max(a, b); // intersect
    } else if (opType === d.u32(3)) {
      result = smin(a, b, k); // smooth union
    } else if (opType === d.u32(4)) {
      result = -smin(-a, b, k); // smooth subtract
    } else if (opType === d.u32(5)) {
      result = -smin(-a, -b, k); // smooth intersect
    }
    return result;
  };

  // Sphere dist is a conservative SDF lower bound. Safe to substitute when
  // every CSG op above this shape is hard union (min): zero-set unchanged.
  // Subtract/intersect/smooth paths keep a full eval (CPU sets unionOnly=0).
  // Skip only when sphereDist > SHAPE_BOUND_SKIP — otherwise the march hit
  // test (`dist < k*t`) treats the bounding sphere as the surface.
  const evalPushedShape = (
    p: d.v3f,
    unionOnly: number,
    boundCenter: d.v3f,
    boundRadius: number,
    row0: d.v3f,
    row1: d.v3f,
    row2: d.v3f,
    offset: d.v3f,
    shapeType: number,
    params: d.v4f,
    factor: number,
  ): number => {
    "use gpu";
    if (unionOnly === d.u32(1)) {
      const sphereDist = std.length(p - boundCenter) - boundRadius;
      if (sphereDist > SHAPE_BOUND_SKIP) {
        return sphereDist;
      }
    }
    const q = bakedLocalPoint(p, row0, row1, row2, offset);
    return evalShape(q, shapeType, params) * factor;
  };

  // ── Stack-machine sdScene ─────────────────────────────────────────────────
  //
  // The JS side compiles each CSG tree into a postorder instruction sequence
  // with all group/layer transforms baked into per-shape inverse affine maps
  // (bake.ts). PUSH_SHAPE evaluates a shape and pushes its SDF value; OP pops
  // two values, applies the operation, and pushes the result. After all
  // instructions the result sits in s0. Max stack depth for 15-node binary
  // tree is 8. Named variables s0..s7 + sp avoid the need for local WGSL arrays.
  const sdScene = (p: d.v3f): number => {
    "use gpu";
    let dist = d.f32(1e10);

    // SDF stack
    let s0 = d.f32(0.0);
    let s1 = d.f32(0.0);
    let s2 = d.f32(0.0);
    let s3 = d.f32(0.0);
    let s4 = d.f32(0.0);
    let s5 = d.f32(0.0);
    let s6 = d.f32(0.0);
    let s7 = d.f32(0.0);
    let sp = d.u32(0);

    const objCount = sceneUniforms.$.objectCount;
    for (let o = d.u32(0); o < objCount; o += d.u32(1)) {
      s0 = d.f32(0.0);
      s1 = d.f32(0.0);
      s2 = d.f32(0.0);
      s3 = d.f32(0.0);
      s4 = d.f32(0.0);
      s5 = d.f32(0.0);
      s6 = d.f32(0.0);
      s7 = d.f32(0.0);
      sp = d.u32(0);

      const info = objectInfoBuffer.$[o];
      const end = info.start + info.count;

      for (let i = info.start; i < end; i += d.u32(1)) {
        const instr = instructionsBuffer.$[i];

        if (instr.opcode === d.u32(0)) {
          const val = evalPushedShape(
            p,
            instr.unionOnly,
            instr.boundCenter,
            instr.boundRadius,
            instr.row0,
            instr.row1,
            instr.row2,
            instr.offset,
            instr.shapeType,
            instr.params,
            instr.factor,
          );
          if (sp === d.u32(0)) s0 = val;
          else if (sp === d.u32(1)) s1 = val;
          else if (sp === d.u32(2)) s2 = val;
          else if (sp === d.u32(3)) s3 = val;
          else if (sp === d.u32(4)) s4 = val;
          else if (sp === d.u32(5)) s5 = val;
          else if (sp === d.u32(6)) s6 = val;
          else s7 = val;
          sp += d.u32(1);
        } else if (instr.opcode === d.u32(1)) {
          sp -= d.u32(1);
          let b = d.f32(0.0);
          if (sp === d.u32(0)) b = s0;
          else if (sp === d.u32(1)) b = s1;
          else if (sp === d.u32(2)) b = s2;
          else if (sp === d.u32(3)) b = s3;
          else if (sp === d.u32(4)) b = s4;
          else if (sp === d.u32(5)) b = s5;
          else if (sp === d.u32(6)) b = s6;
          else b = s7;

          sp -= d.u32(1);
          let a = d.f32(0.0);
          if (sp === d.u32(0)) a = s0;
          else if (sp === d.u32(1)) a = s1;
          else if (sp === d.u32(2)) a = s2;
          else if (sp === d.u32(3)) a = s3;
          else if (sp === d.u32(4)) a = s4;
          else if (sp === d.u32(5)) a = s5;
          else if (sp === d.u32(6)) a = s6;
          else a = s7;

          const result = applyOp(a, b, instr.opType, instr.smoothK);
          if (sp === d.u32(0)) s0 = result;
          else if (sp === d.u32(1)) s1 = result;
          else if (sp === d.u32(2)) s2 = result;
          else if (sp === d.u32(3)) s3 = result;
          else if (sp === d.u32(4)) s4 = result;
          else if (sp === d.u32(5)) s5 = result;
          else if (sp === d.u32(6)) s6 = result;
          else s7 = result;
          sp += d.u32(1);
        }
      }

      dist = std.min(dist, s0);
    }

    return dist;
  };

  // Ray vs sphere: returns (tEnter, tExit); tExit < 0 = miss.
  const raySphere = (
    ro: d.v3f,
    rd: d.v3f,
    center: d.v3f,
    r: number,
  ): d.v2f => {
    "use gpu";
    const oc = ro - center;
    const b = std.dot(oc, rd);
    const h = b * b - (std.dot(oc, oc) - r * r);
    if (h < 0.0) {
      return d.vec2f(0.0, -1.0);
    }
    const sh = std.sqrt(h);
    return d.vec2f(std.max(-b - sh, 0.0), -b + sh);
  };

  // Ray vs scene bounding sphere.
  const rayBounds = (ro: d.v3f, rd: d.v3f): d.v2f => {
    "use gpu";
    return raySphere(
      ro,
      rd,
      sceneUniforms.$.boundsCenter,
      sceneUniforms.$.boundsRadius,
    );
  };

  // Evaluate SDF for an instruction range from the selection buffer.
  const evalSelectionInstructionRange = (
    p: d.v3f,
    start: number,
    count: number,
  ): number => {
    "use gpu";
    let s0 = d.f32(0.0);
    let s1 = d.f32(0.0);
    let s2 = d.f32(0.0);
    let s3 = d.f32(0.0);
    let s4 = d.f32(0.0);
    let s5 = d.f32(0.0);
    let s6 = d.f32(0.0);
    let s7 = d.f32(0.0);
    let sp = d.u32(0);

    const end = start + count;
    for (let i = start; i < end; i += d.u32(1)) {
      const instr = selectionInstructionsBuffer.$[i];

      if (instr.opcode === d.u32(0)) {
        const val = evalPushedShape(
          p,
          instr.unionOnly,
          instr.boundCenter,
          instr.boundRadius,
          instr.row0,
          instr.row1,
          instr.row2,
          instr.offset,
          instr.shapeType,
          instr.params,
          instr.factor,
        );
        if (sp === d.u32(0)) s0 = val;
        else if (sp === d.u32(1)) s1 = val;
        else if (sp === d.u32(2)) s2 = val;
        else if (sp === d.u32(3)) s3 = val;
        else if (sp === d.u32(4)) s4 = val;
        else if (sp === d.u32(5)) s5 = val;
        else if (sp === d.u32(6)) s6 = val;
        else s7 = val;
        sp += d.u32(1);
      } else if (instr.opcode === d.u32(1)) {
        sp -= d.u32(1);
        let b = d.f32(0.0);
        if (sp === d.u32(0)) b = s0;
        else if (sp === d.u32(1)) b = s1;
        else if (sp === d.u32(2)) b = s2;
        else if (sp === d.u32(3)) b = s3;
        else if (sp === d.u32(4)) b = s4;
        else if (sp === d.u32(5)) b = s5;
        else if (sp === d.u32(6)) b = s6;
        else b = s7;

        sp -= d.u32(1);
        let a = d.f32(0.0);
        if (sp === d.u32(0)) a = s0;
        else if (sp === d.u32(1)) a = s1;
        else if (sp === d.u32(2)) a = s2;
        else if (sp === d.u32(3)) a = s3;
        else if (sp === d.u32(4)) a = s4;
        else if (sp === d.u32(5)) a = s5;
        else if (sp === d.u32(6)) a = s6;
        else a = s7;

        const result = applyOp(a, b, instr.opType, instr.smoothK);
        if (sp === d.u32(0)) s0 = result;
        else if (sp === d.u32(1)) s1 = result;
        else if (sp === d.u32(2)) s2 = result;
        else if (sp === d.u32(3)) s3 = result;
        else if (sp === d.u32(4)) s4 = result;
        else if (sp === d.u32(5)) s5 = result;
        else if (sp === d.u32(6)) s6 = result;
        else s7 = result;
        sp += d.u32(1);
      }
    }

    return s0;
  };

  const sdSelection = (p: d.v3f): number => {
    "use gpu";
    return evalSelectionInstructionRange(
      p,
      d.u32(0),
      selectionUniforms.$.count,
    );
  };

  const evalPickInstructionRange = (
    p: d.v3f,
    start: number,
    count: number,
  ): number => {
    "use gpu";
    let s0 = d.f32(0.0);
    let s1 = d.f32(0.0);
    let s2 = d.f32(0.0);
    let s3 = d.f32(0.0);
    let s4 = d.f32(0.0);
    let s5 = d.f32(0.0);
    let s6 = d.f32(0.0);
    let s7 = d.f32(0.0);
    let sp = d.u32(0);

    const end = start + count;
    for (let i = start; i < end; i += d.u32(1)) {
      const instr = pickInstructionsBuffer.$[i];

      if (instr.opcode === d.u32(0)) {
        const val = evalPushedShape(
          p,
          instr.unionOnly,
          instr.boundCenter,
          instr.boundRadius,
          instr.row0,
          instr.row1,
          instr.row2,
          instr.offset,
          instr.shapeType,
          instr.params,
          instr.factor,
        );
        if (sp === d.u32(0)) s0 = val;
        else if (sp === d.u32(1)) s1 = val;
        else if (sp === d.u32(2)) s2 = val;
        else if (sp === d.u32(3)) s3 = val;
        else if (sp === d.u32(4)) s4 = val;
        else if (sp === d.u32(5)) s5 = val;
        else if (sp === d.u32(6)) s6 = val;
        else s7 = val;
        sp += d.u32(1);
      } else if (instr.opcode === d.u32(1)) {
        sp -= d.u32(1);
        let b = d.f32(0.0);
        if (sp === d.u32(0)) b = s0;
        else if (sp === d.u32(1)) b = s1;
        else if (sp === d.u32(2)) b = s2;
        else if (sp === d.u32(3)) b = s3;
        else if (sp === d.u32(4)) b = s4;
        else if (sp === d.u32(5)) b = s5;
        else if (sp === d.u32(6)) b = s6;
        else b = s7;

        sp -= d.u32(1);
        let a = d.f32(0.0);
        if (sp === d.u32(0)) a = s0;
        else if (sp === d.u32(1)) a = s1;
        else if (sp === d.u32(2)) a = s2;
        else if (sp === d.u32(3)) a = s3;
        else if (sp === d.u32(4)) a = s4;
        else if (sp === d.u32(5)) a = s5;
        else if (sp === d.u32(6)) a = s6;
        else a = s7;

        const result = applyOp(a, b, instr.opType, instr.smoothK);
        if (sp === d.u32(0)) s0 = result;
        else if (sp === d.u32(1)) s1 = result;
        else if (sp === d.u32(2)) s2 = result;
        else if (sp === d.u32(3)) s3 = result;
        else if (sp === d.u32(4)) s4 = result;
        else if (sp === d.u32(5)) s5 = result;
        else if (sp === d.u32(6)) s6 = result;
        else s7 = result;
        sp += d.u32(1);
      }
    }

    return s0;
  };

  const resolvePickObjectId = (p: d.v3f): number => {
    "use gpu";
    let bestId = d.u32(0);
    let bestDist = d.f32(1e9);
    let bestCount = d.u32(0xffffffff);
    const n = pickUniforms.$.objectCount;
    for (let o = d.u32(0); o < n; o += d.u32(1)) {
      const info = pickObjectInfoBuffer.$[o];
      const dval = evalPickInstructionRange(p, info.start, info.count);
      const ad = std.abs(dval);
      const closer = ad < bestDist;
      const tiedCloser =
        ad <= bestDist + PICK_TIE_EPS && info.count < bestCount;
      if (closer || tiedCloser) {
        bestDist = ad;
        bestId = o + d.u32(1);
        bestCount = info.count;
      }
    }
    return bestId;
  };

  const evalSelectionDist = (p: d.v3f): number => {
    "use gpu";
    if (selectionUniforms.$.usesSceneSdf === d.u32(1)) {
      return sdScene(p);
    }
    return sdSelection(p);
  };

  // Rim + creases on the inflated selection shell.
  // Tetrahedron gradient: 4 SDF evals instead of 6 (same trick as calcNormal).
  const selectionOutlineMask = (p: d.v3f, rd: d.v3f): number => {
    "use gpu";
    const e = 0.001;
    const k = d.vec2f(1.0, -1.0);
    const grad =
      d.vec3f(k.x, k.y, k.y) * evalSelectionDist(p + d.vec3f(e, -e, -e)) +
      d.vec3f(k.y, k.y, k.x) * evalSelectionDist(p + d.vec3f(-e, -e, e)) +
      d.vec3f(k.y, k.x, k.y) * evalSelectionDist(p + d.vec3f(-e, e, -e)) +
      d.vec3f(k.x, k.x, k.x) * evalSelectionDist(p + d.vec3f(e, e, e));
    const N = std.normalize(grad);
    const gradMag = std.length(grad) / (4.0 * e);
    const viewDir = std.normalize(d.vec3f(-rd.x, -rd.y, -rd.z));
    const rim = std.pow(1.0 - std.abs(std.dot(N, viewDir)), OUTLINE_RIM_POWER);
    const gradEdge = std.smoothstep(OUTLINE_GRAD_LO, OUTLINE_GRAD_HI, gradMag);
    const edge = std.max(rim, gradEdge);
    return std.smoothstep(OUTLINE_EDGE_LO, OUTLINE_EDGE_HI, edge);
  };

  const rayMarchSelectionOutline = (
    ro: d.v3f,
    rd: d.v3f,
    tScene: number,
  ): number => {
    "use gpu";
    // Tight gate: selection's own bounding sphere, not the whole scene.
    const bounds = raySphere(
      ro,
      rd,
      selectionUniforms.$.boundsCenter,
      selectionUniforms.$.boundsRadius,
    );
    if (bounds.y < 0.0) {
      return d.f32(RAY_MISS_T + 10.0);
    }
    // Outline only shows if closer than the scene hit; no point marching past it.
    const tFar = std.min(bounds.y, tScene + 0.01);
    let t = bounds.x;
    if (t > tFar) {
      return d.f32(RAY_MISS_T + 10.0);
    }
    for (
      let i = d.f32(0.0);
      i < qualityUniforms.$.outlineSteps;
      i += d.f32(1.0)
    ) {
      const p = ro + rd * t;
      const dSel = evalSelectionDist(p);
      const dist = std.abs(dSel - OUTLINE_OFFSET) - OUTLINE_BAND;
      t += dist;
      if (dist < 0.0001) {
        return t;
      }
      // Left the gate without converging — explicit miss. With the tight
      // selection sphere the exit point is near the object, where the mask
      // is non-zero, so returning t here would paint the sphere as a halo.
      if (t > tFar) {
        return d.f32(RAY_MISS_T + 10.0);
      }
    }
    // Ran out of steps while still inside: grazing the shell. Keep t so the
    // mask decides, matching pre-optimization behavior at the rim.
    return t;
  };

  const selectionOutlineColor = (): d.v3f => {
    "use gpu";
    const pulse = 0.75 + 0.25 * std.sin(cameraUniforms.$.time * 2.5);
    const v = pulse * OUTLINE_STRENGTH;
    return d.vec3f(v, v, v);
  };

  const GIZMO_ARROW_LEN = 0.85;
  const GIZMO_SHAFT_R = 0.035;
  const GIZMO_HEAD_R = 0.08;
  const GIZMO_HEAD_LEN = 0.18;
  const GIZMO_RING_MAJOR = 0.85;
  const GIZMO_RING_TUBE = 0.035;
  const GIZMO_MODE_ROTATE = 1;
  const GIZMO_MODE_SCALE = 2;
  /** Must match gizmo.ts GIZMO_SCALE_HEAD_HALF_RATIO / GIZMO_CENTER_HALF_RATIO. */
  const GIZMO_SCALE_HEAD_HALF = 0.07;
  const GIZMO_CENTER_HALF = 0.055;

  const sdCapsuleSeg = (p: d.v3f, a: d.v3f, b: d.v3f, r: number): number => {
    "use gpu";
    const pa = p - a;
    const ba = b - a;
    const h = std.clamp(std.dot(pa, ba) / std.dot(ba, ba), 0.0, 1.0);
    return std.length(pa - ba * h) - r;
  };

  const sdAxisArrow = (gp: d.v3f, dir: d.v3f, s: number): number => {
    "use gpu";
    const arrowLen = s * GIZMO_ARROW_LEN;
    const shaftR = s * GIZMO_SHAFT_R;
    const headR = s * GIZMO_HEAD_R;
    const headLen = s * GIZMO_HEAD_LEN;
    const tip = dir * arrowLen;
    const shaftEnd = dir * (arrowLen - headLen);
    const dShaft = sdCapsuleSeg(gp, d.vec3f(0.0), shaftEnd, shaftR);
    const dHead = std.length(gp - tip) - headR;
    return std.min(dShaft, dHead);
  };

  const evalTranslateGizmo = (p: d.v3f): d.v2f => {
    "use gpu";
    const origin = gizmoUniforms.$.position;
    const s = gizmoUniforms.$.scale;
    const gp = p - origin;
    const dx = sdAxisArrow(gp, d.vec3f(1.0, 0.0, 0.0), s);
    const dy = sdAxisArrow(gp, d.vec3f(0.0, 1.0, 0.0), s);
    const dz = sdAxisArrow(gp, d.vec3f(0.0, 0.0, 1.0), s);
    let best = dx;
    let axis = d.f32(1.0);
    if (dy < best) {
      best = dy;
      axis = d.f32(2.0);
    }
    if (dz < best) {
      best = dz;
      axis = d.f32(3.0);
    }
    return d.vec2f(best, axis);
  };

  const sdWorldAxisRing = (gp: d.v3f, axis: d.v3f, s: number): number => {
    "use gpu";
    const majorR = s * GIZMO_RING_MAJOR;
    const tubeR = s * GIZMO_RING_TUBE;
    const along = std.dot(gp, axis);
    const perp = gp - axis * along;
    const q = d.vec2f(std.length(perp) - majorR, along);
    return std.length(q) - tubeR;
  };

  const evalRotateGizmo = (p: d.v3f): d.v2f => {
    "use gpu";
    const origin = gizmoUniforms.$.position;
    const s = gizmoUniforms.$.scale;
    const gp = p - origin;
    const dx = sdWorldAxisRing(gp, gizmoUniforms.$.axisX, s);
    const dy = sdWorldAxisRing(gp, gizmoUniforms.$.axisY, s);
    const dz = sdWorldAxisRing(gp, gizmoUniforms.$.axisZ, s);
    let best = dx;
    let axis = d.f32(1.0);
    if (dy < best) {
      best = dy;
      axis = d.f32(2.0);
    }
    if (dz < best) {
      best = dz;
      axis = d.f32(3.0);
    }
    return d.vec2f(best, axis);
  };

  // Arrow with a cube head (scale-gizmo convention, distinct from the
  // translate arrows). Cube is oriented along the handle axis; the u/v basis
  // must match gizmo.ts ringPlaneBasis so CPU pick sees identical geometry.
  const sdScaleHandle = (gp: d.v3f, axis: d.v3f, s: number): number => {
    "use gpu";
    const arrowLen = s * GIZMO_ARROW_LEN;
    const shaftR = s * GIZMO_SHAFT_R;
    const half = s * GIZMO_SCALE_HEAD_HALF;

    let hint = d.vec3f(0.0, 1.0, 0.0);
    if (std.abs(axis.y) >= 0.9) {
      hint = d.vec3f(1.0, 0.0, 0.0);
    }
    const u = std.normalize(std.cross(hint, axis));
    const v = std.cross(axis, u);

    const dShaft = sdCapsuleSeg(gp, d.vec3f(0.0), axis * (arrowLen - half), shaftR);

    const rel = gp - axis * arrowLen;
    const local = d.vec3f(std.dot(rel, axis), std.dot(rel, u), std.dot(rel, v));
    const dHead = sdBox(local, d.vec3f(half, half, half));
    return std.min(dShaft, dHead);
  };

  const evalScaleGizmo = (p: d.v3f): d.v2f => {
    "use gpu";
    const origin = gizmoUniforms.$.position;
    const s = gizmoUniforms.$.scale;
    const gp = p - origin;
    const dx = sdScaleHandle(gp, gizmoUniforms.$.axisX, s);
    const dy = sdScaleHandle(gp, gizmoUniforms.$.axisY, s);
    const dz = sdScaleHandle(gp, gizmoUniforms.$.axisZ, s);
    const half = s * GIZMO_CENTER_HALF;
    const dc = sdBox(gp, d.vec3f(half, half, half));
    let best = dx;
    let axis = d.f32(1.0);
    if (dy < best) {
      best = dy;
      axis = d.f32(2.0);
    }
    if (dz < best) {
      best = dz;
      axis = d.f32(3.0);
    }
    if (dc < best) {
      best = dc;
      axis = d.f32(4.0);
    }
    return d.vec2f(best, axis);
  };

  const evalGizmo = (p: d.v3f): d.v2f => {
    "use gpu";
    if (gizmoUniforms.$.mode === d.u32(GIZMO_MODE_ROTATE)) {
      return evalRotateGizmo(p);
    }
    if (gizmoUniforms.$.mode === d.u32(GIZMO_MODE_SCALE)) {
      return evalScaleGizmo(p);
    }
    return evalTranslateGizmo(p);
  };

  const gizmoAxisColor = (axis: number): d.v3f => {
    "use gpu";
    const active = gizmoUniforms.$.activeAxis;
    const axisId = d.u32(std.round(axis));
    const isActive =
      (active === d.u32(1) && axisId === d.u32(1)) ||
      (active === d.u32(2) && axisId === d.u32(2)) ||
      (active === d.u32(3) && axisId === d.u32(3)) ||
      (active === d.u32(4) && axisId === d.u32(4));
    let full = d.vec3f(0.38, 0.58, 1.0);
    if (axisId === d.u32(1)) {
      full = d.vec3f(1.0, 0.3, 0.3);
    }
    if (axisId === d.u32(2)) {
      full = d.vec3f(0.3, 1.0, 0.3);
    }
    if (axisId === d.u32(4)) {
      full = d.vec3f(0.9, 0.9, 0.9);
    }
    if (isActive) {
      return full * 1.2;
    }
    // Resting: desaturated + dimmed so the gizmo sits with the muted UI.
    const gray = d.vec3f(0.62, 0.62, 0.62);
    return std.mix(full, gray, d.f32(0.35)) * 0.6;
  };

  const GIZMO_HIT_EPS = 0.00015;
  // Conservative bounding-sphere radius in units of gizmo scale. Max extents:
  // translate 0.85+0.08 (head), rotate 0.85+0.035 (tube),
  // scale 0.85+0.07·√3 (cube corner) — all ≤ 0.98.
  const GIZMO_BOUND_RADIUS = 1.05;
  // Self-check (runs on CPU at shader build): gate sphere must cover every
  // handle, otherwise the march silently clips geometry.
  const maxGizmoExtent = Math.max(
    GIZMO_ARROW_LEN + GIZMO_HEAD_R,
    GIZMO_RING_MAJOR + GIZMO_RING_TUBE,
    GIZMO_ARROW_LEN + GIZMO_SCALE_HEAD_HALF * Math.sqrt(3),
    GIZMO_CENTER_HALF * Math.sqrt(3),
  );
  if (maxGizmoExtent > GIZMO_BOUND_RADIUS) {
    throw new Error(
      `GIZMO_BOUND_RADIUS ${GIZMO_BOUND_RADIUS} < max handle extent ${maxGizmoExtent}`,
    );
  }

  const rayMarchGizmo = (ro: d.v3f, rd: d.v3f): d.v2f => {
    "use gpu";
    // Gizmo covers a few % of the screen; skip the 32-step march everywhere else.
    const bounds = raySphere(
      ro,
      rd,
      gizmoUniforms.$.position,
      gizmoUniforms.$.scale * GIZMO_BOUND_RADIUS,
    );
    if (bounds.y < 0.0) {
      return d.vec2f(RAY_MISS_T + 1.0, 0.0);
    }
    let t = bounds.x;
    let hitAxis = d.f32(0.0);
    let converged = false;
    for (let i = d.f32(0.0); i < d.f32(32.0); i += d.f32(1.0)) {
      const p = ro + rd * t;
      const g = evalGizmo(p);
      const dist = g.x;
      hitAxis = g.y;
      if (dist < GIZMO_HIT_EPS) {
        converged = true;
        break;
      }
      t += dist * 0.85;
      if (t > bounds.y) {
        break;
      }
    }
    if (!converged) {
      return d.vec2f(RAY_MISS_T + 1.0, 0.0);
    }
    return d.vec2f(t, hitAxis);
  };

  // ── Chrome shading ────────────────────────────────────────────────────────

  const acesTonemap = (x: d.v3f): d.v3f => {
    "use gpu";
    const num = mulVec3(x, x * 2.51 + d.vec3f(0.03));
    const den = mulVec3(x, x * 2.43 + d.vec3f(0.59)) + d.vec3f(0.14);
    return std.min(std.max(divAccSclBy(num, den), d.vec3f(0.0)), d.vec3f(1.0));
  };

  // Procedural studio environment: dark gray room + softbox strips.
  // ponytail: analytic bands instead of an HDRI texture — good enough for
  // chrome reflections; upgrade path is a real cubemap if art needs it.
  const envColor = (rd: d.v3f): d.v3f => {
    "use gpu";
    const up = rd.y * 0.5 + 0.5;
    let v = std.mix(0.015, 0.06, up);

    const horizLen = std.max(std.length(d.vec2f(rd.x, rd.z)), 0.0001);
    const hx = rd.x / horizLen;
    const hz = rd.z / horizLen;

    // Tall vertical softbox, front-left
    const a1 = std.max(hx * -0.7071 - hz * 0.7071, 0.0);
    const s1 =
      std.pow(a1, 22.0) * (1.0 - std.smoothstep(0.55, 0.9, std.abs(rd.y)));

    // Narrower vertical strip, right
    const a2 = std.max(hx * 0.866 - hz * 0.5, 0.0);
    const s2 =
      std.pow(a2, 40.0) * (1.0 - std.smoothstep(0.5, 0.85, std.abs(rd.y)));

    // Broad overhead softbox
    const sTop = std.smoothstep(0.35, 0.85, rd.y);

    // Faint floor bounce so downward faces aren't dead black
    const sBot = std.smoothstep(0.3, 0.9, -rd.y);

    v += s1 * 3.2 + s2 * 2.2 + sTop * 1.4 + sBot * 0.05;
    return d.vec3f(v, v, v);
  };

  // 5-tap SDF ambient occlusion (IQ style)
  const calcAO = (p: d.v3f, n: d.v3f): number => {
    "use gpu";
    let occ = d.f32(0.0);
    let sca = d.f32(1.0);
    for (let i = d.f32(1.0); i <= d.f32(3.0); i += d.f32(1.0)) {
      const hr = 0.01 + 0.12 * (i / 3.0);
      const dd = sdScene(p + n * hr);
      occ += (hr - dd) * sca;
      sca *= 0.65;
    }
    return std.clamp(1.0 - 1.6 * occ, 0.0, 1.0);
  };

  // Returns (t, lastDist). lastDist is the SDF value at the hit point,
  // reused by calcNormalCheap so the forward-difference normal needs only
  // 3 extra sdScene taps instead of calcNormal's 4. Breaks before stepping
  // (unlike the primary rayMarch) so t and lastDist describe the same point;
  // the old post-step point could land inside the surface.
  const rayMarchReflection = (ro: d.v3f, rd: d.v3f): d.v2f => {
    "use gpu";
    const bounds = rayBounds(ro, rd);
    if (bounds.y < 0.0) {
      return d.vec2f(100.0, 0.0);
    }
    let t = bounds.x;
    let dist = d.f32(0.0);
    for (
      let i = d.f32(0.0);
      i < qualityUniforms.$.reflSteps;
      i += d.f32(1.0)
    ) {
      const p = ro + rd * t;
      dist = sdScene(p);
      if (dist < 0.001 * std.max(t, 1.0)) {
        break;
      }
      t += dist;
      if (t > bounds.y) {
        return d.vec2f(100.0, 0.0);
      }
    }
    return d.vec2f(t, dist);
  };

  // Cheap normal for reflection hits: forward difference against the SDF
  // value d0 already computed by the reflection march — 3 sdScene evals
  // instead of calcNormal's 4. Subtracting d0 cancels the "hit point is not
  // exactly on the surface" offset, so this matches the tetrahedral normal
  // bit-exact in practice (verified via screenshot diff).
  // ponytail: only feeds envColor of the second bounce; upgrade path is full
  // calcNormal if reflections ever shade more than env.
  const calcNormalCheap = (p: d.v3f, d0: number): d.v3f => {
    "use gpu";
    const eps = 0.001;
    return std.normalize(
      d.vec3f(
        sdScene(p + d.vec3f(eps, 0.0, 0.0)) - d0,
        sdScene(p + d.vec3f(0.0, eps, 0.0)) - d0,
        sdScene(p + d.vec3f(0.0, 0.0, eps)) - d0,
      ),
    );
  };

  const calcNormal = (p: d.v3f): d.v3f => {
    "use gpu";
    const eps = 0.001;
    const k = d.vec2f(1.0, -1.0);
    return std.normalize(
      d.vec3f(k.x, k.y, k.y) * sdScene(p + d.vec3f(eps, -eps, -eps)) +
        d.vec3f(k.y, k.y, k.x) * sdScene(p + d.vec3f(-eps, -eps, eps)) +
        d.vec3f(k.y, k.x, k.y) * sdScene(p + d.vec3f(-eps, eps, -eps)) +
        d.vec3f(k.x, k.x, k.x) * sdScene(p + d.vec3f(eps, eps, eps)),
    );
  };

  const rayMarch = (ro: d.v3f, rd: d.v3f): d.v2f => {
    "use gpu";
    // Bounding-sphere gate: rays that miss the scene skip marching entirely,
    // rays that hit start at the sphere entry instead of the camera.
    const bounds = rayBounds(ro, rd);
    if (bounds.y < 0.0) {
      return d.vec2f(RAY_MISS_T + 10.0, 0.0);
    }
    let t = bounds.x;
    let iterations = d.f32(0.0);
    for (let i = d.f32(0.0); i < qualityUniforms.$.maxSteps; i += d.f32(1.0)) {
      const p = ro + rd * t;
      const dist = sdScene(p);
      iterations += 1.0;
      t += dist;
      if (dist < 0.0001 * std.max(t, 1.0)) {
        return d.vec2f(t, iterations);
      }
      if (t > bounds.y) {
        return d.vec2f(RAY_MISS_T + 10.0, iterations);
      }
    }
    // Out of steps without a surface: miss. Treating this as a hit paints a
    // noisy halo (potato = 24 steps, grazing rays never converge).
    return d.vec2f(RAY_MISS_T + 10.0, iterations);
  };

  const rot2D = (angle: number): d.m2x2f => {
    "use gpu";
    const c = std.cos(angle);
    const s = std.sin(angle);
    return d.mat2x2f(c, -s, s, c);
  };

  const fragment = ({ uv }: { uv: d.v2f }) => {
    "use gpu";

    const pickPass = pickUniforms.$.pickPass;
    const isScenePickPass = pickPass === d.u32(1);
    const isGizmoPickPass = pickPass === d.u32(2);
    const sampleUv = d.vec2f(uv.x, uv.y);
    const uvn = sampleUv * 2.0 - d.vec2f(1.0, 1.0);
    const uvCorrected = d.vec2f(uvn.x * cameraUniforms.$.aspect, uvn.y);
    let ro = d.vec3f(0, 0, 0 - cameraUniforms.$.distance);
    let rd = std.normalize(d.vec3f(uvCorrected.x, uvCorrected.y, 1.0));

    const rotH = rot2D(cameraUniforms.$.mouse.x);
    const roXZ = rotH * d.vec2f(ro.x, ro.z);
    ro = d.vec3f(roXZ.x, ro.y, roXZ.y);
    const rdXZ = rotH * d.vec2f(rd.x, rd.z);
    rd = d.vec3f(rdXZ.x, rd.y, rdXZ.y);

    const rotV = rot2D(cameraUniforms.$.mouse.y);
    const roYZ = rotV * d.vec2f(ro.y, ro.z);
    ro = d.vec3f(ro.x, roYZ.x, roYZ.y);
    const rdYZ = rotV * d.vec2f(rd.y, rd.z);
    rd = d.vec3f(rd.x, rdYZ.x, rdYZ.y);

    if (isGizmoPickPass) {
      if (gizmoUniforms.$.enabled !== d.u32(1)) {
        return d.vec4f(0.0, 0.0, 0.0, 0.0);
      }
      const gizmoPick = rayMarchGizmo(ro, rd);
      if (gizmoPick.x > RAY_MISS_T) {
        return d.vec4f(0.0, 0.0, 0.0, 0.0);
      }
      const idNorm = gizmoPick.y / 255.0;
      return d.vec4f(idNorm, idNorm, idNorm, 1.0);
    }

    if (isScenePickPass) {
      const tPick = rayMarch(ro, rd).x;
      if (tPick > RAY_MISS_T) {
        return d.vec4f(0.0, 0.0, 0.0, 0.0);
      }
      const hitP = ro + rd * tPick;
      const pickId = resolvePickObjectId(hitP);
      const idf = d.f32(pickId);
      const idNorm = idf / 255.0;
      return d.vec4f(idNorm, idNorm, idNorm, 1.0);
    }

    const sceneResult = rayMarch(ro, rd);
    const tScene = sceneResult.x;

    const sceneHit = tScene <= RAY_MISS_T;

    let tOutline = d.f32(RAY_MISS_T + 1.0);
    if (selectionUniforms.$.enabled === d.u32(1)) {
      tOutline = rayMarchSelectionOutline(ro, rd, tScene);
    }

    const outlineHit = tOutline <= RAY_MISS_T;
    const outlineVisible =
      selectionUniforms.$.enabled === d.u32(1) &&
      outlineHit &&
      (!sceneHit || tOutline < tScene - 0.0001);

    // Background: super-dark gray vertical gradient + vignette (sRGB direct)
    const gradT = std.clamp(uvn.y * 0.5 + 0.5, 0.0, 1.0);
    const bg = std.mix(
      d.vec3f(0.055, 0.055, 0.055), // #0E0E0E bottom
      d.vec3f(0.118, 0.118, 0.118), // #1E1E1E top
      gradT,
    );
    const vig = 1.0 - 0.35 * std.dot(uvCorrected, uvCorrected) * 0.5;
    let col = bg * std.clamp(vig, 0.0, 1.0);

    if (sceneHit) {
      const pos = ro + rd * tScene;
      const N = calcNormal(pos);
      const viewDir = d.vec3f(-rd.x, -rd.y, -rd.z);

      if (sceneUniforms.$.renderMode === d.u32(RENDER_MODE_CLASSIC)) {
        // Classic: single light, no secondary marches — cheap.
        const iterations = sceneResult.y;
        const lightDir = std.normalize(d.vec3f(2.0, 3.0, -1.0));
        const diff = std.max(std.dot(N, lightDir), 0.0);
        const fresnel = std.pow(1.0 - std.abs(std.dot(N, viewDir)), 3.0) * 0.5;
        const halfDir = std.normalize(lightDir + viewDir);
        const spec = std.pow(std.max(std.dot(N, halfDir), 0.0), 64.0) * 1.5;
        const ao = 1.0 - std.clamp(iterations / 64.0, 0.0, 1.0) * 0.4;
        const baseColor = d.vec3f(0.72, 0.72, 0.72);
        col = std.sqrt(
          std.max(
            (baseColor * (diff * 0.8 + 0.2) +
              d.vec3f(1.0, 1.0, 1.0) * spec +
              baseColor * fresnel) *
              ao,
            d.vec3f(0.0),
          ),
        );
      } else {
        // Chrome: one real reflection bounce sees the scene itself.
        const R = std.reflect(rd, N);
        const reflRo = pos + N * 0.003;
        let reflCol = envColor(R);
        if (qualityUniforms.$.reflSteps > d.f32(0.0)) {
          const refl = rayMarchReflection(reflRo, R);
          if (refl.x <= RAY_MISS_T) {
            const pRefl = reflRo + R * refl.x;
            const nRefl = calcNormalCheap(pRefl, refl.y);
            // ponytail: second bounce is env-only, no third march — invisible past bounce 2
            reflCol = envColor(std.reflect(R, nRefl)) * 0.85;
          }
        }

        const ndv = std.max(std.dot(N, viewDir), 0.0);
        const fresnel = 0.55 + 0.45 * std.pow(1.0 - ndv, 5.0);

        // Crisp key highlight for extra sparkle
        const lightDir = std.normalize(d.vec3f(-1.5, 2.5, -2.0));
        const halfDir = std.normalize(lightDir + viewDir);
        const spec = std.pow(std.max(std.dot(N, halfDir), 0.0), 220.0) * 2.5;

        const ao = calcAO(pos, N);
        const hdr =
          (reflCol * fresnel + d.vec3f(spec, spec, spec)) *
          std.mix(0.5, 1.0, ao);
        col = std.pow(
          acesTonemap(hdr),
          d.vec3f(1.0 / 2.2, 1.0 / 2.2, 1.0 / 2.2),
        );
      }
    }

    if (outlineVisible) {
      const outlinePos = ro + rd * tOutline;
      const edge = selectionOutlineMask(outlinePos, rd);
      col = col + selectionOutlineColor() * edge;
    }

    if (gizmoUniforms.$.enabled === d.u32(1)) {
      const gizmoResult = rayMarchGizmo(ro, rd);
      const tGizmo = gizmoResult.x;
      const gizmoHit = tGizmo <= RAY_MISS_T;
      if (gizmoHit) {
        const gizmoPos = ro + rd * tGizmo;
        const eps = 0.001;
        const gx =
          evalGizmo(gizmoPos + d.vec3f(eps, 0.0, 0.0)).x -
          evalGizmo(gizmoPos - d.vec3f(eps, 0.0, 0.0)).x;
        const gy =
          evalGizmo(gizmoPos + d.vec3f(0.0, eps, 0.0)).x -
          evalGizmo(gizmoPos - d.vec3f(0.0, eps, 0.0)).x;
        const gz =
          evalGizmo(gizmoPos + d.vec3f(0.0, 0.0, eps)).x -
          evalGizmo(gizmoPos - d.vec3f(0.0, 0.0, eps)).x;
        const gN = std.normalize(d.vec3f(gx, gy, gz));
        const viewDir = std.normalize(d.vec3f(-rd.x, -rd.y, -rd.z));
        const rim = std.pow(1.0 - std.abs(std.dot(gN, viewDir)), 1.5) * 0.35;
        const axisCol = gizmoAxisColor(gizmoResult.y);
        const gizmoLit = axisCol * (0.85 + rim);
        col = std.mix(col, gizmoLit, d.f32(0.92));
      }
    }

    return d.vec4f(col, 1.0);
  };

  const pipeline = root.createRenderPipeline({
    vertex: fullScreenTriangle,
    fragment,
  });

  return {
    pipeline,
    cameraUniforms,
    sceneUniforms,
    selectionUniforms,
    gizmoUniforms,
    instructionsBuffer,
    objectInfoBuffer,
    selectionInstructionsBuffer,
    pickInstructionsBuffer,
    pickObjectInfoBuffer,
    pickUniforms,
    qualityUniforms,
  };
}
