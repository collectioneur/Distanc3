import tgpu, { d, std } from "typegpu";
import { fullScreenTriangle } from "typegpu/common";
import { MAX_OBJECTS, MAX_NODES_PER_OBJECT } from "../store/sceneStore";

type TgpuRoot = Awaited<ReturnType<typeof tgpu.init>>;

export { MAX_OBJECTS, MAX_NODES_PER_OBJECT };

export const SHAPE_TYPE_INT = {
  sphere: 0,
  box: 1,
  torus: 2,
  cylinder: 3,
  capsule: 4,
  cone: 5,
} as const;

export const OP_TYPE_INT = {
  union: 0,
  subtract: 1,
  intersect: 2,
  sUnion: 3,
  sSubtract: 4,
  sIntersect: 5,
} as const;

// Each instruction is either PUSH_SHAPE (opcode=0) or OP (opcode=1).
// Layout: 4×u32/f32 header (16 bytes) + vec3f+pad (16 bytes) + vec4f (16 bytes) + vec3f+pad (16 bytes) = 64 bytes
const Instruction = d.struct({
  opcode: d.u32, // 0 = PUSH_SHAPE, 1 = OP
  shapeType: d.u32, // for PUSH_SHAPE: 0-5
  opType: d.u32, // for OP: 0=union,1=subtract,2=intersect,3=sUnion,4=sSubtract,5=sIntersect
  smoothK: d.f32, // for smooth OPs
  position: d.vec3f, // for PUSH_SHAPE
  _pad: d.f32, // alignment padding after vec3f
  params: d.vec4f, // for PUSH_SHAPE shape parameters
  rotation: d.vec3f, // for PUSH_SHAPE: Euler XYZ angles in radians
  _pad2: d.f32, // alignment padding after vec3f
});

const ObjectInfo = d.struct({
  start: d.u32, // index into flat instruction buffer
  count: d.u32, // number of instructions for this object
});

const TOTAL_INSTRUCTIONS = MAX_OBJECTS * MAX_NODES_PER_OBJECT;

const emptyInstruction = {
  opcode: 0,
  shapeType: 0,
  opType: 0,
  smoothK: 0,
  position: d.vec3f(0, 0, 0),
  _pad: 0,
  params: d.vec4f(0, 0, 0, 0),
  rotation: d.vec3f(0, 0, 0),
  _pad2: 0,
};

const emptyObjectInfo = { start: 0, count: 0 };

const OUTLINE_OFFSET = 0.01;
const OUTLINE_BAND = 0.01;
const OUTLINE_STRENGTH = 0.5;
const OUTLINE_RIM_POWER = 1.0;
const OUTLINE_GRAD_LO = 1.06;
const OUTLINE_GRAD_HI = 1.45;
const OUTLINE_EDGE_LO = 0.25;
const OUTLINE_EDGE_HI = 0.85;
const RAY_MISS_T = 50.0;

export function createShader(root: TgpuRoot) {
  const timeUniform = root.createUniform(d.f32, 0);
  const aspectUniform = root.createUniform(d.f32, 1);
  const mouseUniform = root.createUniform(d.vec2f, d.vec2f(0.3, -0.4));
  const distanceUniform = root.createUniform(d.f32, 2.5);
  const objectCountUniform = root.createUniform(d.u32, 0);
  const renderModeUniform = root.createUniform(d.u32, 0);
  const selectionEnabledUniform = root.createUniform(d.u32, 0);
  const selectionCountUniform = root.createUniform(d.u32, 0);

  const instructionsBuffer = root.createReadonly(
    d.arrayOf(Instruction, TOTAL_INSTRUCTIONS),
    Array.from({ length: TOTAL_INSTRUCTIONS }, () => ({ ...emptyInstruction })),
  );

  const objectInfoBuffer = root.createReadonly(
    d.arrayOf(ObjectInfo, MAX_OBJECTS),
    Array.from({ length: MAX_OBJECTS }, () => ({ ...emptyObjectInfo })),
  );

  const selectionInstructionsBuffer = root.createReadonly(
    d.arrayOf(Instruction, MAX_NODES_PER_OBJECT),
    Array.from({ length: MAX_NODES_PER_OBJECT }, () => ({
      ...emptyInstruction,
    })),
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

  const applyInvRotXYZ = (lp: d.v3f, rot: d.v3f): d.v3f => {
    "use gpu";
    const czn = std.cos(-rot.z);
    const szn = std.sin(-rot.z);
    const p1 = d.vec3f(czn * lp.x - szn * lp.y, szn * lp.x + czn * lp.y, lp.z);
    const cyn = std.cos(-rot.y);
    const syn = std.sin(-rot.y);
    const p2 = d.vec3f(cyn * p1.x + syn * p1.z, p1.y, -syn * p1.x + cyn * p1.z);
    const cxn = std.cos(-rot.x);
    const sxn = std.sin(-rot.x);
    return d.vec3f(p2.x, cxn * p2.y - sxn * p2.z, sxn * p2.y + cxn * p2.z);
  };

  const evalShape = (lp: d.v3f, shapeType: d.u32, params: d.v4f): number => {
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
  const applyOp = (a: number, b: number, opType: d.u32, k: number): number => {
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

  // ── Stack-machine sdScene ─────────────────────────────────────────────────
  //
  // The JS side compiles each CSG tree into a postorder instruction sequence.
  // PUSH_SHAPE instructions push an SDF value; OP instructions pop two values,
  // apply the operation, and push the result. After all instructions the result
  // sits in s0. Max stack depth for 15-node binary tree is 8.
  //
  // Named variables s0..s7 + sp avoid the need for local WGSL arrays.
  const sdScene = (p: d.v3f): number => {
    "use gpu";
    let dist = d.f32(1e10);

    // Stack declared at function scope for WGSL compatibility
    let s0 = d.f32(0.0);
    let s1 = d.f32(0.0);
    let s2 = d.f32(0.0);
    let s3 = d.f32(0.0);
    let s4 = d.f32(0.0);
    let s5 = d.f32(0.0);
    let s6 = d.f32(0.0);
    let s7 = d.f32(0.0);
    let sp = d.u32(0);

    const objCount = objectCountUniform.$;
    for (let o = d.u32(0); o < objCount; o += d.u32(1)) {
      // Reset stack for this object
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
          // PUSH_SHAPE: evaluate SDF and push onto stack
          const lp = applyInvRotXYZ(p - instr.position, instr.rotation);
          const val = evalShape(lp, instr.shapeType, instr.params);
          if (sp === d.u32(0)) s0 = val;
          else if (sp === d.u32(1)) s1 = val;
          else if (sp === d.u32(2)) s2 = val;
          else if (sp === d.u32(3)) s3 = val;
          else if (sp === d.u32(4)) s4 = val;
          else if (sp === d.u32(5)) s5 = val;
          else if (sp === d.u32(6)) s6 = val;
          else s7 = val;
          sp += d.u32(1);
        } else {
          // OP: pop right operand (b), pop left operand (a), push result
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

      // After full evaluation sp==1 and s0 holds this object's SDF value
      dist = std.min(dist, s0);
    }

    return dist;
  };

  // Evaluate SDF for the selected CSG subtree (single instruction sequence).
  const sdSelection = (p: d.v3f): number => {
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

    const count = selectionCountUniform.$;
    for (let i = d.u32(0); i < count; i += d.u32(1)) {
      const instr = selectionInstructionsBuffer.$[i];

      if (instr.opcode === d.u32(0)) {
        const lp = applyInvRotXYZ(p - instr.position, instr.rotation);
        const val = evalShape(lp, instr.shapeType, instr.params);
        if (sp === d.u32(0)) s0 = val;
        else if (sp === d.u32(1)) s1 = val;
        else if (sp === d.u32(2)) s2 = val;
        else if (sp === d.u32(3)) s3 = val;
        else if (sp === d.u32(4)) s4 = val;
        else if (sp === d.u32(5)) s5 = val;
        else if (sp === d.u32(6)) s6 = val;
        else s7 = val;
        sp += d.u32(1);
      } else {
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

  // Raymarch a thin inflated shell around the selected subtree (sdSelection = OUTLINE_OFFSET).
  const rayMarchSelectionOutline = (ro: d.v3f, rd: d.v3f): number => {
    "use gpu";
    let t = d.f32(0.0);
    for (let i = d.f32(0.0); i < d.f32(48.0); i += d.f32(1.0)) {
      const p = ro + rd * t;
      const dSel = sdSelection(p);
      const dist = std.abs(dSel - OUTLINE_OFFSET) - OUTLINE_BAND;
      t += dist;
      if (dist < 0.0001 || t > 100.0) {
        break;
      }
    }
    return t;
  };

  const selectionGradMag = (p: d.v3f): number => {
    "use gpu";
    const e = 0.001;
    const dx =
      sdSelection(p + d.vec3f(e, 0.0, 0.0)) -
      sdSelection(p - d.vec3f(e, 0.0, 0.0));
    const dy =
      sdSelection(p + d.vec3f(0.0, e, 0.0)) -
      sdSelection(p - d.vec3f(0.0, e, 0.0));
    const dz =
      sdSelection(p + d.vec3f(0.0, 0.0, e)) -
      sdSelection(p - d.vec3f(0.0, 0.0, e));
    return std.length(d.vec3f(dx, dy, dz)) / (2.0 * e);
  };

  const calcSelectionNormal = (p: d.v3f): d.v3f => {
    "use gpu";
    const e = 0.001;
    const dx =
      sdSelection(p + d.vec3f(e, 0.0, 0.0)) -
      sdSelection(p - d.vec3f(e, 0.0, 0.0));
    const dy =
      sdSelection(p + d.vec3f(0.0, e, 0.0)) -
      sdSelection(p - d.vec3f(0.0, e, 0.0));
    const dz =
      sdSelection(p + d.vec3f(0.0, 0.0, e)) -
      sdSelection(p - d.vec3f(0.0, 0.0, e));
    return std.normalize(d.vec3f(dx, dy, dz));
  };

  // Silhouette rim + geometric creases — thin line, not a filled shell.
  const selectionOutlineMask = (p: d.v3f, rd: d.v3f): number => {
    "use gpu";
    const viewDir = std.normalize(d.vec3f(-rd.x, -rd.y, -rd.z));
    const N = calcSelectionNormal(p);
    const rim = std.pow(1.0 - std.abs(std.dot(N, viewDir)), OUTLINE_RIM_POWER);
    const gradEdge = std.smoothstep(
      OUTLINE_GRAD_LO,
      OUTLINE_GRAD_HI,
      selectionGradMag(p),
    );
    const edge = std.max(rim, gradEdge);
    return std.smoothstep(OUTLINE_EDGE_LO, OUTLINE_EDGE_HI, edge);
  };

  const selectionOutlineColor = (): d.v3f => {
    "use gpu";
    const pulse = 0.75 + 0.25 * std.sin(timeUniform.$ * 2.5);
    const v = pulse * OUTLINE_STRENGTH;
    return d.vec3f(v, v, v);
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
    let t = d.f32(0.0);
    let iterations = d.f32(0.0);
    for (let i = d.f32(0.0); i < d.f32(48.0); i += d.f32(1.0)) {
      const p = ro + rd * t;
      const dist = sdScene(p);
      iterations += 1.0;
      t += dist;
      if (dist < 0.0001 || t > 100.0) {
        break;
      }
    }
    return d.vec2f(t, iterations);
  };

  const rot2D = (angle: number): d.m2x2f => {
    "use gpu";
    const c = std.cos(angle);
    const s = std.sin(angle);
    return d.mat2x2f(c, -s, s, c);
  };

  const fragment = ({ uv }: { uv: d.v2f }) => {
    "use gpu";

    const uvn = uv * 2.0 - d.vec2f(1.0, 1.0);
    const uvCorrected = d.vec2f(uvn.x * aspectUniform.$, uvn.y);
    let ro = d.vec3f(0, 0, 0 - distanceUniform.$);
    let rd = std.normalize(d.vec3f(uvCorrected.x, uvCorrected.y, 1.0));

    const rotH = rot2D(mouseUniform.$.x);
    const roXZ = rotH * d.vec2f(ro.x, ro.z);
    ro = d.vec3f(roXZ.x, ro.y, roXZ.y);
    const rdXZ = rotH * d.vec2f(rd.x, rd.z);
    rd = d.vec3f(rdXZ.x, rd.y, rdXZ.y);

    const rotV = rot2D(mouseUniform.$.y);
    const roYZ = rotV * d.vec2f(ro.y, ro.z);
    ro = d.vec3f(ro.x, roYZ.x, roYZ.y);
    const rdYZ = rotV * d.vec2f(rd.y, rd.z);
    rd = d.vec3f(rd.x, rdYZ.x, rdYZ.y);

    const sceneResult = rayMarch(ro, rd);
    const tScene = sceneResult.x;
    const iterations = sceneResult.y;
    const mode = renderModeUniform.$;

    let tOutline = d.f32(RAY_MISS_T + 1.0);
    if (selectionEnabledUniform.$ === d.u32(1)) {
      tOutline = rayMarchSelectionOutline(ro, rd);
    }

    const outlineHit = tOutline <= RAY_MISS_T;
    const sceneHit = tScene <= RAY_MISS_T;
    const outlineVisible =
      selectionEnabledUniform.$ === d.u32(1) &&
      outlineHit &&
      (!sceneHit || tOutline < tScene - 0.0001);

    let col = d.vec3f(0.05, 0.05, 0.08);

    if (!sceneHit) {
      if (mode !== d.u32(0)) {
        col = d.vec3f(0.0, 0.0, 0.0);
      }
    } else {
      const pos = ro + rd * tScene;

      if (mode === d.u32(1)) {
        const depth = std.clamp(tScene / 5.0, 0.0, 1.0);
        const brightness = 1.0 - depth;
        col = d.vec3f(brightness, brightness, brightness);
      } else if (mode === d.u32(2)) {
        const brightness = std.clamp(iterations / 64.0, 0.0, 1.0);
        col = d.vec3f(brightness, brightness, brightness);
      } else {
        const N = calcNormal(pos);
        const lightDir = std.normalize(d.vec3f(2.0, 3.0, -1.0));
        const diff = std.max(std.dot(N, lightDir), 0.0);
        const viewDir = d.vec3f(-rd.x, -rd.y, -rd.z);
        const fresnel = std.pow(1.0 - std.abs(std.dot(N, viewDir)), 3.0) * 0.5;
        const halfDir = std.normalize(lightDir + viewDir);
        const spec = std.pow(std.max(std.dot(N, halfDir), 0.0), 64.0) * 1.5;
        const ao = 1.0 - std.clamp(iterations / 64.0, 0.0, 1.0) * 0.4;
        const baseColor = d.vec3f(0.55, 0.65, 0.95);
        col = std.sqrt(
          std.max(
            (baseColor * (diff * 0.8 + 0.2) +
              d.vec3f(1.0, 1.0, 1.0) * spec +
              baseColor * fresnel) *
              ao,
            d.vec3f(0.0),
          ),
        );
      }
    }

    if (outlineVisible) {
      const outlinePos = ro + rd * tOutline;
      const edge = selectionOutlineMask(outlinePos, rd);
      col = col + selectionOutlineColor() * edge;
    }

    return d.vec4f(col, 1.0);
  };

  const pipeline = root.createRenderPipeline({
    vertex: fullScreenTriangle,
    fragment,
  });

  return {
    pipeline,
    timeUniform,
    aspectUniform,
    mouseUniform,
    distanceUniform,
    instructionsBuffer,
    objectInfoBuffer,
    objectCountUniform,
    renderModeUniform,
    selectionInstructionsBuffer,
    selectionCountUniform,
    selectionEnabledUniform,
  };
}
