import tgpu, { d, std } from "typegpu";
import { fullScreenTriangle } from "typegpu/common";
import {
  MAX_GPU_OBJECTS,
  MAX_INSTRUCTIONS,
  MAX_PICK_INSTRUCTIONS,
  MAX_PICK_OBJECTS,
  MAX_TRANSFORM_DEPTH,
} from "../store/sceneStore";

type TgpuRoot = Awaited<ReturnType<typeof tgpu.init>>;

export {
  MAX_GPU_OBJECTS,
  MAX_INSTRUCTIONS,
  MAX_PICK_INSTRUCTIONS,
  MAX_PICK_OBJECTS,
  MAX_TRANSFORM_DEPTH,
};

export const OPCODE_PUSH_SHAPE = 0;
export const OPCODE_OP = 1;
export const OPCODE_TRANSFORM_PUSH = 2;
export const OPCODE_TRANSFORM_POP = 3;

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

// Each instruction is PUSH_SHAPE (0), OP (1), TRANSFORM_PUSH (2), or TRANSFORM_POP (3).
// Layout: 4×u32/f32 header (16 bytes) + vec3f+pad (16 bytes) + vec4f (16 bytes) + vec3f+pad (16 bytes) = 64 bytes
const Instruction = d.struct({
  opcode: d.u32, // 0=PUSH_SHAPE, 1=OP, 2=TRANSFORM_PUSH, 3=TRANSFORM_POP
  shapeType: d.u32, // for PUSH_SHAPE: 0-5
  opType: d.u32, // for OP: 0=union,1=subtract,2=intersect,3=sUnion,4=sSubtract,5=sIntersect
  smoothK: d.f32, // for smooth OPs
  position: d.vec3f, // for PUSH_SHAPE / TRANSFORM_PUSH
  _pad: d.f32, // alignment padding after vec3f
  params: d.vec4f, // for PUSH_SHAPE shape params; TRANSFORM_PUSH scale in xyz
  rotation: d.vec3f, // for PUSH_SHAPE / TRANSFORM_PUSH: Euler XYZ in radians
  _pad2: d.f32, // alignment padding after vec3f
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
});

const SelectionUniforms = d.struct({
  enabled: d.u32,
  usesSceneSdf: d.u32,
  count: d.u32,
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

const TOTAL_INSTRUCTIONS = MAX_INSTRUCTIONS;

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
/** Surface tie epsilon for pick — prefer smaller CSG subtree (more specific item). */
const PICK_TIE_EPS = 0.002;

/** Pick pass: ray-march translate gizmo, output axis id 1/2/3 in R. */
export const PICK_PASS_GIZMO = 2;

export function createShader(root: TgpuRoot) {
  const cameraUniforms = root.createUniform(CameraUniforms, {
    time: 0,
    aspect: 1,
    mouse: d.vec2f(0.3, -0.4),
    distance: 2.5,
  });
  const sceneUniforms = root.createUniform(SceneUniforms, {
    objectCount: 0,
  });
  const selectionUniforms = root.createUniform(SelectionUniforms, {
    enabled: 0,
    usesSceneSdf: 0,
    count: 0,
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

  const applyParentTransform = (
    p: d.v3f,
    pos: d.v3f,
    rot: d.v3f,
    scale: d.v3f,
  ): d.v3f => {
    "use gpu";
    const q = p - pos;
    const r = applyInvRotXYZ(q, rot);
    return d.vec3f(r.x / scale.x, r.y / scale.y, r.z / scale.z);
  };

  const minVec3 = (v: d.v3f): number => {
    "use gpu";
    return std.min(v.x, std.min(v.y, v.z));
  };

  const divAccSclBy = (acc: d.v3f, s: d.v3f): d.v3f => {
    "use gpu";
    return d.vec3f(acc.x / s.x, acc.y / s.y, acc.z / s.z);
  };

  const mulVec3 = (a: d.v3f, b: d.v3f): d.v3f => {
    "use gpu";
    return d.vec3f(a.x * b.x, a.y * b.y, a.z * b.z);
  };

  // Undo a shape's own rotation in a temporarily "unscaled" frame (multiply by
  // the ancestor scale, rotate, then divide back out) so a non-uniform parent
  // scale never shears the shape's own orientation into a skewed parallelogram.
  // Equivalent to leaving the scale outside the rotation in the forward
  // transform (Rgroup * Rshape * Sgroup * local) instead of sandwiching the
  // rotation inside the scale (Rgroup * Sgroup * Rshape * local).
  const applyShapeRotationUnsheared = (
    diff: d.v3f,
    rot: d.v3f,
    outerScl: d.v3f,
  ): d.v3f => {
    "use gpu";
    const scaledUp = mulVec3(diff, outerScl);
    const rotated = applyInvRotXYZ(scaledUp, rot);
    return divAccSclBy(rotated, outerScl);
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
  // apply the operation, and push the result. TRANSFORM_PUSH/POP manage a
  // parent transform stack for nested groups. After all instructions the result
  // sits in s0. Max stack depth for 15-node binary tree is 8.
  //
  // Named variables s0..s7 + sp avoid the need for local WGSL arrays.
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

    // Transform stack (unrolled for WGSL)
    let tPos0 = d.vec3f(0.0, 0.0, 0.0);
    let tRot0 = d.vec3f(0.0, 0.0, 0.0);
    let tScl0 = d.vec3f(1.0, 1.0, 1.0);
    let tPos1 = d.vec3f(0.0, 0.0, 0.0);
    let tRot1 = d.vec3f(0.0, 0.0, 0.0);
    let tScl1 = d.vec3f(1.0, 1.0, 1.0);
    let tPos2 = d.vec3f(0.0, 0.0, 0.0);
    let tRot2 = d.vec3f(0.0, 0.0, 0.0);
    let tScl2 = d.vec3f(1.0, 1.0, 1.0);
    let tPos3 = d.vec3f(0.0, 0.0, 0.0);
    let tRot3 = d.vec3f(0.0, 0.0, 0.0);
    let tScl3 = d.vec3f(1.0, 1.0, 1.0);
    let tPos4 = d.vec3f(0.0, 0.0, 0.0);
    let tRot4 = d.vec3f(0.0, 0.0, 0.0);
    let tScl4 = d.vec3f(1.0, 1.0, 1.0);
    let tPos5 = d.vec3f(0.0, 0.0, 0.0);
    let tRot5 = d.vec3f(0.0, 0.0, 0.0);
    let tScl5 = d.vec3f(1.0, 1.0, 1.0);
    let tPos6 = d.vec3f(0.0, 0.0, 0.0);
    let tRot6 = d.vec3f(0.0, 0.0, 0.0);
    let tScl6 = d.vec3f(1.0, 1.0, 1.0);
    let tPos7 = d.vec3f(0.0, 0.0, 0.0);
    let tRot7 = d.vec3f(0.0, 0.0, 0.0);
    let tScl7 = d.vec3f(1.0, 1.0, 1.0);
    let tPos8 = d.vec3f(0.0, 0.0, 0.0);
    let tRot8 = d.vec3f(0.0, 0.0, 0.0);
    let tScl8 = d.vec3f(1.0, 1.0, 1.0);
    let tPos9 = d.vec3f(0.0, 0.0, 0.0);
    let tRot9 = d.vec3f(0.0, 0.0, 0.0);
    let tScl9 = d.vec3f(1.0, 1.0, 1.0);
    let tPos10 = d.vec3f(0.0, 0.0, 0.0);
    let tRot10 = d.vec3f(0.0, 0.0, 0.0);
    let tScl10 = d.vec3f(1.0, 1.0, 1.0);
    let tPos11 = d.vec3f(0.0, 0.0, 0.0);
    let tRot11 = d.vec3f(0.0, 0.0, 0.0);
    let tScl11 = d.vec3f(1.0, 1.0, 1.0);
    let tPos12 = d.vec3f(0.0, 0.0, 0.0);
    let tRot12 = d.vec3f(0.0, 0.0, 0.0);
    let tScl12 = d.vec3f(1.0, 1.0, 1.0);
    let tPos13 = d.vec3f(0.0, 0.0, 0.0);
    let tRot13 = d.vec3f(0.0, 0.0, 0.0);
    let tScl13 = d.vec3f(1.0, 1.0, 1.0);
    let tPos14 = d.vec3f(0.0, 0.0, 0.0);
    let tRot14 = d.vec3f(0.0, 0.0, 0.0);
    let tScl14 = d.vec3f(1.0, 1.0, 1.0);
    let tPos15 = d.vec3f(0.0, 0.0, 0.0);
    let tRot15 = d.vec3f(0.0, 0.0, 0.0);
    let tScl15 = d.vec3f(1.0, 1.0, 1.0);
    let tsp = d.u32(0);
    let accScl = d.vec3f(1.0, 1.0, 1.0);

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
      tsp = d.u32(0);
      accScl = d.vec3f(1.0, 1.0, 1.0);

      const info = objectInfoBuffer.$[o];
      const end = info.start + info.count;

      for (let i = info.start; i < end; i += d.u32(1)) {
        const instr = instructionsBuffer.$[i];

        if (instr.opcode === d.u32(0)) {
          let pEval = d.vec3f(p.x, p.y, p.z);
          if (tsp > d.u32(0))
            pEval = applyParentTransform(pEval, tPos0, tRot0, tScl0);
          if (tsp > d.u32(1))
            pEval = applyParentTransform(pEval, tPos1, tRot1, tScl1);
          if (tsp > d.u32(2))
            pEval = applyParentTransform(pEval, tPos2, tRot2, tScl2);
          if (tsp > d.u32(3))
            pEval = applyParentTransform(pEval, tPos3, tRot3, tScl3);
          if (tsp > d.u32(4))
            pEval = applyParentTransform(pEval, tPos4, tRot4, tScl4);
          if (tsp > d.u32(5))
            pEval = applyParentTransform(pEval, tPos5, tRot5, tScl5);
          if (tsp > d.u32(6))
            pEval = applyParentTransform(pEval, tPos6, tRot6, tScl6);
          if (tsp > d.u32(7))
            pEval = applyParentTransform(pEval, tPos7, tRot7, tScl7);
          if (tsp > d.u32(8))
            pEval = applyParentTransform(pEval, tPos8, tRot8, tScl8);
          if (tsp > d.u32(9))
            pEval = applyParentTransform(pEval, tPos9, tRot9, tScl9);
          if (tsp > d.u32(10))
            pEval = applyParentTransform(pEval, tPos10, tRot10, tScl10);
          if (tsp > d.u32(11))
            pEval = applyParentTransform(pEval, tPos11, tRot11, tScl11);
          if (tsp > d.u32(12))
            pEval = applyParentTransform(pEval, tPos12, tRot12, tScl12);
          if (tsp > d.u32(13))
            pEval = applyParentTransform(pEval, tPos13, tRot13, tScl13);
          if (tsp > d.u32(14))
            pEval = applyParentTransform(pEval, tPos14, tRot14, tScl14);
          if (tsp > d.u32(15))
            pEval = applyParentTransform(pEval, tPos15, tRot15, tScl15);
          const lp = applyShapeRotationUnsheared(
            pEval - instr.position,
            instr.rotation,
            accScl,
          );
          let val = evalShape(lp, instr.shapeType, instr.params);
          val = val * minVec3(accScl);
          if (sp === d.u32(0)) s0 = val;
          else if (sp === d.u32(1)) s1 = val;
          else if (sp === d.u32(2)) s2 = val;
          else if (sp === d.u32(3)) s3 = val;
          else if (sp === d.u32(4)) s4 = val;
          else if (sp === d.u32(5)) s5 = val;
          else if (sp === d.u32(6)) s6 = val;
          else s7 = val;
          sp += d.u32(1);
        } else if (instr.opcode === d.u32(2)) {
          if (tsp === d.u32(0)) {
            tPos0 = d.vec3f(
              instr.position.x,
              instr.position.y,
              instr.position.z,
            );
            tRot0 = d.vec3f(
              instr.rotation.x,
              instr.rotation.y,
              instr.rotation.z,
            );
            tScl0 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
          } else if (tsp === d.u32(1)) {
            tPos1 = d.vec3f(
              instr.position.x,
              instr.position.y,
              instr.position.z,
            );
            tRot1 = d.vec3f(
              instr.rotation.x,
              instr.rotation.y,
              instr.rotation.z,
            );
            tScl1 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
          } else if (tsp === d.u32(2)) {
            tPos2 = d.vec3f(
              instr.position.x,
              instr.position.y,
              instr.position.z,
            );
            tRot2 = d.vec3f(
              instr.rotation.x,
              instr.rotation.y,
              instr.rotation.z,
            );
            tScl2 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
          } else if (tsp === d.u32(3)) {
            tPos3 = d.vec3f(
              instr.position.x,
              instr.position.y,
              instr.position.z,
            );
            tRot3 = d.vec3f(
              instr.rotation.x,
              instr.rotation.y,
              instr.rotation.z,
            );
            tScl3 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
          } else if (tsp === d.u32(4)) {
            tPos4 = d.vec3f(
              instr.position.x,
              instr.position.y,
              instr.position.z,
            );
            tRot4 = d.vec3f(
              instr.rotation.x,
              instr.rotation.y,
              instr.rotation.z,
            );
            tScl4 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
          } else if (tsp === d.u32(5)) {
            tPos5 = d.vec3f(
              instr.position.x,
              instr.position.y,
              instr.position.z,
            );
            tRot5 = d.vec3f(
              instr.rotation.x,
              instr.rotation.y,
              instr.rotation.z,
            );
            tScl5 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
          } else if (tsp === d.u32(6)) {
            tPos6 = d.vec3f(
              instr.position.x,
              instr.position.y,
              instr.position.z,
            );
            tRot6 = d.vec3f(
              instr.rotation.x,
              instr.rotation.y,
              instr.rotation.z,
            );
            tScl6 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
          } else if (tsp === d.u32(7)) {
            tPos7 = d.vec3f(
              instr.position.x,
              instr.position.y,
              instr.position.z,
            );
            tRot7 = d.vec3f(
              instr.rotation.x,
              instr.rotation.y,
              instr.rotation.z,
            );
            tScl7 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
          } else if (tsp === d.u32(8)) {
            tPos8 = d.vec3f(
              instr.position.x,
              instr.position.y,
              instr.position.z,
            );
            tRot8 = d.vec3f(
              instr.rotation.x,
              instr.rotation.y,
              instr.rotation.z,
            );
            tScl8 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
          } else if (tsp === d.u32(9)) {
            tPos9 = d.vec3f(
              instr.position.x,
              instr.position.y,
              instr.position.z,
            );
            tRot9 = d.vec3f(
              instr.rotation.x,
              instr.rotation.y,
              instr.rotation.z,
            );
            tScl9 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
          } else if (tsp === d.u32(10)) {
            tPos10 = d.vec3f(
              instr.position.x,
              instr.position.y,
              instr.position.z,
            );
            tRot10 = d.vec3f(
              instr.rotation.x,
              instr.rotation.y,
              instr.rotation.z,
            );
            tScl10 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
          } else if (tsp === d.u32(11)) {
            tPos11 = d.vec3f(
              instr.position.x,
              instr.position.y,
              instr.position.z,
            );
            tRot11 = d.vec3f(
              instr.rotation.x,
              instr.rotation.y,
              instr.rotation.z,
            );
            tScl11 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
          } else if (tsp === d.u32(12)) {
            tPos12 = d.vec3f(
              instr.position.x,
              instr.position.y,
              instr.position.z,
            );
            tRot12 = d.vec3f(
              instr.rotation.x,
              instr.rotation.y,
              instr.rotation.z,
            );
            tScl12 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
          } else if (tsp === d.u32(13)) {
            tPos13 = d.vec3f(
              instr.position.x,
              instr.position.y,
              instr.position.z,
            );
            tRot13 = d.vec3f(
              instr.rotation.x,
              instr.rotation.y,
              instr.rotation.z,
            );
            tScl13 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
          } else if (tsp === d.u32(14)) {
            tPos14 = d.vec3f(
              instr.position.x,
              instr.position.y,
              instr.position.z,
            );
            tRot14 = d.vec3f(
              instr.rotation.x,
              instr.rotation.y,
              instr.rotation.z,
            );
            tScl14 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
          } else if (tsp === d.u32(15)) {
            tPos15 = d.vec3f(
              instr.position.x,
              instr.position.y,
              instr.position.z,
            );
            tRot15 = d.vec3f(
              instr.rotation.x,
              instr.rotation.y,
              instr.rotation.z,
            );
            tScl15 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
          }
          accScl = d.vec3f(
            accScl.x * instr.params.x,
            accScl.y * instr.params.y,
            accScl.z * instr.params.z,
          );
          tsp += d.u32(1);
        } else if (instr.opcode === d.u32(3)) {
          if (tsp === d.u32(1)) accScl = divAccSclBy(accScl, tScl0);
          else if (tsp === d.u32(2)) accScl = divAccSclBy(accScl, tScl1);
          else if (tsp === d.u32(3)) accScl = divAccSclBy(accScl, tScl2);
          else if (tsp === d.u32(4)) accScl = divAccSclBy(accScl, tScl3);
          else if (tsp === d.u32(5)) accScl = divAccSclBy(accScl, tScl4);
          else if (tsp === d.u32(6)) accScl = divAccSclBy(accScl, tScl5);
          else if (tsp === d.u32(7)) accScl = divAccSclBy(accScl, tScl6);
          else if (tsp === d.u32(8)) accScl = divAccSclBy(accScl, tScl7);
          else if (tsp === d.u32(9)) accScl = divAccSclBy(accScl, tScl8);
          else if (tsp === d.u32(10)) accScl = divAccSclBy(accScl, tScl9);
          else if (tsp === d.u32(11)) accScl = divAccSclBy(accScl, tScl10);
          else if (tsp === d.u32(12)) accScl = divAccSclBy(accScl, tScl11);
          else if (tsp === d.u32(13)) accScl = divAccSclBy(accScl, tScl12);
          else if (tsp === d.u32(14)) accScl = divAccSclBy(accScl, tScl13);
          else if (tsp === d.u32(15)) accScl = divAccSclBy(accScl, tScl14);
          else if (tsp === d.u32(16)) accScl = divAccSclBy(accScl, tScl15);
          if (tsp > d.u32(0)) tsp -= d.u32(1);
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

  // Evaluate SDF for an instruction range from the selection buffer.
  const evalSelectionInstructionRange = (
    p: d.v3f,
    start: d.u32,
    count: d.u32,
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

    let tPos0 = d.vec3f(0.0, 0.0, 0.0);
    let tRot0 = d.vec3f(0.0, 0.0, 0.0);
    let tScl0 = d.vec3f(1.0, 1.0, 1.0);
    let tPos1 = d.vec3f(0.0, 0.0, 0.0);
    let tRot1 = d.vec3f(0.0, 0.0, 0.0);
    let tScl1 = d.vec3f(1.0, 1.0, 1.0);
    let tPos2 = d.vec3f(0.0, 0.0, 0.0);
    let tRot2 = d.vec3f(0.0, 0.0, 0.0);
    let tScl2 = d.vec3f(1.0, 1.0, 1.0);
    let tPos3 = d.vec3f(0.0, 0.0, 0.0);
    let tRot3 = d.vec3f(0.0, 0.0, 0.0);
    let tScl3 = d.vec3f(1.0, 1.0, 1.0);
    let tPos4 = d.vec3f(0.0, 0.0, 0.0);
    let tRot4 = d.vec3f(0.0, 0.0, 0.0);
    let tScl4 = d.vec3f(1.0, 1.0, 1.0);
    let tPos5 = d.vec3f(0.0, 0.0, 0.0);
    let tRot5 = d.vec3f(0.0, 0.0, 0.0);
    let tScl5 = d.vec3f(1.0, 1.0, 1.0);
    let tPos6 = d.vec3f(0.0, 0.0, 0.0);
    let tRot6 = d.vec3f(0.0, 0.0, 0.0);
    let tScl6 = d.vec3f(1.0, 1.0, 1.0);
    let tPos7 = d.vec3f(0.0, 0.0, 0.0);
    let tRot7 = d.vec3f(0.0, 0.0, 0.0);
    let tScl7 = d.vec3f(1.0, 1.0, 1.0);
    let tPos8 = d.vec3f(0.0, 0.0, 0.0);
    let tRot8 = d.vec3f(0.0, 0.0, 0.0);
    let tScl8 = d.vec3f(1.0, 1.0, 1.0);
    let tPos9 = d.vec3f(0.0, 0.0, 0.0);
    let tRot9 = d.vec3f(0.0, 0.0, 0.0);
    let tScl9 = d.vec3f(1.0, 1.0, 1.0);
    let tPos10 = d.vec3f(0.0, 0.0, 0.0);
    let tRot10 = d.vec3f(0.0, 0.0, 0.0);
    let tScl10 = d.vec3f(1.0, 1.0, 1.0);
    let tPos11 = d.vec3f(0.0, 0.0, 0.0);
    let tRot11 = d.vec3f(0.0, 0.0, 0.0);
    let tScl11 = d.vec3f(1.0, 1.0, 1.0);
    let tPos12 = d.vec3f(0.0, 0.0, 0.0);
    let tRot12 = d.vec3f(0.0, 0.0, 0.0);
    let tScl12 = d.vec3f(1.0, 1.0, 1.0);
    let tPos13 = d.vec3f(0.0, 0.0, 0.0);
    let tRot13 = d.vec3f(0.0, 0.0, 0.0);
    let tScl13 = d.vec3f(1.0, 1.0, 1.0);
    let tPos14 = d.vec3f(0.0, 0.0, 0.0);
    let tRot14 = d.vec3f(0.0, 0.0, 0.0);
    let tScl14 = d.vec3f(1.0, 1.0, 1.0);
    let tPos15 = d.vec3f(0.0, 0.0, 0.0);
    let tRot15 = d.vec3f(0.0, 0.0, 0.0);
    let tScl15 = d.vec3f(1.0, 1.0, 1.0);
    let tsp = d.u32(0);
    let accScl = d.vec3f(1.0, 1.0, 1.0);

    const end = start + count;
    for (let i = start; i < end; i += d.u32(1)) {
      const instr = selectionInstructionsBuffer.$[i];

      if (instr.opcode === d.u32(0)) {
        let pEval = d.vec3f(p.x, p.y, p.z);
        if (tsp > d.u32(0))
          pEval = applyParentTransform(pEval, tPos0, tRot0, tScl0);
        if (tsp > d.u32(1))
          pEval = applyParentTransform(pEval, tPos1, tRot1, tScl1);
        if (tsp > d.u32(2))
          pEval = applyParentTransform(pEval, tPos2, tRot2, tScl2);
        if (tsp > d.u32(3))
          pEval = applyParentTransform(pEval, tPos3, tRot3, tScl3);
        if (tsp > d.u32(4))
          pEval = applyParentTransform(pEval, tPos4, tRot4, tScl4);
        if (tsp > d.u32(5))
          pEval = applyParentTransform(pEval, tPos5, tRot5, tScl5);
        if (tsp > d.u32(6))
          pEval = applyParentTransform(pEval, tPos6, tRot6, tScl6);
        if (tsp > d.u32(7))
          pEval = applyParentTransform(pEval, tPos7, tRot7, tScl7);
        if (tsp > d.u32(8))
          pEval = applyParentTransform(pEval, tPos8, tRot8, tScl8);
        if (tsp > d.u32(9))
          pEval = applyParentTransform(pEval, tPos9, tRot9, tScl9);
        if (tsp > d.u32(10))
          pEval = applyParentTransform(pEval, tPos10, tRot10, tScl10);
        if (tsp > d.u32(11))
          pEval = applyParentTransform(pEval, tPos11, tRot11, tScl11);
        if (tsp > d.u32(12))
          pEval = applyParentTransform(pEval, tPos12, tRot12, tScl12);
        if (tsp > d.u32(13))
          pEval = applyParentTransform(pEval, tPos13, tRot13, tScl13);
        if (tsp > d.u32(14))
          pEval = applyParentTransform(pEval, tPos14, tRot14, tScl14);
        if (tsp > d.u32(15))
          pEval = applyParentTransform(pEval, tPos15, tRot15, tScl15);
        const lp = applyShapeRotationUnsheared(
          pEval - instr.position,
          instr.rotation,
          accScl,
        );
        let val = evalShape(lp, instr.shapeType, instr.params);
        val = val * minVec3(accScl);
        if (sp === d.u32(0)) s0 = val;
        else if (sp === d.u32(1)) s1 = val;
        else if (sp === d.u32(2)) s2 = val;
        else if (sp === d.u32(3)) s3 = val;
        else if (sp === d.u32(4)) s4 = val;
        else if (sp === d.u32(5)) s5 = val;
        else if (sp === d.u32(6)) s6 = val;
        else s7 = val;
        sp += d.u32(1);
      } else if (instr.opcode === d.u32(2)) {
        if (tsp === d.u32(0)) {
          tPos0 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot0 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl0 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(1)) {
          tPos1 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot1 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl1 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(2)) {
          tPos2 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot2 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl2 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(3)) {
          tPos3 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot3 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl3 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(4)) {
          tPos4 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot4 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl4 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(5)) {
          tPos5 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot5 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl5 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(6)) {
          tPos6 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot6 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl6 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(7)) {
          tPos7 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot7 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl7 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(8)) {
          tPos8 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot8 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl8 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(9)) {
          tPos9 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot9 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl9 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(10)) {
          tPos10 = d.vec3f(
            instr.position.x,
            instr.position.y,
            instr.position.z,
          );
          tRot10 = d.vec3f(
            instr.rotation.x,
            instr.rotation.y,
            instr.rotation.z,
          );
          tScl10 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(11)) {
          tPos11 = d.vec3f(
            instr.position.x,
            instr.position.y,
            instr.position.z,
          );
          tRot11 = d.vec3f(
            instr.rotation.x,
            instr.rotation.y,
            instr.rotation.z,
          );
          tScl11 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(12)) {
          tPos12 = d.vec3f(
            instr.position.x,
            instr.position.y,
            instr.position.z,
          );
          tRot12 = d.vec3f(
            instr.rotation.x,
            instr.rotation.y,
            instr.rotation.z,
          );
          tScl12 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(13)) {
          tPos13 = d.vec3f(
            instr.position.x,
            instr.position.y,
            instr.position.z,
          );
          tRot13 = d.vec3f(
            instr.rotation.x,
            instr.rotation.y,
            instr.rotation.z,
          );
          tScl13 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(14)) {
          tPos14 = d.vec3f(
            instr.position.x,
            instr.position.y,
            instr.position.z,
          );
          tRot14 = d.vec3f(
            instr.rotation.x,
            instr.rotation.y,
            instr.rotation.z,
          );
          tScl14 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(15)) {
          tPos15 = d.vec3f(
            instr.position.x,
            instr.position.y,
            instr.position.z,
          );
          tRot15 = d.vec3f(
            instr.rotation.x,
            instr.rotation.y,
            instr.rotation.z,
          );
          tScl15 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        }
        accScl = d.vec3f(
          accScl.x * instr.params.x,
          accScl.y * instr.params.y,
          accScl.z * instr.params.z,
        );
        tsp += d.u32(1);
      } else if (instr.opcode === d.u32(3)) {
        if (tsp === d.u32(1)) accScl = divAccSclBy(accScl, tScl0);
        else if (tsp === d.u32(2)) accScl = divAccSclBy(accScl, tScl1);
        else if (tsp === d.u32(3)) accScl = divAccSclBy(accScl, tScl2);
        else if (tsp === d.u32(4)) accScl = divAccSclBy(accScl, tScl3);
        else if (tsp === d.u32(5)) accScl = divAccSclBy(accScl, tScl4);
        else if (tsp === d.u32(6)) accScl = divAccSclBy(accScl, tScl5);
        else if (tsp === d.u32(7)) accScl = divAccSclBy(accScl, tScl6);
        else if (tsp === d.u32(8)) accScl = divAccSclBy(accScl, tScl7);
        else if (tsp === d.u32(9)) accScl = divAccSclBy(accScl, tScl8);
        else if (tsp === d.u32(10)) accScl = divAccSclBy(accScl, tScl9);
        else if (tsp === d.u32(11)) accScl = divAccSclBy(accScl, tScl10);
        else if (tsp === d.u32(12)) accScl = divAccSclBy(accScl, tScl11);
        else if (tsp === d.u32(13)) accScl = divAccSclBy(accScl, tScl12);
        else if (tsp === d.u32(14)) accScl = divAccSclBy(accScl, tScl13);
        else if (tsp === d.u32(15)) accScl = divAccSclBy(accScl, tScl14);
        else if (tsp === d.u32(16)) accScl = divAccSclBy(accScl, tScl15);
        if (tsp > d.u32(0)) tsp -= d.u32(1);
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
    start: d.u32,
    count: d.u32,
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

    let tPos0 = d.vec3f(0.0, 0.0, 0.0);
    let tRot0 = d.vec3f(0.0, 0.0, 0.0);
    let tScl0 = d.vec3f(1.0, 1.0, 1.0);
    let tPos1 = d.vec3f(0.0, 0.0, 0.0);
    let tRot1 = d.vec3f(0.0, 0.0, 0.0);
    let tScl1 = d.vec3f(1.0, 1.0, 1.0);
    let tPos2 = d.vec3f(0.0, 0.0, 0.0);
    let tRot2 = d.vec3f(0.0, 0.0, 0.0);
    let tScl2 = d.vec3f(1.0, 1.0, 1.0);
    let tPos3 = d.vec3f(0.0, 0.0, 0.0);
    let tRot3 = d.vec3f(0.0, 0.0, 0.0);
    let tScl3 = d.vec3f(1.0, 1.0, 1.0);
    let tPos4 = d.vec3f(0.0, 0.0, 0.0);
    let tRot4 = d.vec3f(0.0, 0.0, 0.0);
    let tScl4 = d.vec3f(1.0, 1.0, 1.0);
    let tPos5 = d.vec3f(0.0, 0.0, 0.0);
    let tRot5 = d.vec3f(0.0, 0.0, 0.0);
    let tScl5 = d.vec3f(1.0, 1.0, 1.0);
    let tPos6 = d.vec3f(0.0, 0.0, 0.0);
    let tRot6 = d.vec3f(0.0, 0.0, 0.0);
    let tScl6 = d.vec3f(1.0, 1.0, 1.0);
    let tPos7 = d.vec3f(0.0, 0.0, 0.0);
    let tRot7 = d.vec3f(0.0, 0.0, 0.0);
    let tScl7 = d.vec3f(1.0, 1.0, 1.0);
    let tPos8 = d.vec3f(0.0, 0.0, 0.0);
    let tRot8 = d.vec3f(0.0, 0.0, 0.0);
    let tScl8 = d.vec3f(1.0, 1.0, 1.0);
    let tPos9 = d.vec3f(0.0, 0.0, 0.0);
    let tRot9 = d.vec3f(0.0, 0.0, 0.0);
    let tScl9 = d.vec3f(1.0, 1.0, 1.0);
    let tPos10 = d.vec3f(0.0, 0.0, 0.0);
    let tRot10 = d.vec3f(0.0, 0.0, 0.0);
    let tScl10 = d.vec3f(1.0, 1.0, 1.0);
    let tPos11 = d.vec3f(0.0, 0.0, 0.0);
    let tRot11 = d.vec3f(0.0, 0.0, 0.0);
    let tScl11 = d.vec3f(1.0, 1.0, 1.0);
    let tPos12 = d.vec3f(0.0, 0.0, 0.0);
    let tRot12 = d.vec3f(0.0, 0.0, 0.0);
    let tScl12 = d.vec3f(1.0, 1.0, 1.0);
    let tPos13 = d.vec3f(0.0, 0.0, 0.0);
    let tRot13 = d.vec3f(0.0, 0.0, 0.0);
    let tScl13 = d.vec3f(1.0, 1.0, 1.0);
    let tPos14 = d.vec3f(0.0, 0.0, 0.0);
    let tRot14 = d.vec3f(0.0, 0.0, 0.0);
    let tScl14 = d.vec3f(1.0, 1.0, 1.0);
    let tPos15 = d.vec3f(0.0, 0.0, 0.0);
    let tRot15 = d.vec3f(0.0, 0.0, 0.0);
    let tScl15 = d.vec3f(1.0, 1.0, 1.0);
    let tsp = d.u32(0);
    let accScl = d.vec3f(1.0, 1.0, 1.0);

    const end = start + count;
    for (let i = start; i < end; i += d.u32(1)) {
      const instr = pickInstructionsBuffer.$[i];

      if (instr.opcode === d.u32(0)) {
        let pEval = d.vec3f(p.x, p.y, p.z);
        if (tsp > d.u32(0))
          pEval = applyParentTransform(pEval, tPos0, tRot0, tScl0);
        if (tsp > d.u32(1))
          pEval = applyParentTransform(pEval, tPos1, tRot1, tScl1);
        if (tsp > d.u32(2))
          pEval = applyParentTransform(pEval, tPos2, tRot2, tScl2);
        if (tsp > d.u32(3))
          pEval = applyParentTransform(pEval, tPos3, tRot3, tScl3);
        if (tsp > d.u32(4))
          pEval = applyParentTransform(pEval, tPos4, tRot4, tScl4);
        if (tsp > d.u32(5))
          pEval = applyParentTransform(pEval, tPos5, tRot5, tScl5);
        if (tsp > d.u32(6))
          pEval = applyParentTransform(pEval, tPos6, tRot6, tScl6);
        if (tsp > d.u32(7))
          pEval = applyParentTransform(pEval, tPos7, tRot7, tScl7);
        if (tsp > d.u32(8))
          pEval = applyParentTransform(pEval, tPos8, tRot8, tScl8);
        if (tsp > d.u32(9))
          pEval = applyParentTransform(pEval, tPos9, tRot9, tScl9);
        if (tsp > d.u32(10))
          pEval = applyParentTransform(pEval, tPos10, tRot10, tScl10);
        if (tsp > d.u32(11))
          pEval = applyParentTransform(pEval, tPos11, tRot11, tScl11);
        if (tsp > d.u32(12))
          pEval = applyParentTransform(pEval, tPos12, tRot12, tScl12);
        if (tsp > d.u32(13))
          pEval = applyParentTransform(pEval, tPos13, tRot13, tScl13);
        if (tsp > d.u32(14))
          pEval = applyParentTransform(pEval, tPos14, tRot14, tScl14);
        if (tsp > d.u32(15))
          pEval = applyParentTransform(pEval, tPos15, tRot15, tScl15);
        const lp = applyShapeRotationUnsheared(
          pEval - instr.position,
          instr.rotation,
          accScl,
        );
        let val = evalShape(lp, instr.shapeType, instr.params);
        val = val * minVec3(accScl);
        if (sp === d.u32(0)) s0 = val;
        else if (sp === d.u32(1)) s1 = val;
        else if (sp === d.u32(2)) s2 = val;
        else if (sp === d.u32(3)) s3 = val;
        else if (sp === d.u32(4)) s4 = val;
        else if (sp === d.u32(5)) s5 = val;
        else if (sp === d.u32(6)) s6 = val;
        else s7 = val;
        sp += d.u32(1);
      } else if (instr.opcode === d.u32(2)) {
        if (tsp === d.u32(0)) {
          tPos0 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot0 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl0 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(1)) {
          tPos1 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot1 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl1 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(2)) {
          tPos2 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot2 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl2 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(3)) {
          tPos3 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot3 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl3 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(4)) {
          tPos4 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot4 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl4 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(5)) {
          tPos5 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot5 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl5 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(6)) {
          tPos6 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot6 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl6 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(7)) {
          tPos7 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot7 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl7 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(8)) {
          tPos8 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot8 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl8 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(9)) {
          tPos9 = d.vec3f(instr.position.x, instr.position.y, instr.position.z);
          tRot9 = d.vec3f(instr.rotation.x, instr.rotation.y, instr.rotation.z);
          tScl9 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(10)) {
          tPos10 = d.vec3f(
            instr.position.x,
            instr.position.y,
            instr.position.z,
          );
          tRot10 = d.vec3f(
            instr.rotation.x,
            instr.rotation.y,
            instr.rotation.z,
          );
          tScl10 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(11)) {
          tPos11 = d.vec3f(
            instr.position.x,
            instr.position.y,
            instr.position.z,
          );
          tRot11 = d.vec3f(
            instr.rotation.x,
            instr.rotation.y,
            instr.rotation.z,
          );
          tScl11 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(12)) {
          tPos12 = d.vec3f(
            instr.position.x,
            instr.position.y,
            instr.position.z,
          );
          tRot12 = d.vec3f(
            instr.rotation.x,
            instr.rotation.y,
            instr.rotation.z,
          );
          tScl12 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(13)) {
          tPos13 = d.vec3f(
            instr.position.x,
            instr.position.y,
            instr.position.z,
          );
          tRot13 = d.vec3f(
            instr.rotation.x,
            instr.rotation.y,
            instr.rotation.z,
          );
          tScl13 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(14)) {
          tPos14 = d.vec3f(
            instr.position.x,
            instr.position.y,
            instr.position.z,
          );
          tRot14 = d.vec3f(
            instr.rotation.x,
            instr.rotation.y,
            instr.rotation.z,
          );
          tScl14 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        } else if (tsp === d.u32(15)) {
          tPos15 = d.vec3f(
            instr.position.x,
            instr.position.y,
            instr.position.z,
          );
          tRot15 = d.vec3f(
            instr.rotation.x,
            instr.rotation.y,
            instr.rotation.z,
          );
          tScl15 = d.vec3f(instr.params.x, instr.params.y, instr.params.z);
        }
        accScl = d.vec3f(
          accScl.x * instr.params.x,
          accScl.y * instr.params.y,
          accScl.z * instr.params.z,
        );
        tsp += d.u32(1);
      } else if (instr.opcode === d.u32(3)) {
        if (tsp === d.u32(1)) accScl = divAccSclBy(accScl, tScl0);
        else if (tsp === d.u32(2)) accScl = divAccSclBy(accScl, tScl1);
        else if (tsp === d.u32(3)) accScl = divAccSclBy(accScl, tScl2);
        else if (tsp === d.u32(4)) accScl = divAccSclBy(accScl, tScl3);
        else if (tsp === d.u32(5)) accScl = divAccSclBy(accScl, tScl4);
        else if (tsp === d.u32(6)) accScl = divAccSclBy(accScl, tScl5);
        else if (tsp === d.u32(7)) accScl = divAccSclBy(accScl, tScl6);
        else if (tsp === d.u32(8)) accScl = divAccSclBy(accScl, tScl7);
        else if (tsp === d.u32(9)) accScl = divAccSclBy(accScl, tScl8);
        else if (tsp === d.u32(10)) accScl = divAccSclBy(accScl, tScl9);
        else if (tsp === d.u32(11)) accScl = divAccSclBy(accScl, tScl10);
        else if (tsp === d.u32(12)) accScl = divAccSclBy(accScl, tScl11);
        else if (tsp === d.u32(13)) accScl = divAccSclBy(accScl, tScl12);
        else if (tsp === d.u32(14)) accScl = divAccSclBy(accScl, tScl13);
        else if (tsp === d.u32(15)) accScl = divAccSclBy(accScl, tScl14);
        else if (tsp === d.u32(16)) accScl = divAccSclBy(accScl, tScl15);
        if (tsp > d.u32(0)) tsp -= d.u32(1);
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


  const resolvePickObjectId = (p: d.v3f): d.u32 => {
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
  const selectionOutlineMask = (p: d.v3f, rd: d.v3f): number => {
    "use gpu";
    const e = 0.001;
    const dx =
      evalSelectionDist(p + d.vec3f(e, 0.0, 0.0)) -
      evalSelectionDist(p - d.vec3f(e, 0.0, 0.0));
    const dy =
      evalSelectionDist(p + d.vec3f(0.0, e, 0.0)) -
      evalSelectionDist(p - d.vec3f(0.0, e, 0.0));
    const dz =
      evalSelectionDist(p + d.vec3f(0.0, 0.0, e)) -
      evalSelectionDist(p - d.vec3f(0.0, 0.0, e));
    const grad = d.vec3f(dx, dy, dz);
    const N = std.normalize(grad);
    const gradMag = std.length(grad) / (2.0 * e);
    const viewDir = std.normalize(d.vec3f(-rd.x, -rd.y, -rd.z));
    const rim = std.pow(1.0 - std.abs(std.dot(N, viewDir)), OUTLINE_RIM_POWER);
    const gradEdge = std.smoothstep(OUTLINE_GRAD_LO, OUTLINE_GRAD_HI, gradMag);
    const edge = std.max(rim, gradEdge);
    return std.smoothstep(OUTLINE_EDGE_LO, OUTLINE_EDGE_HI, edge);
  };

  const rayMarchSelectionOutline = (ro: d.v3f, rd: d.v3f): number => {
    "use gpu";
    let t = d.f32(0.0);
    for (let i = d.f32(0.0); i < d.f32(24.0); i += d.f32(1.0)) {
      const p = ro + rd * t;
      const dSel = evalSelectionDist(p);
      const dist = std.abs(dSel - OUTLINE_OFFSET) - OUTLINE_BAND;
      t += dist;
      if (dist < 0.0001 || t > 100.0) {
        break;
      }
    }
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

  const evalGizmo = (p: d.v3f): d.v2f => {
    "use gpu";
    if (gizmoUniforms.$.mode === d.u32(GIZMO_MODE_ROTATE)) {
      return evalRotateGizmo(p);
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
      (active === d.u32(3) && axisId === d.u32(3));
    let boost = d.f32(1.0);
    if (isActive) {
      boost = d.f32(1.35);
    }
    if (axisId === d.u32(1)) {
      return d.vec3f(1.0, 0.28, 0.28) * boost;
    }
    if (axisId === d.u32(2)) {
      return d.vec3f(0.28, 1.0, 0.28) * boost;
    }
    return d.vec3f(0.35, 0.55, 1.0) * boost;
  };

  const GIZMO_HIT_EPS = 0.00015;

  const rayMarchGizmo = (ro: d.v3f, rd: d.v3f): d.v2f => {
    "use gpu";
    let t = d.f32(0.0);
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
      if (t > 100.0) {
        break;
      }
    }
    if (!converged) {
      return d.vec2f(RAY_MISS_T + 1.0, 0.0);
    }
    return d.vec2f(t, hitAxis);
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
    const iterations = sceneResult.y;

    const sceneHit = tScene <= RAY_MISS_T;

    let tOutline = d.f32(RAY_MISS_T + 1.0);
    if (selectionUniforms.$.enabled === d.u32(1)) {
      tOutline = rayMarchSelectionOutline(ro, rd);
    }

    const outlineHit = tOutline <= RAY_MISS_T;
    const outlineVisible =
      selectionUniforms.$.enabled === d.u32(1) &&
      outlineHit &&
      (!sceneHit || tOutline < tScene - 0.0001);

    let col = d.vec3f(0.05, 0.05, 0.08);

    if (sceneHit) {
      const pos = ro + rd * tScene;
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
  };
}
