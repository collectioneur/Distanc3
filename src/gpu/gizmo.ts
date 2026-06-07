import {
  findItem,
  getAncestorGroups,
  useSceneStore,
  type ObjectGroup,
  type SceneRoot,
} from "../store/sceneStore";
import {
  computeCameraRayFromClient,
  worldToClient,
  worldToView,
  type CameraRay,
  type Vec3,
} from "./camera";

export type GizmoAxis = "x" | "y" | "z";

export const GIZMO_SCALE_FACTOR = 0.15;
/** Floor for projected axis length (px) — stabilizes foreshortened Z drag speed. */
export const MIN_SCREEN_AXIS_DRAG_PX = 24;

/** Screen drag sign per axis (+Z is mirrored vs its screen projection). */
export const GIZMO_DRAG_AXIS_SIGN: Record<GizmoAxis, number> = {
  x: 1,
  y: 1,
  z: -1,
};

/** Ring rotation sign per axis (screen Y-down vs world Y-up; Z unlike X/Y). */
export const GIZMO_ROTATE_AXIS_SIGN: Record<GizmoAxis, number> = {
  x: -1,
  y: -1,
  z: 1,
};
export const GIZMO_ARROW_LENGTH_RATIO = 0.85;
export const GIZMO_SHAFT_RADIUS_RATIO = 0.035;
export const GIZMO_HEAD_RADIUS_RATIO = 0.08;
export const GIZMO_HEAD_LENGTH_RATIO = 0.18;
/** Must match shader GIZMO_HIT_EPS. */
export const GIZMO_HIT_EPS = 0.00015;
/** Screen-space fallback tolerance in CSS pixels (3D pick is primary). */
export const GIZMO_PICK_PIXELS_SHAFT = 10;
export const GIZMO_PICK_PIXELS_HEAD = 14;
export const GIZMO_PICK_PIXELS_RING = 12;
/** Wider tolerance to keep hover cursor stable between GPU single-pixel hits. */
export const GIZMO_HOVER_STICKY_PIXELS_SHAFT = 14;
export const GIZMO_HOVER_STICKY_PIXELS_HEAD = 18;
export const GIZMO_HOVER_STICKY_PIXELS_RING = 16;
/** Visual-only: larger arrows; drawn on top of scene (no camera offset). */
export const GIZMO_RENDER_SCALE_MULT = 1.35;
export const GIZMO_CAMERA_PUSH = 0;

const DEG_TO_RAD = Math.PI / 180;
const RAD_TO_DEG = 180 / Math.PI;

const AXIS_DIRS: Record<GizmoAxis, Vec3> = {
  x: [1, 0, 0],
  y: [0, 1, 0],
  z: [0, 0, 1],
};

function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scaleVec(v: Vec3, s: number): Vec3 {
  return [v[0] * s, v[1] * s, v[2] * s];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function len(v: Vec3): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function rotXYZ(v: Vec3, rot: Vec3): Vec3 {
  const cx = Math.cos(rot[0]);
  const sx = Math.sin(rot[0]);
  let p: Vec3 = [v[0], cx * v[1] - sx * v[2], sx * v[1] + cx * v[2]];

  const cy = Math.cos(rot[1]);
  const sy = Math.sin(rot[1]);
  p = [cy * p[0] + sy * p[2], p[1], -sy * p[0] + cy * p[2]];

  const cz = Math.cos(rot[2]);
  const sz = Math.sin(rot[2]);
  return [cz * p[0] - sz * p[1], sz * p[0] + cz * p[1], p[2]];
}

/** Inverse of rotXYZ (X→Y→Z); matches shader applyInvRotXYZ order. */
function invRotXYZ(v: Vec3, rot: Vec3): Vec3 {
  const czn = Math.cos(-rot[2]);
  const szn = Math.sin(-rot[2]);
  const p1: Vec3 = [
    czn * v[0] - szn * v[1],
    szn * v[0] + czn * v[1],
    v[2],
  ];

  const cyn = Math.cos(-rot[1]);
  const syn = Math.sin(-rot[1]);
  const p2: Vec3 = [
    cyn * p1[0] + syn * p1[2],
    p1[1],
    -syn * p1[0] + cyn * p1[2],
  ];

  const cxn = Math.cos(-rot[0]);
  const sxn = Math.sin(-rot[0]);
  return [p2[0], cxn * p2[1] - sxn * p2[2], sxn * p2[1] + cxn * p2[2]];
}

function groupRotRad(group: ObjectGroup): Vec3 {
  return [
    group.rotation[0] * DEG_TO_RAD,
    group.rotation[1] * DEG_TO_RAD,
    group.rotation[2] * DEG_TO_RAD,
  ];
}

function applyGroupTransform(local: Vec3, group: ObjectGroup): Vec3 {
  const scaled: Vec3 = [
    local[0] * group.scale[0],
    local[1] * group.scale[1],
    local[2] * group.scale[2],
  ];
  return add(rotXYZ(scaled, groupRotRad(group)), group.position);
}

function inverseGroupTransform(world: Vec3, group: ObjectGroup): Vec3 {
  const q = sub(world, group.position);
  const r = invRotXYZ(q, groupRotRad(group));
  return [r[0] / group.scale[0], r[1] / group.scale[1], r[2] / group.scale[2]];
}

export function getItemAncestorGroups(root: SceneRoot, itemId: string): ObjectGroup[] | null {
  const found = findItem(root, itemId);
  if (!found) return null;
  return getAncestorGroups(root, found.container.id);
}

export function getGizmoWorldPosition(root: SceneRoot, itemId: string): Vec3 | null {
  const found = findItem(root, itemId);
  if (!found) return null;

  const ancestors = getAncestorGroups(root, found.container.id);
  return itemLocalToWorldPosition([...found.item.position], ancestors);
}

export const GIZMO_MODE_TRANSLATE = 0;
export const GIZMO_MODE_ROTATE = 1;

export const GIZMO_RING_MAJOR_RATIO = 0.85;
export const GIZMO_RING_TUBE_RATIO = 0.035;
/**
 * Edge-on rings project to a long screen line; screen pick falsely hits far
 * outside the visible torus below this alignment with the view ray.
 */
const RING_SCREEN_PICK_FACE_ON_MIN = 0.35;

export type GizmoWorldAxes = { x: Vec3; y: Vec3; z: Vec3 };
export type GizmoPickMode = "translate" | "rotate";

const RING_SCREEN_SEGMENTS = 32;

/** World-space X/Y/Z basis of the selected item (ancestor + own Euler XYZ). */
export function getGizmoWorldAxes(
  root: SceneRoot,
  itemId: string,
): GizmoWorldAxes | null {
  const found = findItem(root, itemId);
  if (!found) return null;

  const ancestors = getAncestorGroups(root, found.container.id);
  const applyEuler = (v: Vec3, rotDeg: [number, number, number]): Vec3 => {
    const rot: Vec3 = [
      rotDeg[0] * DEG_TO_RAD,
      rotDeg[1] * DEG_TO_RAD,
      rotDeg[2] * DEG_TO_RAD,
    ];
    return rotXYZ(v, rot);
  };

  let x: Vec3 = [1, 0, 0];
  let y: Vec3 = [0, 1, 0];
  let z: Vec3 = [0, 0, 1];

  // Item rotation first, then ancestors inner → outer (matches shader: R_parent * … * R_item).
  x = applyEuler(x, found.item.rotation);
  y = applyEuler(y, found.item.rotation);
  z = applyEuler(z, found.item.rotation);

  for (let i = ancestors.length - 1; i >= 0; i--) {
    x = applyEuler(x, ancestors[i].rotation);
    y = applyEuler(y, ancestors[i].rotation);
    z = applyEuler(z, ancestors[i].rotation);
  }

  const nx = len(x);
  const ny = len(y);
  const nz = len(z);
  if (nx > 1e-8) x = scaleVec(x, 1 / nx);
  if (ny > 1e-8) y = scaleVec(y, 1 / ny);
  if (nz > 1e-8) z = scaleVec(z, 1 / nz);

  return { x, y, z };
}

export function worldToItemLocalPosition(
  worldPos: Vec3,
  ancestors: ObjectGroup[],
): Vec3 {
  let p = worldPos;
  for (const group of ancestors) {
    p = inverseGroupTransform(p, group);
  }
  return p;
}

/** Inner-parent-first forward — matches shader transform stack order. */
export function itemLocalToWorldPosition(
  localPos: Vec3,
  ancestors: ObjectGroup[],
): Vec3 {
  let p = localPos;
  for (let i = ancestors.length - 1; i >= 0; i--) {
    p = applyGroupTransform(p, ancestors[i]);
  }
  return p;
}

export function gizmoScaleForDistance(distance: number): number {
  return distance * GIZMO_SCALE_FACTOR;
}

/** World-space arrow size used by both shader and CPU pick. */
export function gizmoVisualScaleForDistance(distance: number): number {
  return gizmoScaleForDistance(distance) * GIZMO_RENDER_SCALE_MULT;
}

/** @deprecated Use gizmoVisualScaleForDistance — kept for call-site clarity. */
export function gizmoRenderScaleForDistance(distance: number): number {
  return gizmoVisualScaleForDistance(distance);
}

/** Unit vector from pivot toward camera. */
export function computeGizmoPushDir(
  pivotWorld: Vec3,
  cameraWorld: Vec3,
): Vec3 {
  const toCam = sub(cameraWorld, pivotWorld);
  const dist = len(toCam);
  if (dist < 1e-8) return [0, 0, 1];
  return scaleVec(toCam, 1 / dist);
}

/** Display position with a fixed push direction (stable while pivot moves). */
export function getGizmoRenderPosition(
  pivotWorld: Vec3,
  pushDir: Vec3,
  gizmoScale: number,
): Vec3 {
  if (GIZMO_CAMERA_PUSH <= 0) return pivotWorld;
  return add(pivotWorld, scaleVec(pushDir, gizmoScale * GIZMO_CAMERA_PUSH));
}

function sdCapsuleSeg(p: Vec3, a: Vec3, b: Vec3, r: number): number {
  const pa = sub(p, a);
  const ba = sub(b, a);
  const denom = dot(ba, ba);
  const h = denom > 1e-12 ? Math.max(0, Math.min(1, dot(pa, ba) / denom)) : 0;
  const closest = sub(pa, scaleVec(ba, h));
  return len(closest) - r;
}

function sdSphere(p: Vec3, center: Vec3, radius: number): number {
  return len(sub(p, center)) - radius;
}

function axisArrowDistance(gp: Vec3, dir: Vec3, gizmoScale: number): number {
  const arrowLen = gizmoScale * GIZMO_ARROW_LENGTH_RATIO;
  const shaftR = gizmoScale * GIZMO_SHAFT_RADIUS_RATIO;
  const headR = gizmoScale * GIZMO_HEAD_RADIUS_RATIO;
  const headLen = gizmoScale * GIZMO_HEAD_LENGTH_RATIO;

  const tip = scaleVec(dir, arrowLen);
  const shaftEnd = scaleVec(dir, arrowLen - headLen);
  const dShaft = sdCapsuleSeg(gp, [0, 0, 0], shaftEnd, shaftR);
  const dHead = sdSphere(gp, tip, headR);
  return Math.min(dShaft, dHead);
}

function distPointToSeg2d(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): number {
  const abx = bx - ax;
  const aby = by - ay;
  const apx = px - ax;
  const apy = py - ay;
  const denom = abx * abx + aby * aby;
  const t =
    denom > 1e-12
      ? Math.max(0, Math.min(1, (apx * abx + apy * aby) / denom))
      : 0;
  const cx = ax + abx * t;
  const cy = ay + aby * t;
  return Math.hypot(px - cx, py - cy);
}

function maxArrowScreenPixels(
  canvas: HTMLCanvasElement,
  viewZ: number,
  arrowLen: number,
): number {
  const rect = canvas.getBoundingClientRect();
  const pxPerUnit = rect.height / (2 * Math.max(viewZ, 0.05));
  return arrowLen * pxPerUnit * 2.5;
}

/** 3D SDF pick — same geometry + hit epsilon as shader rayMarchGizmo. */
export function hitTestTranslateGizmo(
  ray: CameraRay,
  gizmoPos: Vec3,
  gizmoScale: number,
): GizmoAxis | null {
  let bestAxis: GizmoAxis | null = null;
  let bestDist = Infinity;
  const hitEps = GIZMO_HIT_EPS;

  let t = 0;
  for (let i = 0; i < 64; i++) {
    const p = add(ray.origin, scaleVec(ray.direction, t));
    const gp = sub(p, gizmoPos);

    for (const axis of ["x", "y", "z"] as const) {
      const d = axisArrowDistance(gp, AXIS_DIRS[axis], gizmoScale);
      if (d < hitEps && d < bestDist) {
        bestDist = d;
        bestAxis = axis;
      }
    }
    if (bestAxis) break;

    let minD = Infinity;
    for (const axis of ["x", "y", "z"] as const) {
      minD = Math.min(minD, axisArrowDistance(gp, AXIS_DIRS[axis], gizmoScale));
    }
    t += Math.max(minD * 0.85, 0.0005);
    if (t > 100) break;
  }

  return bestAxis;
}

function worldAxisRingDistance(
  gp: Vec3,
  axis: Vec3,
  gizmoScale: number,
): number {
  const majorR = gizmoScale * GIZMO_RING_MAJOR_RATIO;
  const tubeR = gizmoScale * GIZMO_RING_TUBE_RATIO;
  const along = dot(gp, axis);
  const perp = sub(gp, scaleVec(axis, along));
  const q2 = len(perp) - majorR;
  return Math.hypot(q2, along) - tubeR;
}

/** 3D SDF pick — prefers the ring most face-on to the camera when several overlap. */
export function hitTestRotateGizmo(
  ray: CameraRay,
  gizmoPos: Vec3,
  gizmoScale: number,
  axes: GizmoWorldAxes,
): GizmoAxis | null {
  let bestAxis: GizmoAxis | null = null;
  let bestScore = -Infinity;
  const hitEps = GIZMO_HIT_EPS;
  const axisVecs: Record<GizmoAxis, Vec3> = {
    x: axes.x,
    y: axes.y,
    z: axes.z,
  };

  let t = 0;
  for (let i = 0; i < 64; i++) {
    const p = add(ray.origin, scaleVec(ray.direction, t));
    const gp = sub(p, gizmoPos);

    for (const axis of ["x", "y", "z"] as const) {
      const d = worldAxisRingDistance(gp, axisVecs[axis], gizmoScale);
      if (d < hitEps) {
        const axisLen = len(axisVecs[axis]);
        const faceOn =
          axisLen > 1e-8
            ? Math.abs(dot(scaleVec(axisVecs[axis], 1 / axisLen), ray.direction))
            : 0;
        const score = faceOn - t * 1e-4;
        if (score > bestScore) {
          bestScore = score;
          bestAxis = axis;
        }
      }
    }

    let minD = Infinity;
    for (const axis of ["x", "y", "z"] as const) {
      minD = Math.min(
        minD,
        worldAxisRingDistance(gp, axisVecs[axis], gizmoScale),
      );
    }
    t += Math.max(minD * 0.85, 0.0005);
    if (t > 100) break;
  }

  return bestAxis;
}

/** 1 = ring plane faces camera; 0 = edge-on. */
function ringAxisFaceOn(axis: Vec3, rayDir: Vec3): number {
  const axisLen = len(axis);
  if (axisLen < 1e-8) return 0;
  return Math.abs(dot(scaleVec(axis, 1 / axisLen), rayDir));
}

function isRingScreenPickable(axis: Vec3, rayDir: Vec3): boolean {
  return ringAxisFaceOn(axis, rayDir) >= RING_SCREEN_PICK_FACE_ON_MIN;
}

function ringPlaneBasis(axis: Vec3): [Vec3, Vec3] | null {
  const hint: Vec3 = Math.abs(axis[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
  const uRaw = [
    hint[1] * axis[2] - hint[2] * axis[1],
    hint[2] * axis[0] - hint[0] * axis[2],
    hint[0] * axis[1] - hint[1] * axis[0],
  ] as Vec3;
  const uLen = len(uRaw);
  if (uLen < 1e-8) return null;
  const u = scaleVec(uRaw, 1 / uLen);
  const v: Vec3 = [
    axis[1] * u[2] - axis[2] * u[1],
    axis[2] * u[0] - axis[0] * u[2],
    axis[0] * u[1] - axis[1] * u[0],
  ];
  return [u, v];
}

function ringScreenDistance(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
  rotX: number,
  rotY: number,
  distance: number,
  gizmoPos: Vec3,
  gizmoScale: number,
  axis: Vec3,
): number | null {
  const basis = ringPlaneBasis(axis);
  if (!basis) return null;

  const [u, v] = basis;
  const majorR = gizmoScale * GIZMO_RING_MAJOR_RATIO;
  let best = Infinity;

  for (let i = 0; i < RING_SCREEN_SEGMENTS; i++) {
    const a0 = (2 * Math.PI * i) / RING_SCREEN_SEGMENTS;
    const a1 = (2 * Math.PI * (i + 1)) / RING_SCREEN_SEGMENTS;
    const p0 = add(
      gizmoPos,
      add(scaleVec(u, majorR * Math.cos(a0)), scaleVec(v, majorR * Math.sin(a0))),
    );
    const p1 = add(
      gizmoPos,
      add(scaleVec(u, majorR * Math.cos(a1)), scaleVec(v, majorR * Math.sin(a1))),
    );
    const s0 = worldToClient(p0, canvas, rotX, rotY, distance);
    const s1 = worldToClient(p1, canvas, rotX, rotY, distance);
    if (!s0 || !s1) continue;

    const d = distPointToSeg2d(
      clientX,
      clientY,
      s0.x,
      s0.y,
      s1.x,
      s1.y,
    );
    if (d < best) best = d;
  }

  return best === Infinity ? null : best;
}

/** Screen-space ring pick first (matches visible overlap); 3D fallback. */
export function hitTestRotateGizmoScreen(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
  rotX: number,
  rotY: number,
  distance: number,
  gizmoPos: Vec3,
  gizmoScale: number,
  axes: GizmoWorldAxes,
): GizmoAxis | null {
  const ray = computeCameraRayFromClient(
    clientX,
    clientY,
    canvas,
    rotX,
    rotY,
    distance,
  );
  if (!ray) return null;

  const axisVecs: Record<GizmoAxis, Vec3> = {
    x: axes.x,
    y: axes.y,
    z: axes.z,
  };

  let bestAxis: GizmoAxis | null = null;
  let bestDist = Infinity;

  for (const axis of ["x", "y", "z"] as const) {
    const axisDir = axisVecs[axis];
    if (!isRingScreenPickable(axisDir, ray.direction)) continue;

    const d = ringScreenDistance(
      clientX,
      clientY,
      canvas,
      rotX,
      rotY,
      distance,
      gizmoPos,
      gizmoScale,
      axisDir,
    );
    if (d === null || d >= GIZMO_PICK_PIXELS_RING || d >= bestDist) continue;
    bestDist = d;
    bestAxis = axis;
  }

  if (bestAxis) return bestAxis;

  return hitTestRotateGizmo(ray, gizmoPos, gizmoScale, axes);
}

/** Mode-aware 3D + screen pick. */
export function hitTestGizmoScreen(
  mode: GizmoPickMode,
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
  rotX: number,
  rotY: number,
  distance: number,
  gizmoPos: Vec3,
  gizmoScale: number,
  axes: GizmoWorldAxes | null,
): GizmoAxis | null {
  if (mode === "rotate") {
    if (!axes) return null;
    return hitTestRotateGizmoScreen(
      clientX,
      clientY,
      canvas,
      rotX,
      rotY,
      distance,
      gizmoPos,
      gizmoScale,
      axes,
    );
  }
  return hitTestTranslateGizmoScreen(
    clientX,
    clientY,
    canvas,
    rotX,
    rotY,
    distance,
    gizmoPos,
    gizmoScale,
  );
}

/** 3D pick first; screen-space fallback with degenerate-segment rejection. */
export function hitTestTranslateGizmoScreen(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
  rotX: number,
  rotY: number,
  distance: number,
  gizmoPos: Vec3,
  gizmoScale: number,
): GizmoAxis | null {
  const ray = computeCameraRayFromClient(
    clientX,
    clientY,
    canvas,
    rotX,
    rotY,
    distance,
  );
  if (!ray) return null;

  const pick3d = hitTestTranslateGizmo(ray, gizmoPos, gizmoScale);
  if (pick3d) return pick3d;

  const origin = worldToClient(gizmoPos, canvas, rotX, rotY, distance);
  const originView = worldToView(gizmoPos, rotX, rotY, distance);
  if (!origin || !originView) return null;

  const arrowLen = gizmoScale * GIZMO_ARROW_LENGTH_RATIO;
  const headLen = gizmoScale * GIZMO_HEAD_LENGTH_RATIO;

  let bestAxis: GizmoAxis | null = null;
  let bestDist = Infinity;

  for (const axis of ["x", "y", "z"] as const) {
    const tipWorld = add(gizmoPos, scaleVec(AXIS_DIRS[axis], arrowLen));
    const tip = worldToClient(tipWorld, canvas, rotX, rotY, distance);
    const tipView = worldToView(tipWorld, rotX, rotY, distance);
    if (!tip || !tipView) continue;

    const segLen = Math.hypot(tip.x - origin.x, tip.y - origin.y);
    const maxScreenLen = maxArrowScreenPixels(
      canvas,
      Math.min(originView[2], tipView[2]),
      arrowLen,
    );
    if (segLen < 2 || segLen > maxScreenLen) continue;

    const d = distPointToSeg2d(
      clientX,
      clientY,
      origin.x,
      origin.y,
      tip.x,
      tip.y,
    );

    const headStart = 1 - headLen / arrowLen;
    const abx = tip.x - origin.x;
    const aby = tip.y - origin.y;
    const denom = abx * abx + aby * aby;
    const t =
      denom > 1e-12
        ? Math.max(
            0,
            Math.min(
              1,
              ((clientX - origin.x) * abx + (clientY - origin.y) * aby) / denom,
            ),
          )
        : 0;
    const pickRadius =
      t >= headStart ? GIZMO_PICK_PIXELS_HEAD : GIZMO_PICK_PIXELS_SHAFT;

    if (d >= pickRadius || d >= bestDist) continue;

    bestDist = d;
    bestAxis = axis;
  }

  return bestAxis;
}

function axisScreenSegmentPick(
  clientX: number,
  clientY: number,
  axis: GizmoAxis,
  canvas: HTMLCanvasElement,
  rotX: number,
  rotY: number,
  distance: number,
  gizmoPos: Vec3,
  gizmoScale: number,
): { dist: number; onHead: boolean } | null {
  const origin = worldToClient(gizmoPos, canvas, rotX, rotY, distance);
  if (!origin) return null;

  const arrowLen = gizmoScale * GIZMO_ARROW_LENGTH_RATIO;
  const headLen = gizmoScale * GIZMO_HEAD_LENGTH_RATIO;
  const tipWorld = add(gizmoPos, scaleVec(AXIS_DIRS[axis], arrowLen));
  const tip = worldToClient(tipWorld, canvas, rotX, rotY, distance);
  if (!tip) return null;

  const dist = distPointToSeg2d(
    clientX,
    clientY,
    origin.x,
    origin.y,
    tip.x,
    tip.y,
  );
  const abx = tip.x - origin.x;
  const aby = tip.y - origin.y;
  const denom = abx * abx + aby * aby;
  const t =
    denom > 1e-12
      ? Math.max(
          0,
          Math.min(
            1,
            ((clientX - origin.x) * abx + (clientY - origin.y) * aby) / denom,
          ),
        )
      : 0;
  const headStart = 1 - headLen / arrowLen;
  return { dist, onHead: t >= headStart };
}

function ringScreenAxisPick(
  clientX: number,
  clientY: number,
  axis: GizmoAxis,
  canvas: HTMLCanvasElement,
  rotX: number,
  rotY: number,
  distance: number,
  gizmoPos: Vec3,
  gizmoScale: number,
  axes: GizmoWorldAxes,
  rayDir: Vec3,
): { dist: number } | null {
  const axisDir = axes[axis];
  if (!isRingScreenPickable(axisDir, rayDir)) return null;

  const dist = ringScreenDistance(
    clientX,
    clientY,
    canvas,
    rotX,
    rotY,
    distance,
    gizmoPos,
    gizmoScale,
    axisDir,
  );
  if (dist === null) return null;
  return { dist };
}

/** True when the pointer is within pick tolerance of a specific gizmo axis. */
export function isGizmoAxisNearScreen(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
  rotX: number,
  rotY: number,
  distance: number,
  gizmoPos: Vec3,
  gizmoScale: number,
  axis: GizmoAxis,
  mode: GizmoPickMode,
  axes: GizmoWorldAxes | null,
): boolean {
  if (mode === "rotate") {
    if (!axes) return false;
    const ray = computeCameraRayFromClient(
      clientX,
      clientY,
      canvas,
      rotX,
      rotY,
      distance,
    );
    if (!ray) return false;
    const ring = ringScreenAxisPick(
      clientX,
      clientY,
      axis,
      canvas,
      rotX,
      rotY,
      distance,
      gizmoPos,
      gizmoScale,
      axes,
      ray.direction,
    );
    return ring !== null && ring.dist <= GIZMO_HOVER_STICKY_PIXELS_RING;
  }

  const seg = axisScreenSegmentPick(
    clientX,
    clientY,
    axis,
    canvas,
    rotX,
    rotY,
    distance,
    gizmoPos,
    gizmoScale,
  );
  if (!seg) return false;

  const pickRadius = seg.onHead
    ? GIZMO_HOVER_STICKY_PIXELS_HEAD
    : GIZMO_HOVER_STICKY_PIXELS_SHAFT;
  return seg.dist <= pickRadius;
}

/** CPU screen pick + hover-axis stabilization (shared by hover and click). */
export function pickGizmoAxisScreen(
  mode: GizmoPickMode,
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
  rotX: number,
  rotY: number,
  distance: number,
  gizmoPos: Vec3,
  gizmoScale: number,
  axes: GizmoWorldAxes | null,
  previousAxis: GizmoAxis | null,
): GizmoAxis | null {
  const raw = hitTestGizmoScreen(
    mode,
    clientX,
    clientY,
    canvas,
    rotX,
    rotY,
    distance,
    gizmoPos,
    gizmoScale,
    axes,
  );
  return stabilizeGizmoHoverAxis(
    clientX,
    clientY,
    canvas,
    rotX,
    rotY,
    distance,
    gizmoPos,
    gizmoScale,
    raw,
    previousAxis,
    mode,
    axes,
  );
}

/** Keep hover axis when GPU single-pixel pick flickers on thin gizmo geometry. */
export function stabilizeGizmoHoverAxis(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
  rotX: number,
  rotY: number,
  distance: number,
  gizmoPos: Vec3,
  gizmoScale: number,
  gpuAxis: GizmoAxis | null,
  previousAxis: GizmoAxis | null,
  mode: GizmoPickMode,
  axes: GizmoWorldAxes | null,
): GizmoAxis | null {
  if (gpuAxis) return gpuAxis;
  if (!previousAxis) return null;

  if (mode === "rotate") {
    if (!axes) return null;
    const ray = computeCameraRayFromClient(
      clientX,
      clientY,
      canvas,
      rotX,
      rotY,
      distance,
    );
    if (!ray) return null;
    const ring = ringScreenAxisPick(
      clientX,
      clientY,
      previousAxis,
      canvas,
      rotX,
      rotY,
      distance,
      gizmoPos,
      gizmoScale,
      axes,
      ray.direction,
    );
    if (!ring) return null;
    return ring.dist <= GIZMO_HOVER_STICKY_PIXELS_RING ? previousAxis : null;
  }

  const seg = axisScreenSegmentPick(
    clientX,
    clientY,
    previousAxis,
    canvas,
    rotX,
    rotY,
    distance,
    gizmoPos,
    gizmoScale,
  );
  if (!seg) return null;

  const stickyRadius = seg.onHead
    ? GIZMO_HOVER_STICKY_PIXELS_HEAD
    : GIZMO_HOVER_STICKY_PIXELS_SHAFT;
  return seg.dist <= stickyRadius ? previousAxis : null;
}

export type AxisScreenDragState = {
  startClientX: number;
  startClientY: number;
  screenAxisDir: [number, number];
  worldPerScreenPx: number;
};

/** Capture screen-axis drag basis at pointer down (fixed for whole drag). */
export function beginAxisScreenDrag(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
  rotX: number,
  rotY: number,
  distance: number,
  pivotWorld: Vec3,
  axisDir: Vec3,
  referenceWorldLen: number,
): AxisScreenDragState | null {
  const origin = worldToClient(pivotWorld, canvas, rotX, rotY, distance);
  if (!origin) return null;

  const refWorld = add(pivotWorld, scaleVec(axisDir, referenceWorldLen));
  const tip = worldToClient(refWorld, canvas, rotX, rotY, distance);
  if (!tip) return null;

  const sx = tip.x - origin.x;
  const sy = tip.y - origin.y;
  const screenLen = Math.hypot(sx, sy);
  if (screenLen < 2) return null;

  const effectiveLen = Math.max(screenLen, MIN_SCREEN_AXIS_DRAG_PX);

  return {
    startClientX: clientX,
    startClientY: clientY,
    screenAxisDir: [sx / screenLen, sy / screenLen],
    worldPerScreenPx: referenceWorldLen / effectiveLen,
  };
}

/** World-space delta along axis from screen mouse movement. */
export function axisDeltaFromScreenDrag(
  clientX: number,
  clientY: number,
  state: AxisScreenDragState,
  axis: GizmoAxis,
): number {
  const mx = clientX - state.startClientX;
  const my = clientY - state.startClientY;
  const screenDelta =
    mx * state.screenAxisDir[0] + my * state.screenAxisDir[1];
  return (
    screenDelta * state.worldPerScreenPx * GIZMO_DRAG_AXIS_SIGN[axis]
  );
}

type Mat3 = [
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
  number,
];

function mat3FromColumns(cx: Vec3, cy: Vec3, cz: Vec3): Mat3 {
  return [cx[0], cx[1], cx[2], cy[0], cy[1], cy[2], cz[0], cz[1], cz[2]];
}

function mat3TransformVec3(m: Mat3, v: Vec3): Vec3 {
  return [
    m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
    m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
    m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
  ];
}

function mat3Mul(a: Mat3, b: Mat3): Mat3 {
  return mat3FromColumns(
    mat3TransformVec3(a, [b[0], b[1], b[2]]),
    mat3TransformVec3(a, [b[3], b[4], b[5]]),
    mat3TransformVec3(a, [b[6], b[7], b[8]]),
  );
}

function rotateVecAxisAngle(v: Vec3, axis: Vec3, angle: number): Vec3 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const d = dot(axis, v);
  return add(
    add(scaleVec(v, c), scaleVec(cross(axis, v), s)),
    scaleVec(axis, d * (1 - c)),
  );
}

function mat3FromAxisAngle(axis: Vec3, angle: number): Mat3 {
  return mat3FromColumns(
    rotateVecAxisAngle([1, 0, 0], axis, angle),
    rotateVecAxisAngle([0, 1, 0], axis, angle),
    rotateVecAxisAngle([0, 0, 1], axis, angle),
  );
}

/** Euler XYZ degrees → mat3 matching rotXYZ / shader applyInvRotXYZ inverse. */
function eulerDegreesToMat3(rotDeg: [number, number, number]): Mat3 {
  const rot: Vec3 = [
    rotDeg[0] * DEG_TO_RAD,
    rotDeg[1] * DEG_TO_RAD,
    rotDeg[2] * DEG_TO_RAD,
  ];
  return mat3FromColumns(
    rotXYZ([1, 0, 0], rot),
    rotXYZ([0, 1, 0], rot),
    rotXYZ([0, 0, 1], rot),
  );
}

/** mat3 → Euler XYZ degrees (R = Rz * Ry * Rx, column-major). */
function mat3ToEulerDegrees(m: Mat3): [number, number, number] {
  const sy = Math.max(-1, Math.min(1, -m[2]));
  const ry = Math.asin(sy);
  const cy = Math.cos(ry);

  let rx: number;
  let rz: number;

  if (Math.abs(cy) > 1e-6) {
    rx = Math.atan2(m[5] / cy, m[8] / cy);
    rz = Math.atan2(m[1] / cy, m[0] / cy);
  } else {
    rx = Math.atan2(-m[3], m[4]);
    rz = 0;
  }

  return [rx * RAD_TO_DEG, ry * RAD_TO_DEG, rz * RAD_TO_DEG];
}

function mat3Drift(a: Mat3, b: Mat3): number {
  return a.reduce((sum, v, i) => sum + Math.abs(v - b[i]), 0);
}

function eulerNearPrev(
  rot: [number, number, number],
  prev: [number, number, number],
): [number, number, number] {
  const out: [number, number, number] = [...rot];
  for (let i = 0; i < 3; i++) {
    while (out[i] - prev[i] > 180) out[i] -= 360;
    while (out[i] - prev[i] < -180) out[i] += 360;
  }
  return out;
}

function eulerDeltaSq(
  a: [number, number, number],
  b: [number, number, number],
): number {
  return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2;
}

function mat3ToEulerBranch(m: Mat3, ryRad: number): [number, number, number] {
  const cy = Math.cos(ryRad);
  let rx: number;
  let rz: number;

  if (Math.abs(cy) > 1e-6) {
    rx = Math.atan2(m[5] / cy, m[8] / cy);
    rz = Math.atan2(m[1] / cy, m[0] / cy);
  } else {
    rx = Math.atan2(-m[3], m[4]);
    rz = 0;
  }

  return [rx * RAD_TO_DEG, ryRad * RAD_TO_DEG, rz * RAD_TO_DEG];
}

/** Euler XYZ matching m, continuous with prev, stable through gimbal. */
function mat3ToEulerDegreesHinted(
  m: Mat3,
  prev: [number, number, number],
): [number, number, number] {
  const sy = Math.max(-1, Math.min(1, -m[2]));
  const ryA = Math.asin(sy);
  const ryB = Math.PI - ryA;

  const roots: [number, number, number][] = [
    mat3ToEulerBranch(m, ryA),
    mat3ToEulerBranch(m, ryB),
  ];

  const candidates: [number, number, number][] = [];
  for (const root of roots) {
    for (let kx = -2; kx <= 2; kx++) {
      for (let ky = -1; ky <= 1; ky++) {
        for (let kz = -2; kz <= 2; kz++) {
          candidates.push(
            eulerNearPrev(
              [root[0] + kx * 360, root[1] + ky * 360, root[2] + kz * 360],
              prev,
            ),
          );
        }
      }
    }
  }

  if (Math.abs(m[2]) >= 0.99999) {
    const ry = Math.sign(-m[2]) * 90;
    const sumDeg = Math.atan2(m[1], m[0]) * RAD_TO_DEG;
    const poleRx = Math.atan2(-m[3], m[4]) * RAD_TO_DEG;
    candidates.push(
      eulerNearPrev([poleRx, ry, prev[2]], prev),
      eulerNearPrev([prev[0], ry, sumDeg - prev[0]], prev),
      eulerNearPrev([sumDeg - prev[2], ry, prev[2]], prev),
      eulerNearPrev([poleRx + 180, ry, prev[2] + 180], prev),
    );
  }

  let best = eulerNearPrev(mat3ToEulerDegrees(m), prev);
  let bestDrift = mat3Drift(m, eulerDegreesToMat3(best));
  let bestScore = eulerDeltaSq(best, prev);

  for (const c of candidates) {
    const drift = mat3Drift(m, eulerDegreesToMat3(c));
    const score = eulerDeltaSq(c, prev);
    if (
      drift < bestDrift - 1e-6 ||
      (Math.abs(drift - bestDrift) < 1e-6 && score < bestScore)
    ) {
      best = c;
      bestDrift = drift;
      bestScore = score;
    }
  }

  return best;
}

const LOCAL_GIZMO_AXIS: Record<GizmoAxis, Vec3> = {
  x: [1, 0, 0],
  y: [0, 1, 0],
  z: [0, 0, 1],
};

function rayPlaneIntersect(
  origin: Vec3,
  direction: Vec3,
  planePoint: Vec3,
  planeNormal: Vec3,
): Vec3 | null {
  const denom = dot(direction, planeNormal);
  if (Math.abs(denom) < 1e-6) return null;
  const t = dot(sub(planePoint, origin), planeNormal) / denom;
  return add(origin, scaleVec(direction, t));
}

function vectorOnRingPlane(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
  rotX: number,
  rotY: number,
  distance: number,
  pivot: Vec3,
  axis: Vec3,
): Vec3 | null {
  const ray = computeCameraRayFromClient(
    clientX,
    clientY,
    canvas,
    rotX,
    rotY,
    distance,
  );
  if (!ray) return null;

  const hit = rayPlaneIntersect(ray.origin, ray.direction, pivot, axis);
  if (!hit) return null;

  const v = sub(hit, pivot);
  const onPlane = sub(v, scaleVec(axis, dot(v, axis)));
  const dist = len(onPlane);
  if (dist < 1e-6) return null;
  return scaleVec(onPlane, 1 / dist);
}

/** Signed angle from startDir to dir around axis (radians, (-π, π]). */
function signedRingAngle(startDir: Vec3, dir: Vec3, axis: Vec3): number {
  return Math.atan2(dot(cross(startDir, dir), axis), dot(startDir, dir));
}

/** Keep total drag angle continuous past ±180° (Blender-style unwrap). */
function unwrapRingAngle(angle: number, prev: number): number {
  let out = angle;
  while (out - prev > Math.PI) out -= 2 * Math.PI;
  while (out - prev < -Math.PI) out += 2 * Math.PI;
  return out;
}

export type RingDragState = {
  pivotWorld: Vec3;
  worldAxis: Vec3;
  startDir: Vec3;
  totalAngleRad: number;
  R_itemStart: Mat3;
  startEuler: [number, number, number];
  lastEuler: [number, number, number];
  dragAxis: GizmoAxis;
};

/** Capture ring-plane basis at pointer down (fixed for whole drag). */
export function beginRingDrag(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
  rotX: number,
  rotY: number,
  distance: number,
  pivotWorld: Vec3,
  worldAxis: Vec3,
  dragAxis: GizmoAxis,
  itemRotation: [number, number, number],
): RingDragState | null {
  const axisLen = len(worldAxis);
  if (axisLen < 1e-8) return null;
  const axis = scaleVec(worldAxis, 1 / axisLen);

  const startDir = vectorOnRingPlane(
    clientX,
    clientY,
    canvas,
    rotX,
    rotY,
    distance,
    pivotWorld,
    axis,
  );
  if (!startDir) return null;

  return {
    pivotWorld,
    worldAxis: axis,
    startDir,
    totalAngleRad: 0,
    R_itemStart: eulerDegreesToMat3(itemRotation),
    startEuler: [...itemRotation],
    lastEuler: [...itemRotation],
    dragAxis,
  };
}

/**
 * Total ring angle from drag start to current pointer (Blender-style).
 * Returns null when the ring plane is edge-on to the cursor ray.
 */
export function ringTotalAngleFromScreenDrag(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
  rotX: number,
  rotY: number,
  distance: number,
  state: RingDragState,
): number | null {
  const dir = vectorOnRingPlane(
    clientX,
    clientY,
    canvas,
    rotX,
    rotY,
    distance,
    state.pivotWorld,
    state.worldAxis,
  );
  if (!dir) return null;

  const raw =
    signedRingAngle(state.startDir, dir, state.worldAxis) *
    GIZMO_ROTATE_AXIS_SIGN[state.dragAxis];
  return unwrapRingAngle(raw, state.totalAngleRad);
}

export function applyWorldRotationToItem(
  root: SceneRoot,
  itemId: string,
  totalAngleRad: number,
  state: RingDragState,
): void {
  const found = findItem(root, itemId);
  if (!found) return;

  state.totalAngleRad = totalAngleRad;
  const rNew = mat3Mul(
    state.R_itemStart,
    mat3FromAxisAngle(
      LOCAL_GIZMO_AXIS[state.dragAxis],
      totalAngleRad,
    ),
  );
  const rotation = mat3ToEulerDegreesHinted(rNew, state.startEuler);
  state.lastEuler = rotation;

  const { updateLayer, updateGroup } = useSceneStore.getState();
  if (found.item.kind === "layer") {
    updateLayer(found.container.id, found.item.id, { rotation });
  } else {
    updateGroup(found.item.id, { rotation });
  }
}

export function applyWorldDeltaToItem(
  root: SceneRoot,
  itemId: string,
  worldDelta: Vec3,
): void {
  const found = findItem(root, itemId);
  if (!found) return;

  const ancestors = getAncestorGroups(root, found.container.id);
  const currentWorld = getGizmoWorldPosition(root, itemId);
  if (!currentWorld) return;

  const newWorld = add(currentWorld, worldDelta);
  const newLocal = worldToItemLocalPosition(newWorld, ancestors);

  const { updateLayer, updateGroup } = useSceneStore.getState();

  if (found.item.kind === "layer") {
    updateLayer(found.container.id, found.item.id, { position: newLocal });
  } else {
    updateGroup(found.item.id, { position: newLocal });
  }
}
