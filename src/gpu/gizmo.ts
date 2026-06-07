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
export const GIZMO_ARROW_LENGTH_RATIO = 0.85;
export const GIZMO_SHAFT_RADIUS_RATIO = 0.035;
export const GIZMO_HEAD_RADIUS_RATIO = 0.08;
export const GIZMO_HEAD_LENGTH_RATIO = 0.18;
/** Must match shader GIZMO_HIT_EPS. */
export const GIZMO_HIT_EPS = 0.00015;
/** Screen-space fallback tolerance in CSS pixels (3D pick is primary). */
export const GIZMO_PICK_PIXELS_SHAFT = 10;
export const GIZMO_PICK_PIXELS_HEAD = 14;
/** Wider tolerance to keep hover cursor stable between GPU single-pixel hits. */
export const GIZMO_HOVER_STICKY_PIXELS_SHAFT = 14;
export const GIZMO_HOVER_STICKY_PIXELS_HEAD = 18;
/** Visual-only: larger arrows; drawn on top of scene (no camera offset). */
export const GIZMO_RENDER_SCALE_MULT = 1.35;
export const GIZMO_CAMERA_PUSH = 0;

const DEG_TO_RAD = Math.PI / 180;

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

/** Keep hover axis when GPU single-pixel pick flickers along a thin shaft. */
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
): GizmoAxis | null {
  if (gpuAxis) return gpuAxis;
  if (!previousAxis) return null;

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
