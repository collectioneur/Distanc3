export type Vec3 = [number, number, number];

export type CameraRay = {
  origin: Vec3;
  direction: Vec3;
};

function rot2D(angle: number, x: number, z: number): [number, number] {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [c * x - s * z, s * x + c * z];
}

/** Match fragment shader camera: orbit around origin, perspective along +Z. */
export function computeCameraRay(
  u: number,
  v: number,
  aspect: number,
  rotX: number,
  rotY: number,
  distance: number,
): CameraRay {
  const uvnX = u * 2 - 1;
  const uvnY = v * 2 - 1;
  const uvCorrectedX = uvnX * aspect;
  const uvCorrectedY = uvnY;

  let ro: Vec3 = [0, 0, -distance];
  let rd: Vec3 = normalize([uvCorrectedX, uvCorrectedY, 1]);

  const [roX, roZ] = rot2D(rotX, ro[0], ro[2]);
  ro = [roX, ro[1], roZ];
  const [rdX, rdZ] = rot2D(rotX, rd[0], rd[2]);
  rd = [rdX, rd[1], rdZ];

  const [roY, roZ2] = rot2D(rotY, ro[1], ro[2]);
  ro = [ro[0], roY, roZ2];
  const [rdY, rdZ2] = rot2D(rotY, rd[1], rd[2]);
  rd = [rd[0], rdY, rdZ2];

  return { origin: ro, direction: rd };
}

/** Aspect of the displayed canvas — must match mouse UV mapping. */
export function getCanvasAspect(canvas: HTMLCanvasElement): number {
  const rect = canvas.getBoundingClientRect();
  return rect.width > 0 ? rect.width / rect.height : 1;
}

export function clientToUv(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
): { u: number; v: number } | null {
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return null;
  // Match fullScreenTriangle: uv.y=0 at top, uv.y=1 at bottom.
  return {
    u: (clientX - rect.left) / rect.width,
    v: (clientY - rect.top) / rect.height,
  };
}

/** Framebuffer pixel coords for GPU pick readback (row 0 = top). */
export function clientToPickPixel(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
): { x: number; y: number } | null {
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0 || canvas.width <= 0 || canvas.height <= 0) {
    return null;
  }
  const u = (clientX - rect.left) / rect.width;
  const vScreen = (clientY - rect.top) / rect.height;
  return {
    x: Math.min(
      canvas.width - 1,
      Math.max(0, Math.floor(u * canvas.width)),
    ),
    y: Math.min(
      canvas.height - 1,
      Math.max(0, Math.floor(vScreen * canvas.height)),
    ),
  };
}

export function computeCameraRayFromClient(
  clientX: number,
  clientY: number,
  canvas: HTMLCanvasElement,
  rotX: number,
  rotY: number,
  distance: number,
): CameraRay | null {
  const uv = clientToUv(clientX, clientY, canvas);
  if (!uv) return null;
  return computeCameraRay(
    uv.u,
    uv.v,
    getCanvasAspect(canvas),
    rotX,
    rotY,
    distance,
  );
}

function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]);
  if (len < 1e-12) return [0, 0, 1];
  return [v[0] / len, v[1] / len, v[2] / len];
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

/** World point in view space (camera looks along +Z, origin at orbit eye). */
export function worldToView(
  world: Vec3,
  rotX: number,
  rotY: number,
  distance: number,
): Vec3 | null {
  let ro: Vec3 = [0, 0, -distance];
  const [roX, roZ] = rot2D(rotX, ro[0], ro[2]);
  ro = [roX, ro[1], roZ];
  const [roY, roZ2] = rot2D(rotY, ro[1], ro[2]);
  ro = [ro[0], roY, roZ2];

  let rel = sub(world, ro);
  const [iry, irz] = rot2D(-rotY, rel[1], rel[2]);
  rel = [rel[0], iry, irz];
  const [irx, irz2] = rot2D(-rotX, rel[0], rel[2]);
  rel = [irx, rel[1], irz2];
  if (rel[2] <= 0.05) return null;
  return rel;
}

/** NDC-style uv in [0,1], matching fragment shader + clientToUv. */
export function worldToUv(
  world: Vec3,
  rotX: number,
  rotY: number,
  distance: number,
  aspect: number,
): { u: number; v: number } | null {
  const rel = worldToView(world, rotX, rotY, distance);
  if (!rel) return null;
  const uvnX = rel[0] / rel[2];
  const uvnY = rel[1] / rel[2];
  const u = uvnX / aspect * 0.5 + 0.5;
  const v = uvnY * 0.5 + 0.5;
  if (u < -0.15 || u > 1.15 || v < -0.15 || v > 1.15) return null;
  return { u, v };
}

/** Project world point to CSS client coordinates on the canvas. */
export function worldToClient(
  world: Vec3,
  canvas: HTMLCanvasElement,
  rotX: number,
  rotY: number,
  distance: number,
): { x: number; y: number } | null {
  const uv = worldToUv(
    world,
    rotX,
    rotY,
    distance,
    getCanvasAspect(canvas),
  );
  if (!uv) return null;
  const rect = canvas.getBoundingClientRect();
  const x = rect.left + uv.u * rect.width;
  const y = rect.top + uv.v * rect.height;
  const margin = 80;
  if (
    x < rect.left - margin ||
    x > rect.right + margin ||
    y < rect.top - margin ||
    y > rect.bottom + margin
  ) {
    return null;
  }
  return { x, y };
}

/** World-space camera position (matches shader orbit). */
export function getCameraWorldPosition(
  rotX: number,
  rotY: number,
  distance: number,
): Vec3 {
  let ro: Vec3 = [0, 0, -distance];
  const [roX, roZ] = rot2D(rotX, ro[0], ro[2]);
  ro = [roX, ro[1], roZ];
  const [roY, roZ2] = rot2D(rotY, ro[1], ro[2]);
  return [ro[0], roY, roZ2];
}
