import tgpu, { d, std } from "typegpu";
import { fullScreenTriangle } from "typegpu/common";

type TgpuRoot = Awaited<ReturnType<typeof tgpu.init>>;

export const MAX_SHAPES = 8;

export const SHAPE_TYPE_INT = {
  sphere: 0,
  box: 1,
  torus: 2,
  cylinder: 3,
  capsule: 4,
  cone: 5,
} as const;

const ShapeData = d.struct({
  shapeType: d.u32,
  position: d.vec3f,
  params: d.vec4f,
});

const emptyShape = {
  shapeType: 0,
  position: d.vec3f(0, 0, 0),
  params: d.vec4f(0, 0, 0, 0),
};

export function createShader(root: TgpuRoot) {
  const timeUniform = root.createUniform(d.f32, 0);
  const aspectUniform = root.createUniform(d.f32, 1);
  const mouseUniform = root.createUniform(d.vec2f, d.vec2f(0.3, -0.4));
  const shapeCountUniform = root.createUniform(d.u32, 0);
  const renderModeUniform = root.createUniform(d.u32, 0);
  const shapesBuffer = root.createReadonly(
    d.arrayOf(ShapeData, MAX_SHAPES),
    Array.from({ length: MAX_SHAPES }, () => ({ ...emptyShape })),
  );

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
    return std.min(std.max(d2.x, d2.y), 0.0) + std.length(std.max(d2, d.vec2f(0.0)));
  };

  const sdCapsule = (p: d.v3f, r: number, h: number): number => {
    "use gpu";
    const py = std.clamp(p.y, -h, h);
    return std.length(p - d.vec3f(0.0, py, 0.0)) - r;
  };

  const sdCone = (p: d.v3f, r: number, h: number): number => {
    "use gpu";
    // Capped cone: base at y=-h (radius r), apex at y=+h (radius 0), IQ-style
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

  const sdScene = (p: d.v3f): number => {
    "use gpu";
    let dist = d.f32(1e10);
    const count = shapeCountUniform.$;
    for (let i = d.u32(0); i < count; i += d.u32(1)) {
      const shape = shapesBuffer.$[i];
      const lp = p - shape.position;
      if (shape.shapeType === d.u32(0)) {
        dist = std.min(dist, sdSphere(lp, shape.params.x));
      } else if (shape.shapeType === d.u32(1)) {
        dist = std.min(
          dist,
          sdBox(
            lp,
            d.vec3f(shape.params.x, shape.params.y, shape.params.z),
          ),
        );
      } else if (shape.shapeType === d.u32(2)) {
        dist = std.min(
          dist,
          sdTorus(lp, d.vec2f(shape.params.x, shape.params.y)),
        );
      } else if (shape.shapeType === d.u32(3)) {
        dist = std.min(dist, sdCylinder(lp, shape.params.x, shape.params.y));
      } else if (shape.shapeType === d.u32(4)) {
        dist = std.min(dist, sdCapsule(lp, shape.params.x, shape.params.y));
      } else if (shape.shapeType === d.u32(5)) {
        dist = std.min(dist, sdCone(lp, shape.params.x, shape.params.y));
      }
    }
    return dist;
  };

  const calcNormal = (p: d.v3f): d.v3f => {
    "use gpu";
    const eps = 0.001;
    return std.normalize(
      d.vec3f(
        sdScene(p + d.vec3f(eps, 0.0, 0.0)) -
          sdScene(p - d.vec3f(eps, 0.0, 0.0)),
        sdScene(p + d.vec3f(0.0, eps, 0.0)) -
          sdScene(p - d.vec3f(0.0, eps, 0.0)),
        sdScene(p + d.vec3f(0.0, 0.0, eps)) -
          sdScene(p - d.vec3f(0.0, 0.0, eps)),
      ),
    );
  };

  const rayMarch = (ro: d.v3f, rd: d.v3f): d.v2f => {
    "use gpu";
    let t = d.f32(0.0);
    let iterations = d.f32(0.0);
    for (let i = d.f32(0.0); i < d.f32(64.0); i += d.f32(1.0)) {
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
    let ro = d.vec3f(0, 0, -2.5);
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

    const result = rayMarch(ro, rd);
    const t = result.x;
    const iterations = result.y;
    const mode = renderModeUniform.$;

    if (mode === d.u32(1)) {
      // Depth grayscale: white = close, black = far, black = miss
      if (t > 50.0) {
        return d.vec4f(0.0, 0.0, 0.0, 1.0);
      }
      const depth = std.clamp(t / 5.0, 0.0, 1.0);
      const brightness = 1.0 - depth;
      return d.vec4f(brightness, brightness, brightness, 1.0);
    }

    if (mode === d.u32(2)) {
      // Step count grayscale: black = 0 steps, white = 64 steps, black = miss
      if (t > 50.0) {
        return d.vec4f(0.0, 0.0, 0.0, 1.0);
      }
      const brightness = std.clamp(iterations / 64.0, 0.0, 1.0);
      return d.vec4f(brightness, brightness, brightness, 1.0);
    }

    // Mode 0: lit shading (default)
    if (t > 50.0) {
      return d.vec4f(0.05, 0.05, 0.08, 1.0);
    }

    const pos = ro + rd * t;
    const N = calcNormal(pos);
    const lightDir = std.normalize(d.vec3f(2.0, 3.0, -1.0));
    const diff = std.max(std.dot(N, lightDir), 0.0);
    const viewDir = d.vec3f(-rd.x, -rd.y, -rd.z);
    const fresnel =
      std.pow(1.0 - std.abs(std.dot(N, viewDir)), 3.0) * 0.5;
    const halfDir = std.normalize(lightDir + viewDir);
    const spec = std.pow(std.max(std.dot(N, halfDir), 0.0), 64.0) * 1.5;
    const ao = 1.0 - std.clamp(iterations / 64.0, 0.0, 1.0) * 0.4;
    const baseColor = d.vec3f(0.55, 0.65, 0.95);
    const col =
      (baseColor * (diff * 0.8 + 0.2) +
        d.vec3f(1.0, 1.0, 1.0) * spec +
        baseColor * fresnel) *
      ao;
    return d.vec4f(std.sqrt(std.max(col, d.vec3f(0.0))), 1.0);
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
    shapesBuffer,
    shapeCountUniform,
    renderModeUniform,
  };
}
