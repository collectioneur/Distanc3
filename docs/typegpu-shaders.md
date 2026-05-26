# TypeGPU Shaders — Writing GPU Functions

## `"use gpu"` Directive

All GPU shader code is written as regular TypeScript functions with the `"use gpu"` directive at the top:

```typescript
const myFunction = (p: d.v3f, r: number): number => {
  "use gpu";
  return std.length(p) - r;
};
```

### Rules

1. **Must start with `"use gpu"`** — first statement in the function body
2. **Types use `d.*` namespace** — parameters and return types use TypeGPU types (`d.v3f`, `d.v2f`, `number`, etc.)
3. **`number` maps to `f32`** — in GPU context, TypeScript `number` becomes WGSL `f32`
4. **Use `d.f32()` for explicit casting** — when you need to ensure a value is f32: `let t = d.f32(0.0);`
5. **Operator overloads require `unplugin-typegpu`** — vec/mat arithmetic (`+`, `-`, `*`) is transformed at build time
6. **No swizzle assignment** — `v.xz = ...` does NOT work, reconstruct the vector manually
7. **Access uniforms via `.$`** — e.g., `timeUniform.$` reads the GPU-side value

### TypeScript linter errors

Operator overloads in `"use gpu"` blocks will show TS errors like:
- "The left-hand side of an arithmetic operation must be of type 'any', 'number'..."
- "Property 'x' does not exist on type 'number'"
- "Operator '+' cannot be applied to types 'v3f' and 'number'"

These are expected and safe to ignore — `unplugin-typegpu` handles the transformation at build time. Setup `tsover` to eliminate these.

## Fragment Shader

The fragment function receives `{ uv: d.v2f }` and returns `d.v4f` (RGBA color):

```typescript
const fragment = ({ uv }: { uv: d.v2f }) => {
  "use gpu";

  // uv is in [0, 1] range, convert to [-1, 1]
  const uvn = uv * 2.0 - d.vec2f(1.0, 1.0);

  // Aspect ratio correction
  const uvCorrected = d.vec2f(uvn.x * aspectUniform.$, uvn.y);

  // ... shader logic ...

  return d.vec4f(r, g, b, a);
};
```

### Common pattern: full-screen shader

```typescript
import { fullScreenTriangle } from "typegpu/common";

const pipeline = root.createRenderPipeline({
  vertex: fullScreenTriangle,
  fragment,
});
```

`fullScreenTriangle` is a built-in vertex shader that covers the entire screen with a single triangle.

## Vertex Shader

Custom vertex shaders receive vertex attributes and return position + interpolated data:

```typescript
const vertex = ({ position, color }: { position: d.v3f; color: d.v4f }) => {
  "use gpu";
  return {
    position: d.vec4f(position, 1.0),
    color,
  };
};
```

## Composing GPU Functions

GPU functions can call other GPU functions:

```typescript
const sdSphere = (p: d.v3f, R: number): number => {
  "use gpu";
  return std.length(p) - R;
};

const sdScene = (p: d.v3f): number => {
  "use gpu";
  return sdSphere(p, 1.0); // calls sdSphere on GPU
};
```

Functions can reference uniforms from outer scope:

```typescript
const timeUniform = root.createUniform(d.f32, 0);

const animate = (p: d.v3f): d.v3f => {
  "use gpu";
  const t = timeUniform.$; // reads uniform inside GPU function
  return p + d.vec3f(std.sin(t), 0, 0);
};
```

## 2D Rotation Helper

Since there's no built-in `rot2D`, use a manual 2×2 rotation matrix:

```typescript
const rot2D = (angle: number): d.m2x2f => {
  "use gpu";
  const c = std.cos(angle);
  const s = std.sin(angle);
  return d.mat2x2f(c, -s, s, c);
};
```

Apply to a plane (e.g., XZ rotation):

```typescript
const rotM = rot2D(angle);
const rotated = rotM * d.vec2f(v.x, v.z);
v = d.vec3f(rotated.x, v.y, rotated.y);
```
