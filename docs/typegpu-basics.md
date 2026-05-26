# TypeGPU Basics — Initialization, Root, Uniforms, Buffers

## Initialization

```typescript
import tgpu, { d, std } from "typegpu";

const root = await tgpu.init();
```

`tgpu.init()` returns a `TgpuRoot` — the main entry point for all GPU operations.

## TgpuRoot

The root object provides methods for creating resources and pipelines.

### Key methods

| Method | Description |
|--------|-------------|
| `root.createUniform(schema, initial?)` | GPU buffer, read-only on GPU, optimized for small data |
| `root.createMutable(schema, initial?)` | GPU buffer, read-write on GPU (storage) |
| `root.createReadonly(schema, initial?)` | GPU buffer, read-only on GPU (storage) |
| `root.createBuffer(schema, initial?)` | General-purpose GPU buffer |
| `root.createRenderPipeline(descriptor)` | Creates a render pipeline |
| `root.configureContext(options)` | Configures a canvas for rendering |
| `root.destroy()` | Destroys root and releases GPU resources |

## Uniforms

Uniforms are small read-only (on GPU side) buffers, optimized for frequently updated data like time, mouse position, aspect ratio.

### Creating

```typescript
const timeUniform = root.createUniform(d.f32, 0);
const mouseUniform = root.createUniform(d.vec2f, d.vec2f(0, 0));
const aspectUniform = root.createUniform(d.f32, canvas.width / canvas.height);
```

Signature: `root.createUniform<TData>(typeSchema, initial?)`

- `typeSchema` — a TypeGPU data type (`d.f32`, `d.vec2f`, `d.vec3f`, etc.)
- `initial` — optional initial value

### Writing (CPU → GPU)

```typescript
timeUniform.write(1.5);
mouseUniform.write(d.vec2f(0.5, -0.3));
```

### Reading in shaders (GPU side)

Use `.$` to access the GPU-side value inside `"use gpu"` functions:

```typescript
const fragment = ({ uv }: { uv: d.v2f }) => {
  "use gpu";
  const time = timeUniform.$;       // d.f32 → number on GPU
  const mouse = mouseUniform.$;     // d.v2f on GPU
  const aspect = aspectUniform.$;   // d.f32 → number on GPU
};
```

### Reading back (GPU → CPU)

```typescript
const value = await timeUniform.read(); // returns Promise<number>
```

## Buffers (general-purpose)

For larger data or read-write access:

```typescript
const buffer = root.createBuffer(d.vec3f, d.vec3f(1, 2, 3));
buffer.write(d.vec3f(4, 5, 6));
const data = await buffer.read();
```

## Buffer shorthands summary

| Shorthand | GPU access | Created via |
|-----------|-----------|-------------|
| `TgpuUniform` | read-only | `root.createUniform()` |
| `TgpuMutable` | read-write | `root.createMutable()` |
| `TgpuReadonly` | read-only | `root.createReadonly()` |

All shorthands share these methods:
- `.write(data)` — write from CPU
- `.read()` — read back to CPU (async)
- `.$` — access GPU-side value in shaders (replaces deprecated `.value`)
