# TypeGPU Data Types — Vectors, Matrices, Scalars

All data types are accessed via the `d` namespace: `import { d } from "typegpu"`.

## Scalars

| Type | Schema | GPU type | Description |
|------|--------|----------|-------------|
| `d.f32` | `f32` | `f32` | 32-bit float |
| `d.f16` | `f16` | `f16` | 16-bit float (requires f16 feature) |
| `d.i32` | `i32` | `i32` | 32-bit signed integer |
| `d.u32` | `u32` | `u32` | 32-bit unsigned integer |
| `d.bool` | `bool` | `bool` | Boolean |

In `"use gpu"` functions, use `d.f32(value)` to explicitly cast a number:

```typescript
let t = d.f32(0.0);
```

## Vectors

### Constructors

```typescript
d.vec2f()          // (0.0, 0.0)
d.vec2f(1)         // (1.0, 1.0)          — splat
d.vec2f(0.5, 0.1)  // (0.5, 0.1)

d.vec3f()              // (0.0, 0.0, 0.0)
d.vec3f(1)             // (1.0, 1.0, 1.0)
d.vec3f(1, 2, 3.5)     // (1.0, 2.0, 3.5)

d.vec4f()                  // (0.0, 0.0, 0.0, 0.0)
d.vec4f(1)                 // (1.0, 1.0, 1.0, 1.0)
d.vec4f(1, 2, 3, 4)       // (1.0, 2.0, 3.0, 4.0)
d.vec4f(d.vec3f(1,2,3), 1) // (1.0, 2.0, 3.0, 1.0) — from vec3 + scalar
```

### All vector types

| Constructor | GPU type | TS type (GPU) | Components |
|-------------|----------|---------------|------------|
| `d.vec2f` | `vec2<f32>` | `d.v2f` | 2 × f32 |
| `d.vec2h` | `vec2<f16>` | `d.v2h` | 2 × f16 |
| `d.vec2i` | `vec2<i32>` | `d.v2i` | 2 × i32 |
| `d.vec2u` | `vec2<u32>` | `d.v2u` | 2 × u32 |
| `d.vec2b` | `vec2<bool>` | `d.v2b` | 2 × bool |
| `d.vec3f` | `vec3<f32>` | `d.v3f` | 3 × f32 |
| `d.vec3h` | `vec3<f16>` | `d.v3h` | 3 × f16 |
| `d.vec3i` | `vec3<i32>` | `d.v3i` | 3 × i32 |
| `d.vec3u` | `vec3<u32>` | `d.v3u` | 3 × u32 |
| `d.vec3b` | `vec3<bool>` | `d.v3b` | 3 × bool |
| `d.vec4f` | `vec4<f32>` | `d.v4f` | 4 × f32 |
| `d.vec4h` | `vec4<f16>` | `d.v4h` | 4 × f16 |
| `d.vec4i` | `vec4<i32>` | `d.v4i` | 4 × i32 |
| `d.vec4u` | `vec4<u32>` | `d.v4u` | 4 × u32 |
| `d.vec4b` | `vec4<bool>` | `d.v4b` | 4 × bool |

### Accessing components

```typescript
const v = d.vec3f(1, 2, 3);
v.x  // 1.0
v.y  // 2.0
v.z  // 3.0
```

### Swizzling

Swizzle reads are supported (e.g., `v.xz`, `v.xy`), but **swizzle assignment is NOT supported**:

```typescript
// ❌ DOES NOT WORK
ro.xz *= someMatrix;

// ✅ Correct approach: manually decompose and reconstruct
const rotated = someMatrix * d.vec2f(ro.x, ro.z);
ro = d.vec3f(rotated.x, ro.y, rotated.y);
```

## Matrices

| Constructor | GPU type | TS type (GPU) | Size |
|-------------|----------|---------------|------|
| `d.mat2x2f` | `mat2x2<f32>` | `d.m2x2f` | 2×2 |
| `d.mat3x3f` | `mat3x3<f32>` | `d.m3x3f` | 3×3 |
| `d.mat4x4f` | `mat4x4<f32>` | `d.m4x4f` | 4×4 |

### Important naming convention

**NOT** `mat2f` / `m22f` — these don't exist!
Always use the full form: `mat2x2f`, `mat3x3f`, `mat4x4f` and corresponding types `m2x2f`, `m3x3f`, `m4x4f`.

### Creating matrices

```typescript
// 2D rotation matrix
const rot = d.mat2x2f(
  cos, -sin,
  sin,  cos
);
```

### Matrix × vector multiplication

```typescript
const result = rot * d.vec2f(x, z); // returns d.v2f
```

## Structs

```typescript
const MyStruct = d.struct({
  position: d.vec3f,
  color: d.vec4f,
  intensity: d.f32,
});
```
