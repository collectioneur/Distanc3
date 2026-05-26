# TypeGPU std — Standard Library Functions

All functions are accessed via `std` namespace: `import { std } from "typegpu"`.

These functions work in `"use gpu"` blocks and map directly to WGSL built-in functions.
Most accept both scalars (`number`) and vectors (`d.v2f`, `d.v3f`, `d.v4f`, etc.).

## Math — Trigonometry

| Function | Description | Signature |
|----------|-------------|-----------|
| `std.sin(x)` | Sine | `number → number`, `vec → vec` |
| `std.cos(x)` | Cosine | `number → number`, `vec → vec` |
| `std.tan(x)` | Tangent | `number → number`, `vec → vec` |
| `std.asin(x)` | Arc sine | `number → number`, `vec → vec` |
| `std.acos(x)` | Arc cosine | `number → number`, `vec → vec` |
| `std.atan(x)` | Arc tangent | `number → number`, `vec → vec` |
| `std.atan2(y, x)` | Arc tangent of y/x | `(number, number) → number` |
| `std.sinh(x)` | Hyperbolic sine | `number → number`, `vec → vec` |
| `std.cosh(x)` | Hyperbolic cosine | `number → number`, `vec → vec` |
| `std.tanh(x)` | Hyperbolic tangent | `number → number`, `vec → vec` |
| `std.asinh(x)` | Inverse hyperbolic sine | `number → number`, `vec → vec` |
| `std.acosh(x)` | Inverse hyperbolic cosine | `number → number`, `vec → vec` |
| `std.atanh(x)` | Inverse hyperbolic tangent | `number → number`, `vec → vec` |
| `std.radians(x)` | Degrees to radians | `number → number`, `vec → vec` |
| `std.degrees(x)` | Radians to degrees | `number → number`, `vec → vec` |

## Math — Common

| Function | Description | Signature |
|----------|-------------|-----------|
| `std.abs(x)` | Absolute value | `number → number`, `vec → vec` |
| `std.ceil(x)` | Round up | `number → number`, `vec → vec` |
| `std.floor(x)` | Round down | `number → number`, `vec → vec` |
| `std.round(x)` | Round to nearest | `number → number`, `vec → vec` |
| `std.trunc(x)` | Truncate to integer | `number → number`, `vec → vec` |
| `std.sign(x)` | Sign (-1, 0, 1) | `number → number`, `vec → vec` |
| `std.fract(x)` | Fractional part | `number → number`, `vec → vec` |
| `std.sqrt(x)` | Square root | `number → number`, `vec → vec` |
| `std.inverseSqrt(x)` | 1 / sqrt(x) | `number → number`, `vec → vec` |
| `std.exp(x)` | e^x | `number → number`, `vec → vec` |
| `std.exp2(x)` | 2^x | `number → number`, `vec → vec` |
| `std.log(x)` | Natural logarithm | `number → number`, `vec → vec` |
| `std.log2(x)` | Base-2 logarithm | `number → number`, `vec → vec` |
| `std.pow(base, exp)` | Power | `(number, number) → number`, `(vec, vec) → vec` |
| `std.clamp(x, lo, hi)` | Clamp to range | `(number, number, number) → number` |
| `std.min(a, b)` | Minimum | `(number, number) → number`, `(vec, vec) → vec` |
| `std.max(a, b)` | Maximum | `(number, number) → number`, `(vec, vec) → vec` |
| `std.mix(a, b, t)` | Linear interpolation | `(number, number, number) → number`, `(vec, vec, number) → vec` |
| `std.step(edge, x)` | 0 if x < edge, else 1 | `(number, number) → number` |
| `std.smoothstep(lo, hi, x)` | Smooth Hermite interpolation | `(number, number, number) → number` |
| `std.saturate(x)` | Clamp to [0, 1] | `number → number`, `vec → vec` |
| `std.fma(a, b, c)` | Fused multiply-add (a*b+c) | `(number, number, number) → number` |

## Vector Operations

| Function | Description | Signature |
|----------|-------------|-----------|
| `std.length(v)` | Vector length | `vec → number` |
| `std.distance(a, b)` | Distance between two points | `(vec, vec) → number` |
| `std.normalize(v)` | Unit vector | `vec → vec` |
| `std.dot(a, b)` | Dot product | `(vec, vec) → number` |
| `std.cross(a, b)` | Cross product (3D only) | `(v3f, v3f) → v3f` |
| `std.reflect(I, N)` | Reflect I around normal N | `(vec, vec) → vec` |
| `std.refract(I, N, eta)` | Refract through surface | `(vec, vec, number) → vec` |
| `std.faceForward(N, I, Nref)` | Flip N if dot(I, Nref) < 0 | `(vec, vec, vec) → vec` |

## Matrix Operations

| Function | Description |
|----------|-------------|
| `std.determinant(m)` | Matrix determinant |
| `std.identity2` | 2×2 identity matrix |
| `std.identity3` | 3×3 identity matrix |
| `std.identity4` | 4×4 identity matrix |
| `std.rotationX4(angle)` | 4×4 rotation around X |
| `std.rotationY4(angle)` | 4×4 rotation around Y |
| `std.rotationZ4(angle)` | 4×4 rotation around Z |
| `std.rotateX4(m, angle)` | Rotate existing 4×4 matrix around X |
| `std.rotateY4(m, angle)` | Rotate existing 4×4 matrix around Y |
| `std.rotateZ4(m, angle)` | Rotate existing 4×4 matrix around Z |
| `std.scale4(m, scale)` | Scale 4×4 matrix |

## Fragment Shader Specific

| Function | Description |
|----------|-------------|
| `std.discard()` | Discard fragment |
| `std.dpdx(v)` | Partial derivative in x |
| `std.dpdy(v)` | Partial derivative in y |
| `std.fwidth(v)` | abs(dpdx) + abs(dpdy) |

## Bitwise Operations

| Function | Description |
|----------|-------------|
| `std.countLeadingZeros(x)` | Count leading zeros |
| `std.countOneBits(x)` | Count set bits |
| `std.countTrailingZeros(x)` | Count trailing zeros |
| `std.extractBits(e, offset, count)` | Extract bit field |
| `std.insertBits(e, bits, offset, count)` | Insert bit field |
| `std.firstLeadingBit(x)` | Index of first leading bit |
| `std.firstTrailingBit(x)` | Index of first trailing bit |
| `std.reverseBits(x)` | Reverse bit order |

## Operator Overloads

In `"use gpu"` blocks, standard arithmetic operators work on vectors and matrices (requires `unplugin-typegpu` + `tsover`):

```typescript
const a = d.vec3f(1, 2, 3);
const b = d.vec3f(4, 5, 6);

a + b      // vec3f(5, 7, 9)
a - b      // vec3f(-3, -3, -3)
a * 2.0    // vec3f(2, 4, 6)
a * b      // vec3f(4, 10, 18)  — component-wise
mat * vec  // matrix-vector multiplication
```

Note: TypeScript will show errors on these operators without `tsover`. They compile correctly via `unplugin-typegpu`.
