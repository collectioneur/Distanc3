# d-stance

Browser-only 3D SDF editor. No backend. Scene is a CSG tree (shapes + groups + boolean ops). CPU bakes each shape into one inverse affine map, packs a flat instruction buffer, GPU raymarches it in a fullscreen TypeGPU fragment shader.

Stack: React 19, Vite, TypeScript (`tsover` 5.9 — workspace TS, not the editor builtin), TypeGPU 0.12, Zustand + zundo.

Agents break this repo by desyncing **cross-file contracts**, not by missing a TypeGPU API. `tsc` does not catch most of those.

## Before you edit

| Touching | Read first |
|---|---|
| TypeGPU / `"use gpu"` | `.agents/skills/typegpu`. Never guess APIs. After GPU edits: `npm run inspect:gpu`. |
| New or changed primitive | `src/store/sceneStore.ts` + `src/gpu/shader.ts` (`SHAPE_TYPE_INT`, `sd*`, dispatch switch) + `src/utils/generateSdfCode.ts` + `src/utils/commands.ts` |
| Transforms, gizmo, scale | `src/gpu/bake.ts` (composition order), `src/gpu/gizmo.ts` |
| Tree / DnD / add / remove | `src/scene/arboristAdapter.ts`, `countInstructions`, caps in `sceneStore.ts` |
| Instruction layout / opcodes / SDF eval | `src/gpu/shader.ts` **and** packer in `src/gpu/useGpu.ts` |
| Save / load / URL hash | `src/store/persistence.ts` (`SCHEMA_VERSION`, `normalizePersisted`) |
| Quality presets | `src/utils/quality.ts` (`assertQualityInvariants`) |
| UI | Changed flow in the browser. Screenshot ≠ verification. |

Do not open `src/gpu/shader.ts` (~1700 lines) unless the task needs it. Do not “clean up” it while doing something else.

## Data flow

```
sceneStore (CSG tree)
  → useGpu.ts packs instructions (bake.ts per shape)
  → shader.ts raymarches Instruction[]
  → persistence.ts writes hash + localStorage
  → generateSdfCode.ts exports the same tree as TypeGPU / WGSL / GLSL
```

Selection lives on the store (`selectedItemId`, `selectedContainerId`, `rootSelected`). Undo/redo (`undoScene` / `redoScene`) clears selection back to the scene root.

## Hard contracts

These are the things that silently black-screen, clip the CSG, or lose the user’s scene.

### 1. Shape registry

A shape is not registered until **all** of these know it, with the same key:

- `ShapeType` + `DEFAULT_PARAMS` + `TYPE_LABEL` in `sceneStore.ts`
- `SHAPE_TYPE_INT` in `shader.ts` — unique dense ints `0..n-1`
- `sd*` primitive + dispatch `switch` in `shader.ts`
- TypeGPU / WGSL / GLSL emitters in `generateSdfCode.ts`
- `SHAPE_ICONS` in `commands.ts` (`Record<ShapeType, …>` will fail typecheck if an icon is missing; toolbar `SHAPES` vs palette `EXTRA_SHAPES` is a separate product choice)

Adding a shape without codegen or without a GPU case is a shippable bug. TypeScript will not notice a missing `switch` arm if there is a default.

### 2. Instruction stream

Packed by `useGpu.ts`, consumed by `shader.ts`. Both sides must stay in lockstep.

- Opcodes: `OPCODE_PUSH_SHAPE = 0`, `OPCODE_OP = 1`
- Ops: `OP_TYPE_INT` (`union`…`sIntersect` = 0…5)
- `Instruction` is **112 bytes**. Fields and `_pad*` exist for WGSL alignment. Do not reorder, drop pads, or pack extra fields without updating the CPU writer and every empty-instruction filler.
- GPU eval: `q = (row0·p, row1·p, row2·p) + offset`, `dist = sd(q, params) * factor`
- `countInstructions(tree)` must match the length of the packed stream (excluding the pad-to-`MAX_INSTRUCTIONS` tail). If they drift, the cap lies and the GPU reads garbage or clips the scene.

Group/layer transforms are **not** a GPU stack. They are baked on the CPU (`bake.ts`) into `row0/1/2`, `offset`, `factor`.

### 3. Bake math

`bake.ts` must reproduce the old GPU transform exactly. Image changes if the order changes.

```
pEval = T_{n-1}(…T_0(p))     T_k(x) = D(1/s_k) · R_k⁻¹ · (x − p_k)
lp    = D(1/acc) · R_shape⁻¹ · D(acc) · (pEval − shapePos)   // unsheared trick
q     = D(1/shapeScale) · lp
dist  = sd(q) · min(shapeScale) · min(acc)
```

`acc` = componentwise product of ancestor group scales. Rotation is Euler XYZ, inverted as `Rx(−x)·Ry(−y)·Rz(−z)` (`invRotXYZ`).

`assertBakeParity()` runs on import of `bake.ts`. Do not delete it. Do not “simplify” the matrices.

Scale below `MIN_SCALE` (0.01) is illegal — divide-by-zero in the bake.

### 4. Persistence

`SCHEMA_VERSION = 4`. Stored in `location.hash` (base64 JSON) and `localStorage` key `d-stance_scene`.

- Hash/LS is a **trust boundary**. Users and old links can send missing/hostile fields.
- New persisted field → bump `SCHEMA_VERSION` **and** teach `normalizePersisted` a default. Old blobs with the wrong `v` are dropped (`null`), not partially applied.
- Additive-only example already in tree: missing `scale` → `[1,1,1]`, then clamped with `Math.max(0.01, abs(n))`.
- Counters for shapes added after a save are filled via `{ ...zeroCounters(), ...saved.counters }`. Same idea if you add a counter.

Losing a scene is data loss. Treat persist mistakes like a security bug.

### 5. Caps

| Cap | Value | Meaning |
|---|---|---|
| `MAX_INSTRUCTIONS` | 256 | Scene CSG instruction budget |
| `MAX_TRANSFORM_DEPTH` | 16 | Nested group depth |
| `MAX_GPU_OBJECTS` | 8 | `objectInfo` slots; scene render uses root slot `[0]` |
| `MAX_PICK_OBJECTS` | 255 | Pick IDs fit in one byte (`1–255`) |
| `MAX_PICK_INSTRUCTIONS` | 1024 | Separate pick CSG buffer |

UI toasts exist for cap / cycle / depth. Do not raise a cap in one file and leave the others.

## TypeGPU

`typegpu@0.12` + `unplugin-typegpu@0.12` (Vite). Official skill lives at `.agents/skills/typegpu` and **targets 0.12**. Use the named export: `import { tgpu, d, std } from "typegpu"`.

Operator overloads inside `"use gpu"` are understood by **tsover** (this repo’s `typescript` package). If the editor shows `+` / `*` errors on `d.v3f`, it is using bundled TypeScript — switch to the workspace version. Do not “fix” those errors by rewriting operators into `std.add` unless asked.

Lookup order for TypeGPU / `"use gpu"` code:

1. `.agents/skills/typegpu/SKILL.md` — then the reference file it names for the task (`references/types.md`, `shaders.md`, `std.md`, …).
2. https://docs.swmansion.com/TypeGPU/ — only if the skill does not cover the question.
3. Never guess.

Do not copy API essays from the skill into this file.

Pitfalls:

- No `mat2f` / `m22f` → `d.mat2x2f` / `d.m2x2f`
- No swizzle assignment (`v.xz = …`) → rebuild the vector
- GPU-side uniform/buffer read is `.$` (not `.value`)
- `"use gpu"` must be the first statement in the function body
- `Math.*` is not the GPU stdlib — use `std.*`

## Do not

- Guess TypeGPU APIs or WGSL alignment.
- Reorder / pad-strip `Instruction` or change opcodes without updating `useGpu.ts` in the same change.
- Raise or bypass caps without updating packer, shader array sizes, UI toasts, and this file.
- Bump or skip `SCHEMA_VERSION` “just in case”.
- Refactor `shader.ts` as a drive-by.
- Delete `assertBakeParity` / `assertQualityInvariants`.
- Add a dependency or abstraction that was not asked for.

## After a change

1. `npx tsc -b` (same check as `npm run build`).
2. If you touched bake / quality / persist / instruction packing / shape registry: keep or add a tiny import-time assert in that module (see `assertBakeParity`, `assertQualityInvariants`). There is no `npm test` yet.
3. GPU files (`shader.ts`, `bake.ts`, `useGpu.ts`): `npm run inspect:gpu` (Chromium/WebGPU compiles the raymarch pipeline). `tsc` does not compile WGSL. MCP tool: `typegpu_inspector` → `inspect_typegpu` on `src/gpu/shader.inspect.ts`. Then confirm the canvas is not black.
4. UI: exercise the changed flow in the browser (click/type/drag). Check the other surfaces that read the same store (tree, inspector, gizmo, code export, hash after reload).
5. New persist field: load an old `v: 4` blob and a blob missing the field.

Code, commits, and PR text stay normal prose. Do not commit unless asked.
