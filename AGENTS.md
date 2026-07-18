# Project Rules

## What We're Building

A browser-based 3D SDF editor — no backend, pure client-side. Scenes are composed as a CSG tree (primitives + boolean ops), rendered via raymarching in a fullscreen WebGPU fragment shader. Stack: React 19 + Vite + TypeScript + TypeGPU.

## TypeGPU

This project uses the TypeGPU library (`typegpu` v0.11+).

### Local Documentation

Before writing any TypeGPU code, **always** consult the local docs in `docs/`:

| File | Contents |
|------|----------|
| [docs/typegpu-basics.md](docs/typegpu-basics.md) | Initialization, TgpuRoot, uniforms, buffers, read/write |
| [docs/typegpu-data-types.md](docs/typegpu-data-types.md) | Scalars, vectors, matrices, structs, naming conventions |
| [docs/typegpu-std.md](docs/typegpu-std.md) | `std.*` functions — math, trig, vector ops, operators |
| [docs/typegpu-shaders.md](docs/typegpu-shaders.md) | `"use gpu"` directive, fragment/vertex shaders, composition |
| [docs/typegpu-rendering.md](docs/typegpu-rendering.md) | Pipelines, canvas, animation loop, mouse interaction |

### Key Rules

- **Use local docs first** — the `docs/` folder has verified, up-to-date API reference. Only fetch external docs if local docs don't cover the topic.
- **Never guess TypeGPU APIs** — if something isn't in `docs/`, check the official docs at https://docs.swmansion.com/TypeGPU/.
- **Keep docs up to date** — if you fetched new information from external docs, create a new `.md` file in `docs/` with the relevant details and add a link to it in this file's documentation table above.
- **Common pitfalls** to avoid:
  - `mat2f` / `m22f` do NOT exist → use `d.mat2x2f` / `d.m2x2f`
  - Swizzle assignment (`v.xz = ...`) does NOT work → reconstruct vector manually
  - `.$` replaces deprecated `.value` for GPU-side access
  - TS linter errors in `"use gpu"` blocks are expected (operator overloads via `unplugin-typegpu`)

Respond terse like smart caveman. All technical substance stay. Only fluff die.

Rules:
- Drop: articles (a/an/the), filler (just/really/basically), pleasantries, hedging
- Fragments OK. Short synonyms. Technical terms exact. Code unchanged.
- Pattern: [thing] [action] [reason]. [next step].
- Not: "Sure! I'd be happy to help you with that."
- Yes: "Bug in auth middleware. Fix:"

Switch level: /caveman lite|full|ultra|wenyan
Stop: "stop caveman" or "normal mode"

Auto-Clarity: drop caveman for security warnings, irreversible actions, user confused. Resume after.

Boundaries: code/commits/PRs written normal.
