# TypeGPU Rendering — Pipelines, Canvas, Animation Loop

## Render Pipeline

```typescript
import { fullScreenTriangle } from "typegpu/common";

const pipeline = root.createRenderPipeline({
  vertex: fullScreenTriangle,  // or custom vertex shader
  fragment,                     // fragment shader function
});
```

### Drawing

```typescript
pipeline
  .withColorAttachment({ view: context })
  .draw(3);  // 3 vertices for fullScreenTriangle
```

- `.withColorAttachment({ view })` — sets the render target
- `.draw(vertexCount)` — issues the draw call

## Canvas Setup

```typescript
const canvas = document.querySelector("canvas") as HTMLCanvasElement;
canvas.width = window.innerWidth * window.devicePixelRatio;
canvas.height = window.innerHeight * window.devicePixelRatio;

const context = root.configureContext({
  canvas,
  alphaMode: "premultiplied",  // or "opaque"
});
```

`configureContext` automatically picks `navigator.gpu.getPreferredCanvasFormat()`.

## Animation Loop

Standard `requestAnimationFrame` pattern for continuous rendering:

```typescript
const timeUniform = root.createUniform(d.f32, 0);

let animationFrameId: number;
const startTime = performance.now();

function frame() {
  const elapsed = (performance.now() - startTime) / 1000;
  timeUniform.write(elapsed);
  pipeline.withColorAttachment({ view: context }).draw(3);
  animationFrameId = requestAnimationFrame(frame);
}

animationFrameId = requestAnimationFrame(frame);
```

### Cleanup

Always cancel the animation frame and destroy the root when done:

```typescript
export function onCleanup() {
  cancelAnimationFrame(animationFrameId);
  root.destroy();
}
```

## Mouse Interaction (Drag to Rotate)

Pattern for orbit-style camera control via mouse drag:

```typescript
const mouseUniform = root.createUniform(d.vec2f, d.vec2f(0, 0));

let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;
let rotX = 0;
let rotY = 0;

canvas.addEventListener("mousedown", (e) => {
  isDragging = true;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
});

canvas.addEventListener("mousemove", (e) => {
  if (!isDragging) return;
  const dx = ((e.clientX - dragStartX) / window.innerWidth) * Math.PI * 2;
  const dy = ((e.clientY - dragStartY) / window.innerHeight) * Math.PI * 2;
  mouseUniform.write(d.vec2f(rotX + dx, rotY + dy));
});

canvas.addEventListener("mouseup", (e) => {
  if (!isDragging) return;
  isDragging = false;
  rotX += ((e.clientX - dragStartX) / window.innerWidth) * Math.PI * 2;
  rotY += ((e.clientY - dragStartY) / window.innerHeight) * Math.PI * 2;
});
```

In the shader, apply rotation to camera ray origin and direction:

```typescript
// Horizontal rotation (around Y axis, XZ plane)
const rotH = rot2D(mouseUniform.$.x);
const roXZ = rotH * d.vec2f(ro.x, ro.z);
ro = d.vec3f(roXZ.x, ro.y, roXZ.y);
const rdXZ = rotH * d.vec2f(rd.x, rd.z);
rd = d.vec3f(rdXZ.x, rd.y, rdXZ.y);

// Vertical rotation (around X axis, YZ plane)
const rotV = rot2D(mouseUniform.$.y);
const roYZ = rotV * d.vec2f(ro.y, ro.z);
ro = d.vec3f(ro.x, roYZ.x, roYZ.y);
const rdYZ = rotV * d.vec2f(rd.y, rd.z);
rd = d.vec3f(rd.x, rdYZ.x, rdYZ.y);
```

## Project Setup

### Dependencies

```json
{
  "dependencies": {
    "typegpu": "^0.11.4"
  },
  "devDependencies": {
    "unplugin-typegpu": "^0.11.3",
    "vite": "^8.0.13"
  }
}
```

### Vite Config

```typescript
import { defineConfig } from "vite";
import typegpu from "unplugin-typegpu/vite";

export default defineConfig({
  plugins: [typegpu()],
});
```

### HTML

```html
<canvas></canvas>
<script type="module" src="./index.ts"></script>
```
