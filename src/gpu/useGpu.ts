import { useEffect, type RefObject } from "react";
import tgpu, { d } from "typegpu";
import {
  createShader,
  MAX_OBJECTS,
  MAX_NODES_PER_OBJECT,
  SHAPE_TYPE_INT,
  OP_TYPE_INT,
} from "./shader";
import {
  useSceneStore,
  type CsgNode,
  type SceneObject,
} from "../store/sceneStore";
import { useRenderStore } from "../store/renderStore";

type InstructionData = {
  opcode: number;
  shapeType: number;
  opType: number;
  smoothK: number;
  position: ReturnType<typeof d.vec3f>;
  _pad: number;
  params: ReturnType<typeof d.vec4f>;
  rotation: ReturnType<typeof d.vec3f>;
  _pad2: number;
};

type ObjectInfoData = { start: number; count: number };

const EMPTY_INSTRUCTION: InstructionData = {
  opcode: 0,
  shapeType: 0,
  opType: 0,
  smoothK: 0,
  position: d.vec3f(0, 0, 0),
  _pad: 0,
  params: d.vec4f(0, 0, 0, 0),
  rotation: d.vec3f(0, 0, 0),
  _pad2: 0,
};

const EMPTY_OBJECT_INFO: ObjectInfoData = { start: 0, count: 0 };
const TOTAL_INSTRUCTIONS = MAX_OBJECTS * MAX_NODES_PER_OBJECT;

// Compile a CSG binary tree into a postorder instruction sequence.
// PUSH_SHAPE instructions evaluate a primitive SDF;
// OP instructions pop two values, apply a boolean op, and push the result.
function compileCsgTree(node: CsgNode, out: InstructionData[]): void {
  if (node.kind === "shape") {
    const DEG_TO_RAD = Math.PI / 180;
    const [rx, ry, rz] = node.rotation ?? [0, 0, 0];
    out.push({
      opcode: 0,
      shapeType: SHAPE_TYPE_INT[node.shapeType],
      opType: 0,
      smoothK: 0,
      position: d.vec3f(node.position[0], node.position[1], node.position[2]),
      _pad: 0,
      params: d.vec4f(
        node.params[0],
        node.params[1],
        node.params[2],
        node.params[3],
      ),
      rotation: d.vec3f(rx * DEG_TO_RAD, ry * DEG_TO_RAD, rz * DEG_TO_RAD),
      _pad2: 0,
    });
  } else {
    compileCsgTree(node.left, out);
    compileCsgTree(node.right, out);
    out.push({
      opcode: 1,
      shapeType: 0,
      opType: OP_TYPE_INT[node.op],
      smoothK: node.smoothK,
      position: d.vec3f(0, 0, 0),
      _pad: 0,
      params: d.vec4f(0, 0, 0, 0),
      rotation: d.vec3f(0, 0, 0),
      _pad2: 0,
    });
  }
}

function buildGpuData(objects: SceneObject[]): {
  instructions: InstructionData[];
  objectInfos: ObjectInfoData[];
  objectCount: number;
} {
  const instructions: InstructionData[] = [];
  const objectInfos: ObjectInfoData[] = [];
  let currentStart = 0;

  for (const obj of objects) {
    if (!obj.root) continue; // skip empty objects
    const objInstructions: InstructionData[] = [];
    compileCsgTree(obj.root, objInstructions);
    objectInfos.push({ start: currentStart, count: objInstructions.length });
    instructions.push(...objInstructions);
    currentStart += objInstructions.length;
  }

  const objectCount = objectInfos.length;

  // Pad to fixed GPU buffer sizes
  while (instructions.length < TOTAL_INSTRUCTIONS) {
    instructions.push(EMPTY_INSTRUCTION);
  }
  while (objectInfos.length < MAX_OBJECTS) {
    objectInfos.push(EMPTY_OBJECT_INFO);
  }

  return { instructions, objectInfos, objectCount };
}

export function useGpu(canvasRef: RefObject<HTMLCanvasElement | null>) {
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let cancelled = false;
    let animFrameId = 0;
    let registeredCleanup: (() => void) | undefined;

    tgpu.init().then((root) => {
      if (cancelled) {
        root.destroy();
        return;
      }

      const context = root.configureContext({
        canvas,
        alphaMode: "premultiplied",
      });

      const {
        pipeline,
        timeUniform,
        aspectUniform,
        mouseUniform,
        distanceUniform,
        instructionsBuffer,
        objectInfoBuffer,
        objectCountUniform,
        renderModeUniform,
      } = createShader(root);

      let isDragging = false;
      let dragStartX = 0;
      let dragStartY = 0;
      let rotX = 0.3;
      let rotY = -0.4;
      let distance = 2.5;

      const onMouseDown = (e: MouseEvent) => {
        isDragging = true;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
      };

      const onMouseMove = (e: MouseEvent) => {
        if (!isDragging) return;
        const dx = ((e.clientX - dragStartX) / window.innerWidth) * Math.PI * 2;
        const dy =
          ((e.clientY - dragStartY) / window.innerHeight) * Math.PI * 2;
        mouseUniform.write(d.vec2f(rotX + dx, rotY + dy));
      };

      const onMouseUp = (e: MouseEvent) => {
        if (!isDragging) return;
        isDragging = false;
        rotX += ((e.clientX - dragStartX) / window.innerWidth) * Math.PI * 2;
        rotY += ((e.clientY - dragStartY) / window.innerHeight) * Math.PI * 2;
      };

      const onWheel = (e: WheelEvent) => {
        e.preventDefault();
        const px = e.deltaMode === 1 ? e.deltaY * 10.0 : e.deltaY;
        distance = Math.max(
          0.5,
          Math.min(20.0, distance * Math.exp(px * 0.0001)),
        );
        distanceUniform.write(distance);
      };

      canvas.addEventListener("mousedown", onMouseDown);
      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", onMouseUp);
      canvas.addEventListener("wheel", onWheel, { passive: false });

      function updateSize() {
        const dpr = Math.min(window.devicePixelRatio ?? 1, 1.0);
        canvas!.width = canvas!.clientWidth * dpr;
        canvas!.height = canvas!.clientHeight * dpr;
        if (canvas!.height > 0) {
          aspectUniform.write(canvas!.width / canvas!.height);
        }
      }

      updateSize();

      const observer = new ResizeObserver(updateSize);
      observer.observe(canvas);

      const startTime = performance.now();
      const TARGET_MS = 1000 / 60;
      let lastFrameTime = 0;
      let lastObjects: SceneObject[] = [];
      let lastRenderMode = -1;
      let sceneGpuDirty = true;

      function frame(now: number) {
        if (now - lastFrameTime < TARGET_MS) {
          animFrameId = requestAnimationFrame(frame);
          return;
        }
        lastFrameTime = now;

        const { objects } = useSceneStore.getState();
        const renderMode = useRenderStore.getState().renderMode;

        if (
          sceneGpuDirty ||
          objects !== lastObjects ||
          renderMode !== lastRenderMode
        ) {
          const { instructions, objectInfos, objectCount } =
            buildGpuData(objects);
          instructionsBuffer.write(instructions);
          objectInfoBuffer.write(objectInfos);
          objectCountUniform.write(objectCount);
          renderModeUniform.write(renderMode);
          lastObjects = objects;
          lastRenderMode = renderMode;
          sceneGpuDirty = false;
        }

        timeUniform.write((performance.now() - startTime) / 1000);
        pipeline.withColorAttachment({ view: context }).draw(3);
        animFrameId = requestAnimationFrame(frame);
      }

      animFrameId = requestAnimationFrame(frame);

      registeredCleanup = () => {
        observer.disconnect();
        cancelAnimationFrame(animFrameId);
        canvas.removeEventListener("mousedown", onMouseDown);
        window.removeEventListener("mousemove", onMouseMove);
        window.removeEventListener("mouseup", onMouseUp);
        canvas.removeEventListener("wheel", onWheel);
        root.destroy();
      };
    });

    return () => {
      cancelled = true;
      cancelAnimationFrame(animFrameId);
      registeredCleanup?.();
    };
    // canvasRef is a stable ref object — safe to omit from deps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}
