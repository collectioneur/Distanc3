import { useEffect, type RefObject } from "react";
import tgpu, { d } from "typegpu";
import {
  createShader,
  MAX_GPU_OBJECTS,
  MAX_INSTRUCTIONS,
  OPCODE_OP,
  OPCODE_PUSH_SHAPE,
  OPCODE_TRANSFORM_POP,
  OPCODE_TRANSFORM_PUSH,
  SHAPE_TYPE_INT,
  OP_TYPE_INT,
} from "./shader";
import {
  useSceneStore,
  findItem,
  getAncestorGroups,
  type ObjectGroup,
  type OpType,
  type SceneItem,
  type SceneRoot,
  type ShapeLayer,
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

const DEG_TO_RAD = Math.PI / 180;

function pushShapeInstruction(layer: ShapeLayer, out: InstructionData[]): void {
  const [rx, ry, rz] = layer.rotation;
  out.push({
    opcode: OPCODE_PUSH_SHAPE,
    shapeType: SHAPE_TYPE_INT[layer.shapeType],
    opType: 0,
    smoothK: 0,
    position: d.vec3f(layer.position[0], layer.position[1], layer.position[2]),
    _pad: 0,
    params: d.vec4f(
      layer.params[0],
      layer.params[1],
      layer.params[2],
      layer.params[3],
    ),
    rotation: d.vec3f(rx * DEG_TO_RAD, ry * DEG_TO_RAD, rz * DEG_TO_RAD),
    _pad2: 0,
  });
}

function pushOpInstruction(op: OpType, smoothK: number, out: InstructionData[]): void {
  out.push({
    opcode: OPCODE_OP,
    shapeType: 0,
    opType: OP_TYPE_INT[op],
    smoothK,
    position: d.vec3f(0, 0, 0),
    _pad: 0,
    params: d.vec4f(0, 0, 0, 0),
    rotation: d.vec3f(0, 0, 0),
    _pad2: 0,
  });
}

function pushTransformPush(group: ObjectGroup, out: InstructionData[]): void {
  const [rx, ry, rz] = group.rotation;
  out.push({
    opcode: OPCODE_TRANSFORM_PUSH,
    shapeType: 0,
    opType: 0,
    smoothK: 0,
    position: d.vec3f(group.position[0], group.position[1], group.position[2]),
    _pad: 0,
    params: d.vec4f(group.scale[0], group.scale[1], group.scale[2], 0),
    rotation: d.vec3f(rx * DEG_TO_RAD, ry * DEG_TO_RAD, rz * DEG_TO_RAD),
    _pad2: 0,
  });
}

function pushTransformPop(out: InstructionData[]): void {
  out.push({
    opcode: OPCODE_TRANSFORM_POP,
    shapeType: 0,
    opType: 0,
    smoothK: 0,
    position: d.vec3f(0, 0, 0),
    _pad: 0,
    params: d.vec4f(0, 0, 0, 0),
    rotation: d.vec3f(0, 0, 0),
    _pad2: 0,
  });
}

function compileItems(items: SceneItem[], out: InstructionData[]): void {
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (item.kind === "layer") {
      pushShapeInstruction(item, out);
      if (i > 0) pushOpInstruction(item.op, item.smoothK, out);
    } else {
      if (item.items.length > 0) {
        pushTransformPush(item, out);
        compileItems(item.items, out);
        pushTransformPop(out);
      }
      if (i > 0) pushOpInstruction(item.op, item.smoothK, out);
    }
  }
}

function buildGpuData(root: SceneRoot): {
  instructions: InstructionData[];
  objectInfos: ObjectInfoData[];
  objectCount: number;
} {
  const instructions: InstructionData[] = [];
  const objectInfos: ObjectInfoData[] = [];

  if (root.items.length > 0) {
    compileItems(root.items, instructions);
    objectInfos.push({ start: 0, count: instructions.length });
  }

  const objectCount = objectInfos.length;

  while (instructions.length < MAX_INSTRUCTIONS) {
    instructions.push(EMPTY_INSTRUCTION);
  }
  while (objectInfos.length < MAX_GPU_OBJECTS) {
    objectInfos.push(EMPTY_OBJECT_INFO);
  }

  return { instructions, objectInfos, objectCount };
}

function buildSelectionGpuData(
  root: SceneRoot,
  selectedItemId: string | null,
): {
  instructions: InstructionData[];
  count: number;
  enabled: boolean;
} {
  const instructions: InstructionData[] = [];

  if (!selectedItemId) {
    if (root.items.length > 0) {
      compileItems(root.items, instructions);
    }

    const count = instructions.length;

    while (instructions.length < MAX_INSTRUCTIONS) {
      instructions.push(EMPTY_INSTRUCTION);
    }

    return { instructions, count, enabled: count > 0 };
  }

  const found = findItem(root, selectedItemId);
  if (!found) {
    while (instructions.length < MAX_INSTRUCTIONS) {
      instructions.push(EMPTY_INSTRUCTION);
    }
    return { instructions, count: 0, enabled: false };
  }

  const ancestors = getAncestorGroups(root, found.container.id);
  for (const group of ancestors) {
    pushTransformPush(group, instructions);
  }

  if (found.item.kind === "layer") {
    pushShapeInstruction(found.item, instructions);
  } else if (found.item.items.length > 0) {
    pushTransformPush(found.item, instructions);
    compileItems(found.item.items, instructions);
    pushTransformPop(instructions);
  }

  for (let i = ancestors.length - 1; i >= 0; i--) {
    pushTransformPop(instructions);
  }

  const count = instructions.length;

  while (instructions.length < MAX_INSTRUCTIONS) {
    instructions.push(EMPTY_INSTRUCTION);
  }

  return { instructions, count, enabled: count > 0 };
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
        selectionInstructionsBuffer,
        selectionCountUniform,
        selectionEnabledUniform,
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
      let lastRoot: SceneRoot | null = null;
      let lastRenderMode = -1;
      let lastSelectedItemId: string | null = null;
      let sceneGpuDirty = true;

      function frame(now: number) {
        if (now - lastFrameTime < TARGET_MS) {
          animFrameId = requestAnimationFrame(frame);
          return;
        }
        lastFrameTime = now;

        const { root: sceneRoot, selectedItemId } = useSceneStore.getState();
        const renderMode = useRenderStore.getState().renderMode;

        if (
          sceneGpuDirty ||
          sceneRoot !== lastRoot ||
          renderMode !== lastRenderMode ||
          selectedItemId !== lastSelectedItemId
        ) {
          const { instructions, objectInfos, objectCount } =
            buildGpuData(sceneRoot);
          instructionsBuffer.write(instructions);
          objectInfoBuffer.write(objectInfos);
          objectCountUniform.write(objectCount);
          renderModeUniform.write(renderMode);

          const selection = buildSelectionGpuData(sceneRoot, selectedItemId);
          selectionInstructionsBuffer.write(selection.instructions);
          selectionCountUniform.write(selection.count);
          selectionEnabledUniform.write(selection.enabled ? 1 : 0);

          lastRoot = sceneRoot;
          lastRenderMode = renderMode;
          lastSelectedItemId = selectedItemId;
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}
