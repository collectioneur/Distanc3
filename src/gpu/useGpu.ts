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
  rootSelected: boolean,
): {
  instructions: InstructionData[];
  count: number;
  enabled: boolean;
  usesSceneSdf: boolean;
} {
  const instructions: InstructionData[] = [];

  if (!selectedItemId) {
    const enabled = rootSelected && root.items.length > 0;

    while (instructions.length < MAX_INSTRUCTIONS) {
      instructions.push(EMPTY_INSTRUCTION);
    }

    return { instructions, count: 0, enabled, usesSceneSdf: enabled };
  }

  const found = findItem(root, selectedItemId);
  if (!found) {
    while (instructions.length < MAX_INSTRUCTIONS) {
      instructions.push(EMPTY_INSTRUCTION);
    }
    return { instructions, count: 0, enabled: false, usesSceneSdf: false };
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

  return { instructions, count, enabled: count > 0, usesSceneSdf: false };
}

function compilePickItem(item: SceneItem, out: InstructionData[]): void {
  if (item.kind === "layer") {
    pushShapeInstruction(item, out);
  } else if (item.items.length > 0) {
    pushTransformPush(item, out);
    compileItems(item.items, out);
    pushTransformPop(out);
  }
}

/** MVP: one pick slot per top-level scene item (max MAX_GPU_OBJECTS). */
function buildPickGpuData(root: SceneRoot): {
  instructions: InstructionData[];
  objectInfos: ObjectInfoData[];
  objectCount: number;
  itemIds: string[];
} {
  const instructions: InstructionData[] = [];
  const objectInfos: ObjectInfoData[] = [];
  const itemIds: string[] = [];

  for (const item of root.items) {
    if (item.kind === "group" && item.items.length === 0) continue;

    const start = instructions.length;
    compilePickItem(item, instructions);
    if (instructions.length === start) continue;

    objectInfos.push({ start, count: instructions.length - start });
    itemIds.push(item.id);

    if (itemIds.length >= MAX_GPU_OBJECTS) break;
  }

  const objectCount = objectInfos.length;

  while (instructions.length < MAX_INSTRUCTIONS) {
    instructions.push(EMPTY_INSTRUCTION);
  }
  while (objectInfos.length < MAX_GPU_OBJECTS) {
    objectInfos.push(EMPTY_OBJECT_INFO);
  }

  return { instructions, objectInfos, objectCount, itemIds };
}

const DRAG_THRESHOLD_PX = 4;
const PICK_READBACK_BYTES_PER_ROW = 256;

function readPickIdFromBytes(data: Uint8Array): number {
  return Math.max(data[0] ?? 0, data[1] ?? 0, data[2] ?? 0);
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
        selectionUsesSceneSdfUniform,
        pickInstructionsBuffer,
        pickObjectInfoBuffer,
        pickObjectCountUniform,
        pickUvUniform,
        pickPassUniform,
      } = createShader(root);

      const pickFormat = navigator.gpu.getPreferredCanvasFormat();
      let pickTexture = root.device.createTexture({
        size: [1, 1],
        format: pickFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
      });
      let pickTextureView = pickTexture.createView();
      const pickReadbackBuffer = root.device.createBuffer({
        size: PICK_READBACK_BYTES_PER_ROW,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      function resizePickTarget(width: number, height: number) {
        pickTexture.destroy();
        pickTexture = root.device.createTexture({
          size: [width, height],
          format: pickFormat,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        pickTextureView = pickTexture.createView();
      }

      let pickItemIds: string[] = [];

      let pointerDown = false;
      let dragged = false;
      let dragStartX = 0;
      let dragStartY = 0;
      let rotX = 0.3;
      let rotY = -0.4;
      let distance = 2.5;

      let lastRenderMode = -1;
      let lastSelectedItemId: string | null = null;
      let lastRootSelected = false;
      let sceneGpuDirty = true;
      let pickInProgress = false;

      async function pickAt(clientX: number, clientY: number) {
        if (pickInProgress) return;
        pickInProgress = true;

        try {
          const rect = canvas!.getBoundingClientRect();
          if (rect.width <= 0 || rect.height <= 0) return;

          const u = (clientX - rect.left) / rect.width;
          const v = 1 - (clientY - rect.top) / rect.height;

          pickPassUniform.write(1);

          const pickX = Math.min(
            canvas!.width - 1,
            Math.max(0, Math.floor(u * canvas!.width)),
          );
          const pickY = Math.min(
            canvas!.height - 1,
            Math.max(0, Math.floor((1 - v) * canvas!.height)),
          );

          const encoder = root.device.createCommandEncoder();
          pipeline
            .with(encoder)
            .withColorAttachment({ view: pickTextureView })
            .draw(3);
          encoder.copyTextureToBuffer(
            { texture: pickTexture, origin: { x: pickX, y: pickY, z: 0 } },
            { buffer: pickReadbackBuffer, bytesPerRow: PICK_READBACK_BYTES_PER_ROW },
            { width: 1, height: 1 },
          );
          root.device.queue.submit([encoder.finish()]);
          await root.device.queue.onSubmittedWorkDone();

          await pickReadbackBuffer.mapAsync(GPUMapMode.READ);
          const pickId = readPickIdFromBytes(
            new Uint8Array(pickReadbackBuffer.getMappedRange()),
          );
          pickReadbackBuffer.unmap();

          const { root: sceneRoot, selectItem, deselect } =
            useSceneStore.getState();

          if (pickId === 0) {
            deselect();
            return;
          }

          const itemId = pickItemIds[pickId - 1];
          if (!itemId) return;

          const found = findItem(sceneRoot, itemId);
          if (!found) return;

          const containerId =
            found.item.kind === "group" ? found.item.id : found.container.id;
          selectItem(containerId, itemId);
        } finally {
          pickPassUniform.write(0);
          pickInProgress = false;
        }
      }

      const onPointerDown = (e: PointerEvent) => {
        if (e.button !== 0) return;
        pointerDown = true;
        dragged = false;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
      };

      const onPointerMove = (e: PointerEvent) => {
        if (!pointerDown) return;
        const dx = e.clientX - dragStartX;
        const dy = e.clientY - dragStartY;
        if (!dragged && Math.hypot(dx, dy) >= DRAG_THRESHOLD_PX) {
          dragged = true;
        }
        if (!dragged) return;
        const totalDx =
          ((e.clientX - dragStartX) / window.innerWidth) * Math.PI * 2;
        const totalDy =
          ((e.clientY - dragStartY) / window.innerHeight) * Math.PI * 2;
        mouseUniform.write(d.vec2f(rotX + totalDx, rotY + totalDy));
      };

      const onPointerUp = (e: PointerEvent) => {
        if (!pointerDown || e.button !== 0) return;
        pointerDown = false;

        if (dragged) {
          rotX += ((e.clientX - dragStartX) / window.innerWidth) * Math.PI * 2;
          rotY += ((e.clientY - dragStartY) / window.innerHeight) * Math.PI * 2;
          mouseUniform.write(d.vec2f(rotX, rotY));
        } else {
          void pickAt(e.clientX, e.clientY);
        }
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

      canvas.addEventListener("pointerdown", onPointerDown);
      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", onPointerUp);
      canvas.addEventListener("wheel", onWheel, { passive: false });

      function updateSize() {
        const dpr = Math.min(window.devicePixelRatio ?? 1, 1.0);
        canvas!.width = canvas!.clientWidth * dpr;
        canvas!.height = canvas!.clientHeight * dpr;
        if (canvas!.width > 0 && canvas!.height > 0) {
          aspectUniform.write(canvas!.width / canvas!.height);
          resizePickTarget(canvas!.width, canvas!.height);
        }
      }

      updateSize();

      const observer = new ResizeObserver(updateSize);
      observer.observe(canvas);

      const startTime = performance.now();
      const TARGET_MS = 1000 / 60;
      let lastFrameTime = 0;
      let lastRoot: SceneRoot | null = null;

      function uploadSceneGpu(
        sceneRoot: SceneRoot,
        selectedItemId: string | null,
        rootSelected: boolean,
        renderMode: number,
      ) {
        const { instructions, objectInfos, objectCount } = buildGpuData(sceneRoot);
        instructionsBuffer.write(instructions);
        objectInfoBuffer.write(objectInfos);
        objectCountUniform.write(objectCount);
        renderModeUniform.write(renderMode);

        const selection = buildSelectionGpuData(
          sceneRoot,
          selectedItemId,
          rootSelected,
        );
        selectionInstructionsBuffer.write(selection.instructions);
        selectionCountUniform.write(selection.count);
        selectionEnabledUniform.write(selection.enabled ? 1 : 0);
        selectionUsesSceneSdfUniform.write(selection.usesSceneSdf ? 1 : 0);

        const pick = buildPickGpuData(sceneRoot);
        pickInstructionsBuffer.write(pick.instructions);
        pickObjectInfoBuffer.write(pick.objectInfos);
        pickObjectCountUniform.write(pick.objectCount);
        pickItemIds = pick.itemIds;
      }

      const initialState = useSceneStore.getState();
      uploadSceneGpu(
        initialState.root,
        initialState.selectedItemId,
        initialState.rootSelected,
        useRenderStore.getState().renderMode,
      );
      lastRoot = initialState.root;
      lastRenderMode = useRenderStore.getState().renderMode;
      lastSelectedItemId = initialState.selectedItemId;
      lastRootSelected = initialState.rootSelected;

      function frame(now: number) {
        if (pickInProgress) {
          animFrameId = requestAnimationFrame(frame);
          return;
        }

        if (now - lastFrameTime < TARGET_MS) {
          animFrameId = requestAnimationFrame(frame);
          return;
        }
        lastFrameTime = now;

        const { root: sceneRoot, selectedItemId, rootSelected } =
          useSceneStore.getState();
        const renderMode = useRenderStore.getState().renderMode;

        if (
          sceneGpuDirty ||
          sceneRoot !== lastRoot ||
          renderMode !== lastRenderMode ||
          selectedItemId !== lastSelectedItemId ||
          rootSelected !== lastRootSelected
        ) {
          uploadSceneGpu(sceneRoot, selectedItemId, rootSelected, renderMode);

          lastRoot = sceneRoot;
          lastRenderMode = renderMode;
          lastSelectedItemId = selectedItemId;
          lastRootSelected = rootSelected;
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
        canvas.removeEventListener("pointerdown", onPointerDown);
        window.removeEventListener("pointermove", onPointerMove);
        window.removeEventListener("pointerup", onPointerUp);
        canvas.removeEventListener("wheel", onWheel);
        root.destroy();
        pickTexture.destroy();
        pickReadbackBuffer.destroy();
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
