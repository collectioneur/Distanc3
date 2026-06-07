import { useEffect, type RefObject } from "react";
import tgpu, { d } from "typegpu";
import {
  createShader,
  MAX_GPU_OBJECTS,
  MAX_INSTRUCTIONS,
  MAX_PICK_INSTRUCTIONS,
  MAX_PICK_OBJECTS,
  OPCODE_OP,
  OPCODE_PUSH_SHAPE,
  OPCODE_TRANSFORM_POP,
  OPCODE_TRANSFORM_PUSH,
  PICK_PASS_GIZMO,
  SHAPE_TYPE_INT,
  OP_TYPE_INT,
} from "./shader";
import {
  useSceneStore,
  findItem,
  getAncestorGroups,
  temporalStore,
  type ObjectGroup,
  type OpType,
  type SceneItem,
  type SceneRoot,
  type ShapeLayer,
} from "../store/sceneStore";
import { useGizmoStore } from "../store/gizmoStore";
import { useRenderStore } from "../store/renderStore";
import { clientToPickPixel, getCanvasAspect } from "./camera";
import {
  axisDeltaFromScreenDrag,
  beginAxisScreenDrag,
  getGizmoWorldAxes,
  getGizmoWorldPosition,
  getItemAncestorGroups,
  GIZMO_MODE_ROTATE,
  GIZMO_MODE_TRANSLATE,
  GIZMO_ARROW_LENGTH_RATIO,
  gizmoVisualScaleForDistance,
  hitTestTranslateGizmoScreen,
  stabilizeGizmoHoverAxis,
  worldToItemLocalPosition,
  type AxisScreenDragState,
  type GizmoAxis,
} from "./gizmo";

type GizmoGpuPickResult = {
  axis: GizmoAxis | null;
  skipped: boolean;
};

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

  compileItemSubtreeInstructions(root, selectedItemId, instructions);

  const count = instructions.length;

  while (instructions.length < MAX_INSTRUCTIONS) {
    instructions.push(EMPTY_INSTRUCTION);
  }

  return { instructions, count, enabled: count > 0, usesSceneSdf: false };
}

/** Compile one scene item's CSG subtree with ancestor transforms (for selection + pick). */
function compileItemSubtreeInstructions(
  root: SceneRoot,
  itemId: string,
  out: InstructionData[],
): boolean {
  const found = findItem(root, itemId);
  if (!found) return false;
  if (found.item.kind === "group" && found.item.items.length === 0) return false;

  const ancestors = getAncestorGroups(root, found.container.id);
  for (const group of ancestors) {
    pushTransformPush(group, out);
  }

  if (found.item.kind === "layer") {
    pushShapeInstruction(found.item, out);
  } else {
    pushTransformPush(found.item, out);
    compileItems(found.item.items, out);
    pushTransformPop(out);
  }

  for (let i = ancestors.length - 1; i >= 0; i--) {
    pushTransformPop(out);
  }

  return true;
}

/** Canvas pick: layers only. Groups are selected from the hierarchy panel. */
function collectPickableLayerIds(items: SceneItem[], out: string[]): void {
  for (const item of items) {
    if (item.kind === "layer") {
      out.push(item.id);
    } else if (item.items.length > 0) {
      collectPickableLayerIds(item.items, out);
    }
  }
}

function buildPickGpuData(root: SceneRoot): {
  instructions: InstructionData[];
  objectInfos: ObjectInfoData[];
  objectCount: number;
  itemIds: string[];
} {
  const instructions: InstructionData[] = [];
  const objectInfos: ObjectInfoData[] = [];
  const itemIds: string[] = [];

  const pickIds: string[] = [];
  collectPickableLayerIds(root.items, pickIds);

  for (const id of pickIds) {
    if (itemIds.length >= MAX_PICK_OBJECTS) break;

    const slot: InstructionData[] = [];
    if (!compileItemSubtreeInstructions(root, id, slot) || slot.length === 0) {
      continue;
    }
    if (instructions.length + slot.length > MAX_PICK_INSTRUCTIONS) break;

    const start = instructions.length;
    instructions.push(...slot);
    objectInfos.push({ start, count: slot.length });
    itemIds.push(id);
  }

  const objectCount = objectInfos.length;

  while (instructions.length < MAX_PICK_INSTRUCTIONS) {
    instructions.push(EMPTY_INSTRUCTION);
  }
  while (objectInfos.length < MAX_PICK_OBJECTS) {
    objectInfos.push(EMPTY_OBJECT_INFO);
  }

  return { instructions, objectInfos, objectCount, itemIds };
}

const DRAG_THRESHOLD_PX = 4;
const PICK_READBACK_BYTES_PER_ROW = 256;

const GIZMO_AXIS_DIR: Record<GizmoAxis, [number, number, number]> = {
  x: [1, 0, 0],
  y: [0, 1, 0],
  z: [0, 0, 1],
};

const GIZMO_AXIS_ID: Record<GizmoAxis, number> = {
  x: 1,
  y: 2,
  z: 3,
};

function axisIdToCursor(axis: GizmoAxis | null): string {
  if (!axis) return "default";
  if (axis === "x") return "ew-resize";
  if (axis === "y") return "ns-resize";
  return "nwse-resize";
}

function readPickIdFromBytes(data: Uint8Array): number {
  return Math.max(data[0] ?? 0, data[1] ?? 0, data[2] ?? 0);
}

const PICK_AXIS_ID: Record<number, GizmoAxis> = {
  1: "x",
  2: "y",
  3: "z",
};

function axisFromPickByte(pickId: number): GizmoAxis | null {
  return PICK_AXIS_ID[pickId] ?? null;
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
        cameraUniforms,
        sceneUniforms,
        selectionUniforms,
        gizmoUniforms,
        instructionsBuffer,
        objectInfoBuffer,
        selectionInstructionsBuffer,
        pickInstructionsBuffer,
        pickObjectInfoBuffer,
        pickUniforms,
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
      /** Live orbit preview while dragging; cleared on pointer up. */
      let orbitPreviewMouse: ReturnType<typeof d.vec2f> | null = null;
      let orbitPreviewRot: [number, number] | null = null;

      function getCameraRot(): [number, number] {
        return orbitPreviewRot ?? [rotX, rotY];
      }

      let lastRenderMode = -1;
      let lastSelectedItemId: string | null = null;
      let lastRootSelected = false;
      let sceneGpuDirty = true;
      let pickInProgress = false;

      let gizmoDragging = false;
      let gizmoHoverAxis: GizmoAxis | null = null;
      let gizmoDragAxis: GizmoAxis | null = null;
      let gizmoDragStartWorld: [number, number, number] | null = null;
      let gizmoDragScreen: AxisScreenDragState | null = null;
      let gizmoDragItemId: string | null = null;
      let gizmoDragAncestors: ObjectGroup[] | null = null;
      let cachedPickObjectCount = 0;

      let hoverGpuSeq = 0;
      let hoverPickRunning = false;
      let pendingHoverPick: {
        x: number;
        y: number;
        seq: number;
      } | null = null;

      function isTranslateGizmoMode(): boolean {
        return useGizmoStore.getState().mode === "translate";
      }

      function isGizmoVisibleMode(): boolean {
        const mode = useGizmoStore.getState().mode;
        return mode === "translate" || mode === "rotate";
      }

      function gizmoModeUniform(): number {
        return useGizmoStore.getState().mode === "rotate"
          ? GIZMO_MODE_ROTATE
          : GIZMO_MODE_TRANSLATE;
      }

      function writeGizmoDisabled() {
        gizmoUniforms.write({
          enabled: 0,
          activeAxis: 0,
          mode: 0,
          _pad0: 0,
          position: d.vec3f(0, 0, 0),
          _pad: 0,
          scale: 0,
          axisX: d.vec3f(1, 0, 0),
          _padX: 0,
          axisY: d.vec3f(0, 1, 0),
          _padY: 0,
          axisZ: d.vec3f(0, 0, 1),
          _padZ: 0,
        });
      }

      function clearGizmoInteraction() {
        if (gizmoDragging) endGizmoDrag();
        gizmoHoverAxis = null;
        hoverGpuSeq += 1;
        pendingHoverPick = null;
        canvas!.style.cursor = "default";
      }

      function updateGizmoUniforms(
        sceneRoot: SceneRoot,
        selectedItemId: string | null,
        cameraDistance: number,
        activeAxis: GizmoAxis | null,
      ) {
        if (!isGizmoVisibleMode() || !selectedItemId) {
          writeGizmoDisabled();
          return;
        }

        const pivotWorld = getGizmoWorldPosition(sceneRoot, selectedItemId);
        const axes = getGizmoWorldAxes(sceneRoot, selectedItemId);
        if (!pivotWorld || !axes) {
          writeGizmoDisabled();
          return;
        }

        const visualScale = gizmoVisualScaleForDistance(cameraDistance);
        gizmoUniforms.write({
          enabled: 1,
          activeAxis: activeAxis ? GIZMO_AXIS_ID[activeAxis] : 0,
          mode: gizmoModeUniform(),
          _pad0: 0,
          position: d.vec3f(pivotWorld[0], pivotWorld[1], pivotWorld[2]),
          _pad: 0,
          scale: visualScale,
          axisX: d.vec3f(axes.x[0], axes.x[1], axes.x[2]),
          _padX: 0,
          axisY: d.vec3f(axes.y[0], axes.y[1], axes.y[2]),
          _padY: 0,
          axisZ: d.vec3f(axes.z[0], axes.z[1], axes.z[2]),
          _padZ: 0,
        });
      }

      function setItemWorldPosition(
        itemId: string,
        worldPos: [number, number, number],
        ancestors: ObjectGroup[],
      ) {
        const found = findItem(useSceneStore.getState().root, itemId);
        if (!found) return;

        const localPos = worldToItemLocalPosition(worldPos, ancestors);
        const { updateLayer, updateGroup } = useSceneStore.getState();

        if (found.item.kind === "layer") {
          updateLayer(found.container.id, found.item.id, { position: localPos });
        } else {
          updateGroup(found.item.id, { position: localPos });
        }
      }

      async function pickGizmoAxisGpu(
        clientX: number,
        clientY: number,
      ): Promise<GizmoGpuPickResult> {
        if (pickInProgress) return { axis: null, skipped: true };
        if (!isTranslateGizmoMode()) return { axis: null, skipped: false };

        const { root: sceneRoot, selectedItemId } = useSceneStore.getState();
        if (!selectedItemId) return { axis: null, skipped: false };

        pickInProgress = true;
        try {
          syncCameraUniform();
          updateGizmoUniforms(
            sceneRoot,
            selectedItemId,
            distance,
            gizmoHoverAxis,
          );

          const pickPixel = clientToPickPixel(clientX, clientY, canvas!);
          if (!pickPixel) return { axis: null, skipped: false };

          pickUniforms.write({ objectCount: 0, pickPass: PICK_PASS_GIZMO });

          const encoder = root.device.createCommandEncoder();
          pipeline
            .with(encoder)
            .withColorAttachment({ view: pickTextureView })
            .draw(3);
          encoder.copyTextureToBuffer(
            { texture: pickTexture, origin: { x: pickPixel.x, y: pickPixel.y, z: 0 } },
            {
              buffer: pickReadbackBuffer,
              bytesPerRow: PICK_READBACK_BYTES_PER_ROW,
            },
            { width: 1, height: 1 },
          );
          root.device.queue.submit([encoder.finish()]);
          await root.device.queue.onSubmittedWorkDone();

          await pickReadbackBuffer.mapAsync(GPUMapMode.READ);
          const pickId = readPickIdFromBytes(
            new Uint8Array(pickReadbackBuffer.getMappedRange()),
          );
          pickReadbackBuffer.unmap();

          return { axis: axisFromPickByte(pickId), skipped: false };
        } finally {
          pickUniforms.write({
            objectCount: cachedPickObjectCount,
            pickPass: 0,
          });
          pickInProgress = false;
        }
      }

      async function drainGizmoHoverPicks() {
        if (hoverPickRunning) return;
        hoverPickRunning = true;
        try {
          while (pendingHoverPick) {
            const pick = pendingHoverPick;
            pendingHoverPick = null;

            let result = await pickGizmoAxisGpu(pick.x, pick.y);
            while (result.skipped && pick.seq === hoverGpuSeq) {
              pendingHoverPick = pendingHoverPick ?? pick;
              await new Promise<void>((resolve) =>
                requestAnimationFrame(() => resolve()),
              );
              if (pick.seq !== hoverGpuSeq) break;
              const latest = pendingHoverPick ?? pick;
              pendingHoverPick = null;
              result = await pickGizmoAxisGpu(latest.x, latest.y);
              if (latest.seq !== hoverGpuSeq) break;
            }

            if (pick.seq !== hoverGpuSeq) continue;
            if (result.skipped) continue;

            const { root: sceneRoot, selectedItemId } =
              useSceneStore.getState();
            if (!selectedItemId) {
              gizmoHoverAxis = null;
              canvas!.style.cursor = "default";
              continue;
            }

            const pivotWorld = getGizmoWorldPosition(sceneRoot, selectedItemId);
            if (!pivotWorld) {
              gizmoHoverAxis = null;
              canvas!.style.cursor = "default";
              continue;
            }

            const [rx, ry] = getCameraRot();
            const gizmoScale = gizmoVisualScaleForDistance(distance);
            const stabilized = stabilizeGizmoHoverAxis(
              pick.x,
              pick.y,
              canvas!,
              rx,
              ry,
              distance,
              pivotWorld,
              gizmoScale,
              result.axis,
              gizmoHoverAxis,
            );
            gizmoHoverAxis = stabilized;
            canvas!.style.cursor = axisIdToCursor(stabilized);
          }
        } finally {
          hoverPickRunning = false;
          if (pendingHoverPick) void drainGizmoHoverPicks();
        }
      }

      async function tryStartGizmoDrag(
        clientX: number,
        clientY: number,
      ): Promise<boolean> {
        if (!isTranslateGizmoMode()) return false;

        const { root: sceneRoot, selectedItemId } = useSceneStore.getState();
        if (!selectedItemId) return false;

        const pivotWorld = getGizmoWorldPosition(sceneRoot, selectedItemId);
        const ancestors = getItemAncestorGroups(sceneRoot, selectedItemId);
        if (!pivotWorld || !ancestors) return false;

        const [rx, ry] = getCameraRot();
        const cpuAxis = hitTestTranslateGizmoScreen(
          clientX,
          clientY,
          canvas!,
          rx,
          ry,
          distance,
          pivotWorld,
          gizmoVisualScaleForDistance(distance),
        );
        const gpuPick = await pickGizmoAxisGpu(clientX, clientY);
        const gpuAxis = gpuPick.axis;
        const axis = gpuAxis ?? cpuAxis;
        if (!axis) return false;

        const axisDir = GIZMO_AXIS_DIR[axis];
        const gizmoScale = gizmoVisualScaleForDistance(distance);
        const dragScreen = beginAxisScreenDrag(
          clientX,
          clientY,
          canvas!,
          rx,
          ry,
          distance,
          pivotWorld,
          axisDir,
          gizmoScale * GIZMO_ARROW_LENGTH_RATIO,
        );
        if (!dragScreen) return false;

        gizmoDragging = true;
        gizmoDragAxis = axis;
        gizmoDragItemId = selectedItemId;
        gizmoDragAncestors = ancestors;
        gizmoDragStartWorld = pivotWorld;
        gizmoDragScreen = dragScreen;
        gizmoHoverAxis = axis;
        temporalStore.getState().pause();
        canvas!.style.cursor = axisIdToCursor(axis);
        return true;
      }

      function updateGizmoDrag(clientX: number, clientY: number) {
        if (
          !gizmoDragging ||
          !gizmoDragAxis ||
          !gizmoDragStartWorld ||
          !gizmoDragScreen ||
          !gizmoDragItemId ||
          !gizmoDragAncestors
        ) {
          return;
        }

        const axisDir = GIZMO_AXIS_DIR[gizmoDragAxis];
        const deltaT = axisDeltaFromScreenDrag(
          clientX,
          clientY,
          gizmoDragScreen,
          gizmoDragAxis,
        );

        const newPivot: [number, number, number] = [
          gizmoDragStartWorld[0] + axisDir[0] * deltaT,
          gizmoDragStartWorld[1] + axisDir[1] * deltaT,
          gizmoDragStartWorld[2] + axisDir[2] * deltaT,
        ];
        const freshAncestors = getItemAncestorGroups(
          useSceneStore.getState().root,
          gizmoDragItemId,
        );
        if (!freshAncestors) return;
        setItemWorldPosition(gizmoDragItemId, newPivot, freshAncestors);
      }

      function endGizmoDrag() {
        if (!gizmoDragging) return;
        gizmoDragging = false;
        gizmoDragAxis = null;
        gizmoDragStartWorld = null;
        gizmoDragScreen = null;
        gizmoDragItemId = null;
        gizmoDragAncestors = null;
        temporalStore.getState().resume();
      }

      function updateGizmoHover(clientX: number, clientY: number) {
        if (!isTranslateGizmoMode()) {
          hoverGpuSeq += 1;
          pendingHoverPick = null;
          gizmoHoverAxis = null;
          canvas!.style.cursor = "default";
          return;
        }

        const { selectedItemId } = useSceneStore.getState();
        if (!selectedItemId) {
          hoverGpuSeq += 1;
          pendingHoverPick = null;
          gizmoHoverAxis = null;
          canvas!.style.cursor = "default";
          return;
        }

        const seq = ++hoverGpuSeq;
        pendingHoverPick = { x: clientX, y: clientY, seq };
        void drainGizmoHoverPicks();
      }

      async function pickAt(clientX: number, clientY: number) {
        if (pickInProgress) return;
        pickInProgress = true;

        try {
          const rect = canvas!.getBoundingClientRect();
          if (rect.width <= 0 || rect.height <= 0) return;

          pickUniforms.write({
            objectCount: cachedPickObjectCount,
            pickPass: 1,
          });

          const pickPixel = clientToPickPixel(clientX, clientY, canvas!);
          if (!pickPixel) return;
          const pickX = pickPixel.x;
          const pickY = pickPixel.y;

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
          if (!found || found.item.kind !== "layer") return;

          selectItem(found.container.id, itemId);
        } finally {
          pickUniforms.write({
            objectCount: cachedPickObjectCount,
            pickPass: 0,
          });
          pickInProgress = false;
        }
      }

      const onPointerDown = (e: PointerEvent) => {
        if (e.button !== 0) return;
        void tryStartGizmoDrag(e.clientX, e.clientY).then((started) => {
          if (started) return;
          pointerDown = true;
          dragged = false;
          orbitPreviewMouse = null;
          orbitPreviewRot = null;
          dragStartX = e.clientX;
          dragStartY = e.clientY;
        });
      };

      const onPointerMove = (e: PointerEvent) => {
        if (gizmoDragging) {
          updateGizmoDrag(e.clientX, e.clientY);
          return;
        }

        if (!pointerDown) {
          updateGizmoHover(e.clientX, e.clientY);
          return;
        }

        const dx = e.clientX - dragStartX;
        const dy = e.clientY - dragStartY;
        if (!dragged && Math.hypot(dx, dy) >= DRAG_THRESHOLD_PX) {
          dragged = true;
          canvas!.style.cursor = "grabbing";
        }
        if (!dragged) return;
        const totalDx =
          ((e.clientX - dragStartX) / window.innerWidth) * Math.PI * 2;
        const totalDy =
          ((e.clientY - dragStartY) / window.innerHeight) * Math.PI * 2;
        const previewRx = rotX + totalDx;
        const previewRy = rotY + totalDy;
        orbitPreviewRot = [previewRx, previewRy];
        orbitPreviewMouse = d.vec2f(previewRx, previewRy);
      };

      const onPointerUp = (e: PointerEvent) => {
        if (gizmoDragging && e.button === 0) {
          endGizmoDrag();
          updateGizmoHover(e.clientX, e.clientY);
          return;
        }

        if (!pointerDown || e.button !== 0) return;
        pointerDown = false;

        if (dragged) {
          rotX += ((e.clientX - dragStartX) / window.innerWidth) * Math.PI * 2;
          rotY += ((e.clientY - dragStartY) / window.innerHeight) * Math.PI * 2;
          orbitPreviewMouse = null;
          orbitPreviewRot = null;
          updateGizmoHover(e.clientX, e.clientY);
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
        syncCameraUniform();
      };

      const unsubGizmoMode = useGizmoStore.subscribe((state, prev) => {
        if (state.mode === prev.mode) return;
        clearGizmoInteraction();
      });

      canvas.addEventListener("pointerdown", onPointerDown);
      window.addEventListener("pointermove", onPointerMove);
      window.addEventListener("pointerup", onPointerUp);
      canvas.addEventListener("wheel", onWheel, { passive: false });

      function updateSize() {
        const dpr = Math.min(window.devicePixelRatio ?? 1, 1.0);
        canvas!.width = canvas!.clientWidth * dpr;
        canvas!.height = canvas!.clientHeight * dpr;
        if (canvas!.width > 0 && canvas!.height > 0) {
          resizePickTarget(canvas!.width, canvas!.height);
          syncCameraUniform();
        }
      }

      const startTime = performance.now();

      function syncCameraUniform() {
        const mouse = orbitPreviewMouse ?? d.vec2f(rotX, rotY);
        const aspect = getCanvasAspect(canvas!);
        cameraUniforms.write({
          time: (performance.now() - startTime) / 1000,
          aspect,
          mouse,
          distance,
        });
      }

      updateSize();

      const observer = new ResizeObserver(updateSize);
      observer.observe(canvas);
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
        sceneUniforms.write({
          objectCount,
          renderMode,
        });

        const selection = buildSelectionGpuData(
          sceneRoot,
          selectedItemId,
          rootSelected,
        );
        selectionInstructionsBuffer.write(selection.instructions);
        selectionUniforms.write({
          enabled: selection.enabled ? 1 : 0,
          usesSceneSdf: selection.usesSceneSdf ? 1 : 0,
          count: selection.count,
        });

        const pick = buildPickGpuData(sceneRoot);
        pickInstructionsBuffer.write(pick.instructions);
        pickObjectInfoBuffer.write(pick.objectInfos);
        cachedPickObjectCount = pick.objectCount;
        pickUniforms.write({
          objectCount: cachedPickObjectCount,
          pickPass: 0,
        });
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

        const activeGizmoAxis = gizmoDragging
          ? gizmoDragAxis
          : gizmoHoverAxis;
        updateGizmoUniforms(sceneRoot, selectedItemId, distance, activeGizmoAxis);

        syncCameraUniform();
        pipeline.withColorAttachment({ view: context }).draw(3);
        animFrameId = requestAnimationFrame(frame);
      }

      animFrameId = requestAnimationFrame(frame);

      registeredCleanup = () => {
        unsubGizmoMode();
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
