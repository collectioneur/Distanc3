import { create } from "zustand";
import { shallow } from "zustand/shallow";
import { temporal } from "zundo";
import { applyMoveToRoot } from "../scene/arboristAdapter";
import { loadInitialState } from "./persistence";
import { showToast } from "../utils/toast";

export type ShapeType =
  | "sphere"
  | "box"
  | "torus"
  | "cylinder"
  | "capsule"
  | "cone"
  | "roundedBox"
  | "boxFrame"
  | "cappedTorus"
  | "link"
  | "hexPrism"
  | "triPrism"
  | "roundedCylinder"
  | "roundCone"
  | "solidAngle"
  | "cutSphere"
  | "cutHollowSphere"
  | "deathStar"
  | "rhombus"
  | "octahedron"
  | "pyramid"
  | "vesica";
export type OpType = "union" | "subtract" | "intersect" | "sUnion" | "sSubtract" | "sIntersect";

export type ShapeLayer = {
  kind: "layer";
  id: string;
  shapeType: ShapeType;
  name: string;
  position: [number, number, number];
  rotation: [number, number, number]; // Euler XYZ in degrees
  scale: [number, number, number]; // local scale applied after own rotation
  params: [number, number, number, number];
  op: OpType; // ignored when index 0 in parent items
  smoothK: number;
};

export type ObjectGroup = {
  kind: "group";
  id: string;
  name: string;
  position: [number, number, number];
  rotation: [number, number, number]; // Euler XYZ in degrees
  scale: [number, number, number];
  op: OpType;
  smoothK: number;
  items: SceneItem[];
};

export type SceneItem = ShapeLayer | ObjectGroup;

export type SceneRoot = {
  id: string;
  name: string;
  items: SceneItem[];
};

export type SceneContainer = SceneRoot | ObjectGroup;

export const MAX_INSTRUCTIONS = 256;
/** GPU objectInfo buffer slots — only root slot [0] is used for scene render. */
export const MAX_GPU_OBJECTS = 8;
/** Max pickable scene items (IDs encoded in 1 byte: 1–255). */
export const MAX_PICK_OBJECTS = 255;
/** Packed instruction capacity for the pick CSG buffer (separate from scene). */
export const MAX_PICK_INSTRUCTIONS = 1024;
/** Max nested group transform stack depth in shader. */
export const MAX_TRANSFORM_DEPTH = 16;

const DEFAULT_PARAMS: Record<ShapeType, [number, number, number, number]> = {
  sphere: [0.5, 0, 0, 0],
  box: [0.4, 0.4, 0.4, 0],
  torus: [0.4, 0.15, 0, 0],
  cylinder: [0.3, 0.5, 0, 0],
  capsule: [0.25, 0.4, 0, 0],
  cone: [0.4, 0.6, 0, 0],
  roundedBox: [0.35, 0.35, 0.35, 0.1],
  boxFrame: [0.4, 0.4, 0.4, 0.05],
  cappedTorus: [0.4, 0.15, 120, 0], // R, r, aperture in degrees
  link: [0.25, 0.3, 0.1, 0], // half length, major R, minor r
  hexPrism: [0.4, 0.4, 0, 0],
  triPrism: [0.4, 0.4, 0, 0],
  roundedCylinder: [0.3, 0.1, 0.4, 0],
  roundCone: [0.35, 0.15, 0.7, 0], // bottom r, top r, height
  solidAngle: [45, 0.5, 0, 0], // aperture in degrees, radius
  cutSphere: [0.5, 0.2, 0, 0], // radius, cut height
  cutHollowSphere: [0.5, 0.2, 0.03, 0], // radius, cut height, thickness
  deathStar: [0.5, 0.35, 0.5, 0], // sphere R, carved R, offset
  rhombus: [0.5, 0.3, 0.1, 0.02], // half diag a, half diag b, half height, corner r
  octahedron: [0.5, 0, 0, 0],
  pyramid: [0.7, 0.7, 0, 0], // base width, height
  vesica: [0.5, 0.25, 0, 0], // radius, offset
};

const TYPE_LABEL: Record<ShapeType, string> = {
  sphere: "Sphere",
  box: "Box",
  torus: "Torus",
  cylinder: "Cylinder",
  capsule: "Capsule",
  cone: "Cone",
  roundedBox: "Rounded Box",
  boxFrame: "Box Frame",
  cappedTorus: "Capped Torus",
  link: "Link",
  hexPrism: "Hex Prism",
  triPrism: "Tri Prism",
  roundedCylinder: "Rounded Cylinder",
  roundCone: "Round Cone",
  solidAngle: "Solid Angle",
  cutSphere: "Cut Sphere",
  cutHollowSphere: "Cut Hollow Sphere",
  deathStar: "Death Star",
  rhombus: "Rhombus",
  octahedron: "Octahedron",
  pyramid: "Pyramid",
  vesica: "Vesica",
};

export const ALL_SHAPE_TYPES = Object.keys(TYPE_LABEL) as ShapeType[];

export function shapeLabel(type: ShapeType): string {
  return TYPE_LABEL[type];
}

export function zeroCounters(): Record<ShapeType, number> {
  return Object.fromEntries(ALL_SHAPE_TYPES.map((t) => [t, 0])) as Record<
    ShapeType,
    number
  >;
}

export function createDefaultRoot(): SceneRoot {
  return {
    id: crypto.randomUUID(),
    name: "Scene",
    items: [],
  };
}

export function countInstructions(items: SceneItem[]): number {
  let count = 0;
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (item.kind === "layer") {
      count += 1;
      if (i > 0) count += 1;
    } else {
      if (item.items.length > 0) count += 2 + countInstructions(item.items);
      if (i > 0) count += 1;
    }
  }
  return count;
}

export function wouldExceedCap(root: SceneRoot, extra: number): boolean {
  return countInstructions(root.items) + extra > MAX_INSTRUCTIONS;
}

function isObjectGroup(container: SceneContainer): container is ObjectGroup {
  return (container as ObjectGroup).kind === "group";
}

function extraInstructionsForNewShape(container: SceneContainer): number {
  if (isObjectGroup(container) && container.items.length === 0) {
    return 3;
  }
  return container.items.length === 0 ? 1 : 2;
}

function extraInstructionsForNewGroup(container: SceneContainer): number {
  return container.items.length === 0 ? 0 : 1;
}

function getContainerDepthInItems(
  items: SceneItem[],
  containerId: string,
  depth: number,
): number {
  for (const item of items) {
    if (item.kind === "group") {
      if (item.id === containerId) return depth + 1;
      const found = getContainerDepthInItems(item.items, containerId, depth + 1);
      if (found >= 0) return found;
    }
  }
  return -1;
}

export function getContainerDepth(root: SceneRoot, containerId: string): number {
  if (root.id === containerId) return 0;
  return getContainerDepthInItems(root.items, containerId, 0);
}

export function wouldExceedDepth(root: SceneRoot, containerId: string): boolean {
  const depth = getContainerDepth(root, containerId);
  if (depth < 0) return true;
  return depth + 1 > MAX_TRANSFORM_DEPTH;
}

export function maxGroupDepth(items: SceneItem[], currentDepth = 0): number {
  let max = currentDepth;
  for (const item of items) {
    if (item.kind === "group") {
      max = Math.max(max, maxGroupDepth(item.items, currentDepth + 1));
    }
  }
  return max;
}

export function findContainer(
  root: SceneRoot,
  containerId: string,
): SceneContainer | null {
  if (root.id === containerId) return root;
  for (const item of root.items) {
    if (item.kind === "group") {
      const found = findContainerInGroup(item, containerId);
      if (found) return found;
    }
  }
  return null;
}

function findContainerInGroup(
  group: ObjectGroup,
  containerId: string,
): SceneContainer | null {
  if (group.id === containerId) return group;
  for (const item of group.items) {
    if (item.kind === "group") {
      const found = findContainerInGroup(item, containerId);
      if (found) return found;
    }
  }
  return null;
}

/** Groups from outermost ancestor down to containerId (inclusive when container is a group). */
export function getAncestorGroups(root: SceneRoot, containerId: string): ObjectGroup[] {
  if (root.id === containerId) return [];

  function search(items: SceneItem[], path: ObjectGroup[]): ObjectGroup[] | null {
    for (const item of items) {
      if (item.kind !== "group") continue;
      const nextPath = [...path, item];
      if (item.id === containerId) return nextPath;
      const found = search(item.items, nextPath);
      if (found) return found;
    }
    return null;
  }

  return search(root.items, []) ?? [];
}

export function findItem(
  root: SceneRoot,
  itemId: string,
): { item: SceneItem; container: SceneContainer; index: number } | null {
  return findItemInItems(root.items, root, itemId);
}

function findItemInItems(
  items: SceneItem[],
  container: SceneContainer,
  itemId: string,
): { item: SceneItem; container: SceneContainer; index: number } | null {
  for (let i = 0; i < items.length; i++) {
    const item = items[i];
    if (item.id === itemId) {
      return { item, container, index: i };
    }
    if (item.kind === "group") {
      const found = findItemInItems(item.items, item, itemId);
      if (found) return found;
    }
  }
  return null;
}

function mapRootContainer(
  root: SceneRoot,
  containerId: string,
  mapItems: (items: SceneItem[]) => SceneItem[],
): SceneRoot {
  if (root.id === containerId) {
    return { ...root, items: mapItems(root.items) };
  }
  return {
    ...root,
    items: root.items.map((item) =>
      item.kind === "group" ? mapGroupContainer(item, containerId, mapItems) : item,
    ),
  };
}

function mapGroupContainer(
  group: ObjectGroup,
  containerId: string,
  mapItems: (items: SceneItem[]) => SceneItem[],
): ObjectGroup {
  if (group.id === containerId) {
    return { ...group, items: mapItems(group.items) };
  }
  return {
    ...group,
    items: group.items.map((item) =>
      item.kind === "group" ? mapGroupContainer(item, containerId, mapItems) : item,
    ),
  };
}

function mapRootItem(
  root: SceneRoot,
  itemId: string,
  mapItem: (item: SceneItem) => SceneItem,
): SceneRoot {
  return {
    ...root,
    items: root.items.map((item) => mapItemDeep(item, itemId, mapItem)),
  };
}

function mapItemDeep(
  item: SceneItem,
  itemId: string,
  mapItem: (item: SceneItem) => SceneItem,
): SceneItem {
  if (item.id === itemId) {
    return mapItem(item);
  }
  if (item.kind === "group") {
    return {
      ...item,
      items: item.items.map((child) => mapItemDeep(child, itemId, mapItem)),
    };
  }
  return item;
}

interface SceneState {
  root: SceneRoot;
  selectedContainerId: string;
  selectedItemId: string | null;
  /** True when Scene root is explicitly selected (not deselected). */
  rootSelected: boolean;
  counters: Record<ShapeType, number>;
  groupCounter: number;

  addShapeToContainer: (containerId: string, shapeType: ShapeType) => boolean;
  addGroupToContainer: (containerId: string) => boolean;
  removeItem: (containerId: string, itemId: string) => void;
  updateLayer: (
    containerId: string,
    layerId: string,
    patch: Partial<
      Pick<
        ShapeLayer,
        "name" | "position" | "rotation" | "scale" | "params" | "op" | "smoothK"
      >
    >,
  ) => void;
  moveItems: (dragIds: string[], parentId: string | null, index: number) => boolean;
  renameItem: (itemId: string, name: string) => void;
  updateGroup: (
    groupId: string,
    patch: Partial<
      Pick<ObjectGroup, "name" | "position" | "rotation" | "scale" | "op" | "smoothK">
    >,
  ) => void;
  updateRootName: (name: string) => void;
  selectRoot: () => void;
  deselect: () => void;
  selectItem: (containerId: string, itemId: string) => void;
}

const _saved = loadInitialState();
const _initialRoot = _saved.root ?? createDefaultRoot();

export const useSceneStore = create<SceneState>()(
  temporal(
    (set, get) => ({
      root: _initialRoot,
      selectedContainerId: _initialRoot.id,
      selectedItemId: null,
      rootSelected: false,
      // Merge: scenes saved before new shapes existed lack their counter keys.
      counters: { ...zeroCounters(), ..._saved.counters },
      groupCounter: _saved.groupCounter ?? 0,

      addShapeToContainer: (containerId, shapeType) => {
        const state = get();
        const container = findContainer(state.root, containerId);
        if (!container) return false;

        const extra = extraInstructionsForNewShape(container);
        if (wouldExceedCap(state.root, extra)) return false;

        const count = state.counters[shapeType] + 1;
        const newLayer: ShapeLayer = {
          kind: "layer",
          id: crypto.randomUUID(),
          shapeType,
          name: `${TYPE_LABEL[shapeType]} ${count}`,
          position: [0, 0, 0],
          rotation: [0, 0, 0],
          scale: [1, 1, 1],
          params: [...DEFAULT_PARAMS[shapeType]],
          op: "union",
          smoothK: 0.1,
        };

        set({
          root: mapRootContainer(state.root, containerId, (items) => [
            ...items,
            newLayer,
          ]),
          counters: { ...state.counters, [shapeType]: count },
          selectedContainerId: containerId,
          selectedItemId: newLayer.id,
          rootSelected: false,
        });
        return true;
      },

      addGroupToContainer: (containerId) => {
        const state = get();
        const container = findContainer(state.root, containerId);
        if (!container) return false;

        if (wouldExceedDepth(state.root, containerId)) return false;

        const extra = extraInstructionsForNewGroup(container);
        if (wouldExceedCap(state.root, extra)) return false;

        const count = state.groupCounter + 1;
        const newGroup: ObjectGroup = {
          kind: "group",
          id: crypto.randomUUID(),
          name: `Object ${count}`,
          position: [0, 0, 0],
          rotation: [0, 0, 0],
          scale: [1, 1, 1],
          op: "union",
          smoothK: 0.1,
          items: [],
        };

        set({
          root: mapRootContainer(state.root, containerId, (items) => [
            ...items,
            newGroup,
          ]),
          groupCounter: count,
          selectedContainerId: newGroup.id,
          selectedItemId: newGroup.id,
          rootSelected: false,
        });
        return true;
      },

      removeItem: (containerId, itemId) =>
        set((state) => {
          const container = findContainer(state.root, containerId);
          if (!container) return state;

          const newRoot = mapRootContainer(state.root, containerId, (items) =>
            items.filter((item) => item.id !== itemId),
          );
          const stillSelected =
            state.selectedItemId !== null &&
            findItem(newRoot, state.selectedItemId) !== null;

          return {
            root: newRoot,
            selectedItemId: stillSelected ? state.selectedItemId : null,
            selectedContainerId: stillSelected
              ? state.selectedContainerId
              : newRoot.id,
            rootSelected: stillSelected ? state.rootSelected : false,
          };
        }),

      // Values stored full-precision; rounding is display-only (NumericInput).
      updateLayer: (containerId, layerId, patch) =>
        set((state) => ({
          root: mapRootContainer(state.root, containerId, (items) =>
            items.map((item) =>
              item.kind === "layer" && item.id === layerId ? { ...item, ...patch } : item,
            ),
          ),
        })),

      updateGroup: (groupId, patch) =>
        set((state) => ({
          root: mapRootItem(state.root, groupId, (item) =>
            item.kind === "group" ? { ...item, ...patch } : item,
          ),
        })),

      updateRootName: (name) =>
        set((state) => ({
          root: { ...state.root, name },
        })),

      selectRoot: () =>
        set((state) => ({
          selectedContainerId: state.root.id,
          selectedItemId: null,
          rootSelected: true,
        })),

      deselect: () =>
        set((state) => ({
          selectedContainerId: state.root.id,
          selectedItemId: null,
          rootSelected: false,
        })),

      selectItem: (_containerId, itemId) =>
        set((state) => {
          const found = findItem(state.root, itemId);
          if (!found) return state;

          const selectedContainerId =
            found.item.kind === "group" ? found.item.id : found.container.id;

          return {
            selectedContainerId,
            selectedItemId: itemId,
            rootSelected: false,
          };
        }),

      moveItems: (dragIds, parentId, index) => {
        const state = get();
        const result = applyMoveToRoot(state.root, dragIds, parentId, index);
        if ("reason" in result) {
          if (result.reason === "cycle") {
            showToast("Cannot move into itself or its children");
          } else if (result.reason === "depth") {
            showToast("Nesting too deep (max 16 levels)");
          } else if (result.reason === "invalid_parent") {
            showToast("Invalid drop target");
          }
          return false;
        }

        const newRoot = result.root;
        let selectedContainerId = state.selectedContainerId;
        let selectedItemId = state.selectedItemId;

        if (state.selectedItemId && findItem(newRoot, state.selectedItemId)) {
          const found = findItem(newRoot, state.selectedItemId)!;
          selectedContainerId =
            found.item.kind === "group" ? found.item.id : found.container.id;
          selectedItemId = state.selectedItemId;
        } else if (dragIds.length > 0 && findItem(newRoot, dragIds[0])) {
          const found = findItem(newRoot, dragIds[0])!;
          selectedContainerId =
            found.item.kind === "group" ? found.item.id : found.container.id;
          selectedItemId = dragIds[0];
        }

        set({
          root: newRoot,
          selectedContainerId,
          selectedItemId,
          rootSelected: false,
        });
        return true;
      },

      renameItem: (itemId, name) => {
        const state = get();
        const found = findItem(state.root, itemId);
        if (!found) return;

        if (found.item.kind === "layer") {
          set({
            root: mapRootContainer(state.root, found.container.id, (items) =>
              items.map((item) =>
                item.kind === "layer" && item.id === itemId ? { ...item, name } : item,
              ),
            ),
          });
        } else {
          set({
            root: mapRootItem(state.root, itemId, (item) =>
              item.kind === "group" ? { ...item, name } : item,
            ),
          });
        }
      },
    }),
    {
      partialize: (state) => ({
        root: state.root,
        counters: state.counters,
        groupCounter: state.groupCounter,
      }),
      equality: shallow,
      limit: 100,
    },
  ),
);

export const temporalStore = useSceneStore.temporal;

if (_saved.root?.items?.length) {
  temporalStore.getState().clear();
}

export function undoScene() {
  temporalStore.getState().undo();
  const root = useSceneStore.getState().root;
  useSceneStore.setState({
    selectedContainerId: root.id,
    selectedItemId: null,
    rootSelected: false,
  });
}

export function redoScene() {
  temporalStore.getState().redo();
  const root = useSceneStore.getState().root;
  useSceneStore.setState({
    selectedContainerId: root.id,
    selectedItemId: null,
    rootSelected: false,
  });
}
