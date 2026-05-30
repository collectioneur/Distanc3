import { create } from "zustand";
import { shallow } from "zustand/shallow";
import { temporal } from "zundo";

export type ShapeType = "sphere" | "box" | "torus" | "cylinder" | "capsule" | "cone";
export type OpType = "union" | "subtract" | "intersect" | "sUnion" | "sSubtract" | "sIntersect";

export type ShapeNode = {
  kind: "shape";
  id: string;
  shapeType: ShapeType;
  name: string;
  position: [number, number, number];
  rotation: [number, number, number]; // Euler XYZ in degrees
  params: [number, number, number, number];
  // sphere:   [radius, 0, 0, 0]
  // box:      [halfX, halfY, halfZ, 0]
  // torus:    [majorR, minorR, 0, 0]
  // cylinder: [radius, halfHeight, 0, 0]
  // capsule:  [radius, halfHeight, 0, 0]
  // cone:     [radius, halfHeight, 0, 0]
};

export type OpNode = {
  kind: "op";
  id: string;
  op: OpType;
  smoothK: number;
  left: CsgNode;  // left subtree (base operand)
  right: CsgNode; // right subtree (modifier operand)
};

export type CsgNode = ShapeNode | OpNode;

export type SceneObject = {
  id: string;
  name: string;
  root: CsgNode | null;
};

export const MAX_OBJECTS = 8;
export const MAX_NODES_PER_OBJECT = 15;

interface SceneState {
  objects: SceneObject[];
  selectedObjectId: string | null;
  selectedNodeId: string | null;
  counters: Record<ShapeType, number>;
  objectCounter: number;

  addObject: () => void;
  removeObject: (objectId: string) => void;
  addShapeToObject: (objectId: string, shapeType: ShapeType) => void;
  removeNode: (objectId: string, nodeId: string) => void;
  updateShapeNode: (
    objectId: string,
    nodeId: string,
    patch: Partial<Pick<ShapeNode, "position" | "rotation" | "params">>,
  ) => void;
  updateOpNode: (
    objectId: string,
    nodeId: string,
    patch: Partial<Pick<OpNode, "op" | "smoothK">>,
  ) => void;
  updateObjectName: (objectId: string, name: string) => void;
  selectObject: (objectId: string | null) => void;
  selectNode: (nodeId: string | null) => void;
  selectNodeInObject: (objectId: string, nodeId: string) => void;
}

const DEFAULT_PARAMS: Record<ShapeType, [number, number, number, number]> = {
  sphere: [0.5, 0, 0, 0],
  box: [0.4, 0.4, 0.4, 0],
  torus: [0.4, 0.15, 0, 0],
  cylinder: [0.3, 0.5, 0, 0],
  capsule: [0.25, 0.4, 0, 0],
  cone: [0.4, 0.6, 0, 0],
};

const TYPE_LABEL: Record<ShapeType, string> = {
  sphere: "Sphere",
  box: "Box",
  torus: "Torus",
  cylinder: "Cylinder",
  capsule: "Capsule",
  cone: "Cone",
};

// ── Tree traversal utilities ───────────────────────────────────────────────────

export function countNodes(node: CsgNode): number {
  if (node.kind === "shape") return 1;
  return 1 + countNodes(node.left) + countNodes(node.right);
}

export function findNodeInTree(
  root: CsgNode | null,
  targetId: string | null,
): CsgNode | null {
  if (!root || !targetId) return null;
  if (root.id === targetId) return root;
  if (root.kind === "shape") return null;
  return findNodeInTree(root.left, targetId) ?? findNodeInTree(root.right, targetId);
}

function removeNodeFromTree(root: CsgNode, targetId: string): CsgNode | null {
  if (root.id === targetId) return null;
  if (root.kind === "shape") return root;

  // If a direct child matches, replace this op with the sibling
  if (root.left.id === targetId) return root.right;
  if (root.right.id === targetId) return root.left;

  const newLeft = removeNodeFromTree(root.left, targetId);
  const newRight = removeNodeFromTree(root.right, targetId);

  if (newLeft === null) return newRight;
  if (newRight === null) return newLeft;
  if (newLeft === root.left && newRight === root.right) return root;

  return { ...root, left: newLeft, right: newRight };
}

function updateNodeInTree(
  root: CsgNode,
  targetId: string,
  updater: (node: CsgNode) => CsgNode,
): CsgNode {
  if (root.id === targetId) return updater(root);
  if (root.kind === "shape") return root;

  const newLeft = updateNodeInTree(root.left, targetId, updater);
  const newRight = updateNodeInTree(root.right, targetId, updater);

  if (newLeft === root.left && newRight === root.right) return root;
  return { ...root, left: newLeft, right: newRight };
}

// ── Store ──────────────────────────────────────────────────────────────────────

export const useSceneStore = create<SceneState>()(
  temporal(
    (set) => ({
  objects: [],
  selectedObjectId: null,
  selectedNodeId: null,
  counters: { sphere: 0, box: 0, torus: 0, cylinder: 0, capsule: 0, cone: 0 },
  objectCounter: 0,

  addObject: () =>
    set((state) => {
      if (state.objects.length >= MAX_OBJECTS) return state;
      const count = state.objectCounter + 1;
      const obj: SceneObject = {
        id: crypto.randomUUID(),
        name: `Object ${count}`,
        root: null,
      };
      return {
        objects: [...state.objects, obj],
        objectCounter: count,
        selectedObjectId: obj.id,
        selectedNodeId: null,
      };
    }),

  removeObject: (objectId) =>
    set((state) => ({
      objects: state.objects.filter((o) => o.id !== objectId),
      selectedObjectId: state.selectedObjectId === objectId ? null : state.selectedObjectId,
      selectedNodeId: state.selectedObjectId === objectId ? null : state.selectedNodeId,
    })),

  addShapeToObject: (objectId, shapeType) =>
    set((state) => {
      const obj = state.objects.find((o) => o.id === objectId);
      if (!obj) return state;
      const currentNodes = obj.root ? countNodes(obj.root) : 0;
      // A new shape adds 1 leaf + 1 op node (except when tree is empty)
      const wouldAdd = obj.root ? 2 : 1;
      if (currentNodes + wouldAdd > MAX_NODES_PER_OBJECT) return state;

      const count = state.counters[shapeType] + 1;
      const newShape: ShapeNode = {
        kind: "shape",
        id: crypto.randomUUID(),
        shapeType,
        name: `${TYPE_LABEL[shapeType]} ${count}`,
        position: [0, 0, 0],
        rotation: [0, 0, 0],
        params: [...DEFAULT_PARAMS[shapeType]],
      };

      const newRoot: CsgNode = obj.root
        ? {
            kind: "op",
            id: crypto.randomUUID(),
            op: "union",
            smoothK: 0.1,
            left: obj.root,  // existing tree on the left (base)
            right: newShape, // new shape on the right (modifier)
          }
        : newShape;

      return {
        objects: state.objects.map((o) =>
          o.id === objectId ? { ...o, root: newRoot } : o,
        ),
        counters: { ...state.counters, [shapeType]: count },
        selectedObjectId: objectId,
        selectedNodeId: newShape.id,
      };
    }),

  removeNode: (objectId, nodeId) =>
    set((state) => {
      const obj = state.objects.find((o) => o.id === objectId);
      if (!obj || !obj.root) return state;
      const newRoot = removeNodeFromTree(obj.root, nodeId);
      return {
        objects: state.objects.map((o) =>
          o.id === objectId ? { ...o, root: newRoot } : o,
        ),
        selectedNodeId: state.selectedNodeId === nodeId ? null : state.selectedNodeId,
      };
    }),

  updateShapeNode: (objectId, nodeId, patch) =>
    set((state) => {
      const obj = state.objects.find((o) => o.id === objectId);
      if (!obj || !obj.root) return state;
      const newRoot = updateNodeInTree(obj.root, nodeId, (node) => {
        if (node.kind !== "shape") return node;
        return { ...node, ...patch };
      });
      return {
        objects: state.objects.map((o) =>
          o.id === objectId ? { ...o, root: newRoot } : o,
        ),
      };
    }),

  updateOpNode: (objectId, nodeId, patch) =>
    set((state) => {
      const obj = state.objects.find((o) => o.id === objectId);
      if (!obj || !obj.root) return state;
      const newRoot = updateNodeInTree(obj.root, nodeId, (node) => {
        if (node.kind !== "op") return node;
        return { ...node, ...patch };
      });
      return {
        objects: state.objects.map((o) =>
          o.id === objectId ? { ...o, root: newRoot } : o,
        ),
      };
    }),

  updateObjectName: (objectId, name) =>
    set((state) => ({
      objects: state.objects.map((o) =>
        o.id === objectId ? { ...o, name } : o,
      ),
    })),

  selectObject: (objectId) =>
    set({ selectedObjectId: objectId, selectedNodeId: null }),

  selectNode: (nodeId) => set({ selectedNodeId: nodeId }),

  selectNodeInObject: (objectId, nodeId) =>
    set({ selectedObjectId: objectId, selectedNodeId: nodeId }),
    }),
    {
      partialize: (state) => ({
        objects: state.objects,
        counters: state.counters,
        objectCounter: state.objectCounter,
      }),
      equality: shallow,
      limit: 100,
    },
  ),
);

export const temporalStore = useSceneStore.temporal;

export function undoScene() {
  temporalStore.getState().undo();
  useSceneStore.setState({ selectedObjectId: null, selectedNodeId: null });
}

export function redoScene() {
  temporalStore.getState().redo();
  useSceneStore.setState({ selectedObjectId: null, selectedNodeId: null });
}
