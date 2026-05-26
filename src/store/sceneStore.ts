import { create } from "zustand";

export type ShapeType = "sphere" | "box" | "torus" | "cylinder" | "capsule" | "cone";

export interface ShapeInstance {
  id: string;
  type: ShapeType;
  name: string;
  position: [number, number, number];
  params: [number, number, number, number];
  // sphere:   [radius, 0, 0, 0]
  // box:      [halfX, halfY, halfZ, 0]
  // torus:    [majorR, minorR, 0, 0]
  // cylinder: [radius, halfHeight, 0, 0]
  // capsule:  [radius, halfHeight, 0, 0]
  // cone:     [radius, halfHeight, 0, 0]
}

interface SceneState {
  shapes: ShapeInstance[];
  selectedId: string | null;
  counters: Record<ShapeType, number>;
  addShape: (type: ShapeType) => void;
  removeShape: (id: string) => void;
  selectShape: (id: string | null) => void;
  updateShape: (
    id: string,
    patch: Partial<Pick<ShapeInstance, "position" | "params">>,
  ) => void;
}

export const MAX_SCENE_SHAPES = 8;

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

export const useSceneStore = create<SceneState>((set) => ({
  shapes: [],
  selectedId: null,
  counters: { sphere: 0, box: 0, torus: 0, cylinder: 0, capsule: 0, cone: 0 },

  addShape: (type) =>
    set((state) => {
      if (state.shapes.length >= MAX_SCENE_SHAPES) return state;
      const count = state.counters[type] + 1;
      const shape: ShapeInstance = {
        id: crypto.randomUUID(),
        type,
        name: `${TYPE_LABEL[type]} ${count}`,
        position: [0, 0, 0],
        params: [...DEFAULT_PARAMS[type]],
      };
      return {
        shapes: [...state.shapes, shape],
        counters: { ...state.counters, [type]: count },
        selectedId: shape.id,
      };
    }),

  removeShape: (id) =>
    set((state) => ({
      shapes: state.shapes.filter((s) => s.id !== id),
      selectedId: state.selectedId === id ? null : state.selectedId,
    })),

  selectShape: (id) => set({ selectedId: id }),

  updateShape: (id, patch) =>
    set((state) => ({
      shapes: state.shapes.map((s) => (s.id === id ? { ...s, ...patch } : s)),
    })),
}));
