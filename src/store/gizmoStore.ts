import { create } from "zustand";

export type GizmoMode = "translate" | "rotate" | "scale";

const LS_KEY = "distanc3-gizmo-mode";

function loadMode(): GizmoMode {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (raw === "translate" || raw === "rotate" || raw === "scale") return raw;
  } catch {
    /* ignore */
  }
  return "translate";
}

interface GizmoState {
  mode: GizmoMode;
  setMode: (mode: GizmoMode) => void;
}

export const useGizmoStore = create<GizmoState>((set) => ({
  mode: loadMode(),
  setMode: (mode) => {
    try {
      localStorage.setItem(LS_KEY, mode);
    } catch {
      /* ignore */
    }
    set({ mode });
  },
}));
