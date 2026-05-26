import { create } from "zustand";

export type RenderMode = 0 | 1 | 2;

interface RenderState {
  renderMode: RenderMode;
  setRenderMode: (m: RenderMode) => void;
}

export const useRenderStore = create<RenderState>((set) => ({
  renderMode: 0,
  setRenderMode: (m) => set({ renderMode: m }),
}));
