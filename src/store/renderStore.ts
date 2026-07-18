import { create } from "zustand";
import { persist } from "zustand/middleware";

export type RenderMode = "classic" | "chrome";

interface RenderState {
  renderMode: RenderMode;
  setRenderMode: (m: RenderMode) => void;
}

export const useRenderStore = create<RenderState>()(
  persist(
    (set) => ({
      renderMode: "classic",
      setRenderMode: (m) => set({ renderMode: m }),
    }),
    { name: "distanc3_render_mode" },
  ),
);
