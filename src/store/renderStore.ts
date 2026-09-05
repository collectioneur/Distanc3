import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  DEFAULT_QUALITY,
  QUALITY_PRESET_COUNT,
} from "../utils/quality";

export type RenderMode = "classic" | "chrome";

interface RenderState {
  renderMode: RenderMode;
  quality: number;
  setRenderMode: (m: RenderMode) => void;
  setQuality: (q: number) => void;
}

export const useRenderStore = create<RenderState>()(
  persist(
    (set) => ({
      renderMode: "classic",
      quality: DEFAULT_QUALITY,
      setRenderMode: (m) => set({ renderMode: m }),
      setQuality: (q) =>
        set({
          quality: Math.max(0, Math.min(QUALITY_PRESET_COUNT - 1, Math.round(q))),
        }),
    }),
    { name: "d-stance_render_mode" },
  ),
);
