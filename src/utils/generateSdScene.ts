import type { SceneRoot } from "../store/sceneStore";

export interface GeneratedCode {
  typegpu: string;
  wgsl: string;
}

const PLACEHOLDER = "// Code export temporarily unavailable";

export function generateSdScene(_root: SceneRoot): GeneratedCode {
  return { typegpu: PLACEHOLDER, wgsl: PLACEHOLDER };
}
