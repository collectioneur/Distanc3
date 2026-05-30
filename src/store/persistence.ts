import { useEffect } from "react";
import { type SceneObject, type ShapeType, useSceneStore } from "./sceneStore";

const SCHEMA_VERSION = 1;
const LS_KEY = "distanc3_scene";

export type PersistedScene = {
  v: number;
  objects: SceneObject[];
  counters: Record<ShapeType, number>;
  objectCounter: number;
};

export function encodeHash(s: PersistedScene): string {
  const json = JSON.stringify(s);
  const bytes = new TextEncoder().encode(json);
  let binary = "";
  bytes.forEach((b) => (binary += String.fromCharCode(b)));
  return btoa(binary);
}

function decodeHash(raw: string): PersistedScene | null {
  try {
    const binary = atob(raw);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const parsed = JSON.parse(new TextDecoder().decode(bytes));
    if (parsed?.v !== SCHEMA_VERSION) return null;
    return parsed as PersistedScene;
  } catch {
    return null;
  }
}

export function loadInitialState(): Partial<PersistedScene> {
  const hash = window.location.hash.slice(1);
  if (hash) {
    const fromUrl = decodeHash(hash);
    if (fromUrl) return fromUrl;
    // Corrupt hash — remove it so we don't confuse the user
    history.replaceState(null, "", window.location.pathname + window.location.search);
  }
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) {
      const fromLS = JSON.parse(raw) as PersistedScene;
      if (fromLS?.v === SCHEMA_VERSION) return fromLS;
    }
  } catch {
    // ignore
  }
  return {};
}

export function usePersistence(): void {
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;

    const unsub = useSceneStore.subscribe((state) => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        const persisted: PersistedScene = {
          v: SCHEMA_VERSION,
          objects: state.objects,
          counters: state.counters,
          objectCounter: state.objectCounter,
        };
        const encoded = encodeHash(persisted);
        location.replace("#" + encoded);
        try {
          localStorage.setItem(LS_KEY, JSON.stringify(persisted));
        } catch {
          // ignore quota errors
        }
      }, 1000);
    });

    return () => {
      unsub();
      clearTimeout(timer);
    };
  }, []);
}
