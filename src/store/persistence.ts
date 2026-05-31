import { useEffect, useRef } from "react";
import { type SceneObject, type ShapeType, useSceneStore } from "./sceneStore";
import { temporalStore } from "./sceneStore";

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

function loadFromStorage(): PersistedScene | null {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) {
      const fromLS = JSON.parse(raw) as PersistedScene;
      if (fromLS?.v === SCHEMA_VERSION) return fromLS;
    }
  } catch {
    // ignore
  }
  return null;
}

export function loadInitialState(): Partial<PersistedScene> {
  const hash = window.location.hash.slice(1);
  if (hash) {
    const fromUrl = decodeHash(hash);
    if (fromUrl) return fromUrl;
    // Corrupt hash — remove it so we don't confuse the user
    history.replaceState(null, "", window.location.pathname + window.location.search);
  }
  return loadFromStorage() ?? {};
}

function applyScene(scene: PersistedScene) {
  useSceneStore.setState({
    objects: scene.objects,
    counters: scene.counters,
    objectCounter: scene.objectCounter,
    selectedObjectId: null,
    selectedNodeId: null,
  });
  temporalStore.getState().clear();
}

export function usePersistence(): void {
  const lastWrittenHash = useRef<string>("");

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;

    // Write store → URL + localStorage (debounced)
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
        lastWrittenHash.current = encoded;
        location.replace("#" + encoded);
        try {
          localStorage.setItem(LS_KEY, JSON.stringify(persisted));
        } catch {
          // ignore quota errors
        }
      }, 1000);
    });

    // Read URL → store (external navigation: back/forward, pasted link)
    function onHashChange() {
      const hash = window.location.hash.slice(1);
      // Skip if we wrote this hash ourselves
      if (hash === lastWrittenHash.current) return;

      const fromUrl = decodeHash(hash);
      if (fromUrl) {
        lastWrittenHash.current = hash;
        try {
          localStorage.setItem(LS_KEY, JSON.stringify(fromUrl));
        } catch {
          // ignore quota errors
        }
        applyScene(fromUrl);
        return;
      }

      // Corrupt / missing hash — fall back to localStorage
      const fromLS = loadFromStorage();
      if (fromLS) {
        applyScene(fromLS);
      } else {
        useSceneStore.setState({
          objects: [],
          counters: { sphere: 0, box: 0, torus: 0, cylinder: 0, capsule: 0, cone: 0 },
          objectCounter: 0,
          selectedObjectId: null,
          selectedNodeId: null,
        });
        temporalStore.getState().clear();
      }
    }

    window.addEventListener("hashchange", onHashChange);

    return () => {
      unsub();
      clearTimeout(timer);
      window.removeEventListener("hashchange", onHashChange);
    };
  }, []);
}
