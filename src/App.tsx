import { useEffect, useState } from "react";
import Canvas from "./components/Canvas";
import TopPanel from "./components/panels/TopPanel";
import LeftPanel from "./components/panels/LeftPanel";
import RightPanel from "./components/panels/RightPanel";
import GizmoPanel from "./components/panels/GizmoPanel";
import CommandPalette from "./components/CommandPalette";
import RenderModePanel from "./components/panels/RenderModePanel";
import { undoScene, redoScene } from "./store/sceneStore";
import { useGizmoStore } from "./store/gizmoStore";
import { usePersistence } from "./store/persistence";
import { setToastListener } from "./utils/toast";

export default function App() {
  usePersistence();
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    setToastListener((message) => {
      setToast(message);
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => setToast(null), 2500);
    });
    return () => {
      setToastListener(null);
      if (timer) clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (document.activeElement as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.code === "KeyZ") {
        e.preventDefault();
        undoScene();
      }
      if ((e.metaKey || e.ctrlKey) && (e.code === "KeyY" || (e.shiftKey && e.code === "KeyZ"))) {
        e.preventDefault();
        redoScene();
      }

      if (!e.metaKey && !e.ctrlKey && !e.altKey && !e.shiftKey) {
        const setMode = useGizmoStore.getState().setMode;
        if (e.code === "KeyW") {
          e.preventDefault();
          setMode("translate");
        } else if (e.code === "KeyE") {
          e.preventDefault();
          setMode("rotate");
        } else if (e.code === "KeyR") {
          e.preventDefault();
          setMode("scale");
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  if (!navigator.gpu) {
    return (
      <div className="no-webgpu">
        WebGPU is not supported in this browser.
        <br />
        <small>Try Chrome 113+ or Edge 113+</small>
      </div>
    );
  }

  return (
    <>
      <Canvas />
      <TopPanel />
      <RenderModePanel />
      <LeftPanel />
      <RightPanel />
      <GizmoPanel />
      <CommandPalette />
      {toast && (
        <div
          style={{
            position: "fixed",
            bottom: 24,
            left: "50%",
            transform: "translateX(-50%)",
            background: "rgba(20, 20, 20, 0.92)",
            color: "#f0f0f0",
            padding: "10px 16px",
            borderRadius: 8,
            fontSize: 13,
            zIndex: 1000,
            pointerEvents: "none",
            border: "1px solid rgba(255,255,255,0.12)",
          }}
        >
          {toast}
        </div>
      )}
    </>
  );
}
