import { useEffect } from "react";
import Canvas from "./components/Canvas";
import TopPanel from "./components/panels/TopPanel";
import LeftPanel from "./components/panels/LeftPanel";
import RightPanel from "./components/panels/RightPanel";
import BottomPanel from "./components/panels/BottomPanel";
import CodeExportPanel from "./components/panels/CodeExportPanel";
import { undoScene, redoScene } from "./store/sceneStore";
import { usePersistence } from "./store/persistence";

export default function App() {
  usePersistence();

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
      <LeftPanel />
      <RightPanel />
      <CodeExportPanel />
      <BottomPanel />
    </>
  );
}
