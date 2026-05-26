import Canvas from "./components/Canvas";
import TopPanel from "./components/panels/TopPanel";
import LeftPanel from "./components/panels/LeftPanel";
import RightPanel from "./components/panels/RightPanel";
import BottomPanel from "./components/panels/BottomPanel";
import CodeExportPanel from "./components/panels/CodeExportPanel";

export default function App() {
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
