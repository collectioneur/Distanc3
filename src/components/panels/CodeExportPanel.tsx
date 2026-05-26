import { useMemo, useState, useEffect } from "react";
import { useSceneStore } from "../../store/sceneStore";
import { generateSdScene } from "../../utils/generateSdScene";

type Tab = "typegpu" | "wgsl";

export default function CodeExportPanel() {
  const shapes = useSceneStore((s) => s.shapes);
  const [activeTab, setActiveTab] = useState<Tab>("typegpu");
  const [copied, setCopied] = useState(false);

  const code = useMemo(() => generateSdScene(shapes), [shapes]);
  const displayCode = activeTab === "typegpu" ? code.typegpu : code.wgsl;

  useEffect(() => {
    if (!copied) return;
    const id = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(id);
  }, [copied]);

  function handleCopy() {
    navigator.clipboard.writeText(displayCode).then(() => setCopied(true));
  }

  return (
    <div className="panel panel-code-export">
      <div className="code-header">
        <div className="code-tabs">
          <button
            className={`code-tab${activeTab === "typegpu" ? " code-tab--active" : ""}`}
            onClick={() => setActiveTab("typegpu")}
          >
            TypeGPU
          </button>
          <button
            className={`code-tab${activeTab === "wgsl" ? " code-tab--active" : ""}`}
            onClick={() => setActiveTab("wgsl")}
          >
            WGSL
          </button>
        </div>
        <button className="code-copy-btn" onClick={handleCopy}>
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
      <pre className="code-pre">{displayCode}</pre>
    </div>
  );
}
