import { useRenderStore, type RenderMode } from "../../store/renderStore";

const MODES: { value: RenderMode; label: string; icon: string }[] = [
  { value: 0, label: "Lit", icon: "💡" },
  { value: 1, label: "Depth", icon: "📏" },
  { value: 2, label: "Steps", icon: "🔢" },
];

export default function BottomPanel() {
  const renderMode = useRenderStore((s) => s.renderMode);
  const setRenderMode = useRenderStore((s) => s.setRenderMode);

  return (
    <div className="panel panel-bottom">
      <span className="panel-label">Render</span>
      <div className="top-panel-shapes">
        {MODES.map((m) => (
          <button
            key={m.value}
            className={`shape-btn${renderMode === m.value ? " shape-btn--active" : ""}`}
            onClick={() => setRenderMode(m.value)}
          >
            <span className="shape-btn-icon">{m.icon}</span>
            <span className="shape-btn-label">{m.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
