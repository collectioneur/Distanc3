import { Hash, Ruler, Sun, type LucideIcon } from "lucide-react";
import { useRenderStore, type RenderMode } from "../../store/renderStore";

const MODES: { value: RenderMode; label: string; icon: LucideIcon }[] = [
  { value: 0, label: "Lit", icon: Sun },
  { value: 1, label: "Depth", icon: Ruler },
  { value: 2, label: "Steps", icon: Hash },
];

export default function BottomPanel() {
  const renderMode = useRenderStore((s) => s.renderMode);
  const setRenderMode = useRenderStore((s) => s.setRenderMode);

  return (
    <div className="panel panel-bottom">
      <span className="panel-label">Render</span>
      <div className="top-panel-shapes">
        {MODES.map(({ value, label, icon: Icon }) => (
          <button
            key={value}
            className={`shape-btn${renderMode === value ? " shape-btn--active" : ""}`}
            onClick={() => setRenderMode(value)}
          >
            <span className="shape-btn-icon">
              <Icon size={14} />
            </span>
            <span className="shape-btn-label">{label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
