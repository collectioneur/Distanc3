import { Move, RotateCw, type LucideIcon } from "lucide-react";
import { useGizmoStore, type GizmoMode } from "../../store/gizmoStore";

const MODES: { value: GizmoMode; label: string; icon: LucideIcon }[] = [
  { value: "translate", label: "Move", icon: Move },
  { value: "rotate", label: "Rotate", icon: RotateCw },
];

export default function GizmoPanel() {
  const mode = useGizmoStore((s) => s.mode);
  const setMode = useGizmoStore((s) => s.setMode);

  return (
    <div className="panel panel-bottom-left">
      <span className="panel-label">Gizmo</span>
      <div className="top-panel-shapes">
        {MODES.map(({ value, label, icon: Icon }) => (
          <button
            key={value}
            className={`shape-btn${mode === value ? " shape-btn--active" : ""}`}
            onClick={() => setMode(value)}
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
