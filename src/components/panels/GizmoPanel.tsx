import { Move, RotateCw, Scaling, type LucideIcon } from "lucide-react";
import { useGizmoStore, type GizmoMode } from "../../store/gizmoStore";
import { useSceneStore } from "../../store/sceneStore";

const MODES: { value: GizmoMode; label: string; icon: LucideIcon }[] = [
  { value: "translate", label: "Move", icon: Move },
  { value: "rotate", label: "Rotate", icon: RotateCw },
  { value: "scale", label: "Scale", icon: Scaling },
];

export default function GizmoPanel() {
  const mode = useGizmoStore((s) => s.mode);
  const setMode = useGizmoStore((s) => s.setMode);
  const hasSelection = useSceneStore((s) => s.selectedItemId !== null);

  if (!hasSelection) return null;

  return (
    <div className="panel panel-bottom-center">
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
