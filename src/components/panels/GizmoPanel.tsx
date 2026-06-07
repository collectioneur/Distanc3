import { useGizmoStore, type GizmoMode } from "../../store/gizmoStore";

const MODES: { value: GizmoMode; label: string; icon: string }[] = [
  { value: "translate", label: "Move", icon: "↔" },
  { value: "rotate", label: "Rotate", icon: "↻" },
];

export default function GizmoPanel() {
  const mode = useGizmoStore((s) => s.mode);
  const setMode = useGizmoStore((s) => s.setMode);

  return (
    <div className="panel panel-bottom-left">
      <span className="panel-label">Gizmo</span>
      <div className="top-panel-shapes">
        {MODES.map((m) => (
          <button
            key={m.value}
            className={`shape-btn${mode === m.value ? " shape-btn--active" : ""}`}
            onClick={() => setMode(m.value)}
          >
            <span className="shape-btn-icon">{m.icon}</span>
            <span className="shape-btn-label">{m.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
