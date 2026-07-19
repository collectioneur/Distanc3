import { Move, RotateCw, Scaling, type LucideIcon } from "lucide-react";
import { useGizmoStore, type GizmoMode } from "../../store/gizmoStore";
import { useSceneStore } from "../../store/sceneStore";

const MODES: { value: GizmoMode; label: string; icon: LucideIcon; hotkey: string }[] = [
  { value: "translate", label: "Move", icon: Move, hotkey: "W" },
  { value: "rotate", label: "Rotate", icon: RotateCw, hotkey: "E" },
  { value: "scale", label: "Scale", icon: Scaling, hotkey: "R" },
];

export default function GizmoPanel() {
  const mode = useGizmoStore((s) => s.mode);
  const setMode = useGizmoStore((s) => s.setMode);
  const hasSelection = useSceneStore((s) => s.selectedItemId !== null);

  if (!hasSelection) return null;

  const activeIndex = MODES.findIndex((m) => m.value === mode);

  return (
    <div
      className="segmented segmented--floating"
      role="radiogroup"
      aria-label="Gizmo mode"
    >
      <span
        className="segmented-indicator"
        style={{ transform: `translateX(${activeIndex * 100}%)` }}
      />
      {MODES.map(({ value, label, icon: Icon, hotkey }) => (
        <button
          key={value}
          type="button"
          role="radio"
          aria-checked={mode === value}
          className={`segmented-btn${mode === value ? " segmented-btn--active" : ""}`}
          title={`${label} (${hotkey})`}
          onClick={() => setMode(value)}
        >
          <span className="shape-btn-icon">
            <Icon size={14} />
          </span>
          <span className="shape-btn-label">{label}</span>
          <span className="shape-btn-key">{hotkey}</span>
        </button>
      ))}
    </div>
  );
}
