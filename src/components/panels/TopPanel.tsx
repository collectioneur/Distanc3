import { useState, useRef } from "react";
import { useStore } from "zustand";
import {
  Box,
  Check,
  Circle,
  Cone,
  Cylinder,
  Link2,
  Pill,
  Redo2,
  Torus,
  Undo2,
  type LucideIcon,
} from "lucide-react";
import {
  useSceneStore,
  temporalStore,
  undoScene,
  redoScene,
  type ShapeType,
} from "../../store/sceneStore";
import { showToast } from "../../utils/toast";

const SHAPES: { type: ShapeType; label: string; icon: LucideIcon }[] = [
  { type: "box", label: "Box", icon: Box },
  { type: "sphere", label: "Sphere", icon: Circle },
  { type: "torus", label: "Torus", icon: Torus },
  { type: "cylinder", label: "Cylinder", icon: Cylinder },
  { type: "capsule", label: "Capsule", icon: Pill },
  { type: "cone", label: "Cone", icon: Cone },
];

export default function TopPanel() {
  const addShapeToContainer = useSceneStore((s) => s.addShapeToContainer);
  const selectedContainerId = useSceneStore((s) => s.selectedContainerId);
  const canUndo = useStore(temporalStore, (s) => s.pastStates.length > 0);
  const canRedo = useStore(temporalStore, (s) => s.futureStates.length > 0);

  const [copied, setCopied] = useState(false);
  const copyTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  function handleCopyLink() {
    navigator.clipboard.writeText(window.location.href).then(() => {
      setCopied(true);
      if (copyTimer.current) clearTimeout(copyTimer.current);
      copyTimer.current = setTimeout(() => setCopied(false), 1500);
    });
  }

  function handleAddShape(type: ShapeType) {
    const ok = addShapeToContainer(selectedContainerId, type);
    if (!ok) {
      showToast("Scene too complex (max 256 operations)");
    }
  }

  return (
    <div className="panel panel-top">
      <div className="top-panel-history">
        <button
          className="history-btn"
          onClick={undoScene}
          disabled={!canUndo}
          title="Undo (Cmd+Z)"
        >
          <Undo2 size={15} />
        </button>
        <button
          className="history-btn"
          onClick={redoScene}
          disabled={!canRedo}
          title="Redo (Cmd+Shift+Z)"
        >
          <Redo2 size={15} />
        </button>
        <button
          className="history-btn"
          onClick={handleCopyLink}
          title="Copy share link"
        >
          {copied ? <Check size={15} /> : <Link2 size={15} />}
        </button>
      </div>
      <div className="top-panel-divider" />
      <span className="panel-label">Shapes</span>
      <div className="top-panel-shapes">
        {SHAPES.map(({ type, label, icon: Icon }) => (
          <button
            key={type}
            className="shape-btn"
            onClick={() => handleAddShape(type)}
            title={`Add ${label}`}
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
