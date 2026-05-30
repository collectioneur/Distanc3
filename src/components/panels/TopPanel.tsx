import { useState, useRef } from "react";
import { useStore } from "zustand";
import {
  useSceneStore,
  temporalStore,
  undoScene,
  redoScene,
  MAX_NODES_PER_OBJECT,
  countNodes,
  type ShapeType,
} from "../../store/sceneStore";

const SHAPES: { type: ShapeType; label: string; icon: string }[] = [
  { type: "box", label: "Box", icon: "□" },
  { type: "sphere", label: "Sphere", icon: "○" },
  { type: "torus", label: "Torus", icon: "◎" },
  { type: "cylinder", label: "Cylinder", icon: "⌀" },
  { type: "capsule", label: "Capsule", icon: "⬬" },
  { type: "cone", label: "Cone", icon: "△" },
];

export default function TopPanel() {
  const addShapeToObject = useSceneStore((s) => s.addShapeToObject);
  const selectedObjectId = useSceneStore((s) => s.selectedObjectId);
  const objects = useSceneStore((s) => s.objects);
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

  const selectedObject = objects.find((o) => o.id === selectedObjectId) ?? null;
  const nodeCount = selectedObject?.root ? countNodes(selectedObject.root) : 0;
  // Adding a shape creates 1 leaf + 1 op node (unless the object is empty)
  const wouldAdd = selectedObject?.root ? 2 : 1;
  const isFull = nodeCount + wouldAdd > MAX_NODES_PER_OBJECT;
  const noObject = !selectedObjectId;

  function getTitle(label: string): string {
    if (noObject) return "Select or create an object first";
    if (isFull) return `Object is full (max ${MAX_NODES_PER_OBJECT} nodes)`;
    return `Add ${label}`;
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
          ↩
        </button>
        <button
          className="history-btn"
          onClick={redoScene}
          disabled={!canRedo}
          title="Redo (Cmd+Shift+Z)"
        >
          ↪
        </button>
        <button
          className="history-btn"
          onClick={handleCopyLink}
          title="Copy share link"
        >
          {copied ? "✓" : "⎘"}
        </button>
      </div>
      <div className="top-panel-divider" />
      <span className="panel-label">Shapes</span>
      <div className="top-panel-shapes">
        {SHAPES.map(({ type, label, icon }) => (
          <button
            key={type}
            className="shape-btn"
            onClick={() => {
              if (selectedObjectId) addShapeToObject(selectedObjectId, type);
            }}
            disabled={noObject || isFull}
            title={getTitle(label)}
          >
            <span className="shape-btn-icon">{icon}</span>
            <span className="shape-btn-label">{label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
