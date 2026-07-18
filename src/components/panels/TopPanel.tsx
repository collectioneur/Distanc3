import { useState, useRef } from "react";
import { useStore } from "zustand";
import { Check, Link2, Plus, Redo2, Undo2 } from "lucide-react";
import { temporalStore, undoScene, redoScene } from "../../store/sceneStore";
import { SHAPES, addShape } from "../../utils/commands";
import { openPalette } from "../CommandPalette";

const isMac = navigator.platform.startsWith("Mac");
const paletteKey = isMac ? "⌘K" : "Ctrl+K";

export default function TopPanel() {
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
      <div className="top-panel-shapes">
        {SHAPES.map(({ type, label, icon: Icon }, i) => (
          <button
            key={type}
            className="shape-btn shape-btn--icon"
            onClick={() => addShape(type)}
            title={`Add ${label} (${i + 1})`}
          >
            <Icon size={15} />
            <span className="shape-btn-key">{i + 1}</span>
          </button>
        ))}
        <button
          className="shape-btn shape-btn--icon"
          onClick={openPalette}
          title={`All commands (${paletteKey})`}
        >
          <Plus size={15} />
        </button>
      </div>
    </div>
  );
}
