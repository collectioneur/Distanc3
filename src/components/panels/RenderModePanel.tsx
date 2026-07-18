import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown, Gem, Sun, type LucideIcon } from "lucide-react";
import { useRenderStore, type RenderMode } from "../../store/renderStore";

const MODES: { value: RenderMode; label: string; icon: LucideIcon }[] = [
  { value: "classic", label: "Classic", icon: Sun },
  { value: "chrome", label: "Chrome", icon: Gem },
];

export default function RenderModePanel() {
  const renderMode = useRenderStore((s) => s.renderMode);
  const setRenderMode = useRenderStore((s) => s.setRenderMode);
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: PointerEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const active = MODES.find((m) => m.value === renderMode) ?? MODES[1];
  const ActiveIcon = active.icon;

  return (
    <div className="panel panel-render-mode" ref={rootRef}>
      <span className="panel-label">Render mode</span>
      <button
        className="shape-btn render-mode-trigger"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        title="Render mode"
      >
        <span className="shape-btn-icon">
          <ActiveIcon size={14} />
        </span>
        <span className="shape-btn-label">{active.label}</span>
        <ChevronDown
          size={13}
          className={`render-mode-chevron${open ? " render-mode-chevron--open" : ""}`}
        />
      </button>
      {open && (
        <div className="render-mode-list" role="listbox">
          {MODES.map(({ value, label, icon: Icon }) => (
            <button
              key={value}
              className={`render-mode-option${value === renderMode ? " render-mode-option--active" : ""}`}
              role="option"
              aria-selected={value === renderMode}
              onClick={() => {
                setRenderMode(value);
                setOpen(false);
              }}
            >
              <span className="shape-btn-icon">
                <Icon size={14} />
              </span>
              <span className="shape-btn-label">{label}</span>
              {value === renderMode && (
                <Check size={13} className="render-mode-check" />
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
