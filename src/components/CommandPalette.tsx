import { useEffect, useState } from "react";
import { Command } from "cmdk";
import { COMMANDS, SHAPES, addShape } from "../utils/commands";

let externalSetOpen: ((open: boolean) => void) | null = null;

/** Open the palette from anywhere (e.g. the «+» button in TopPanel). */
export function openPalette() {
  externalSetOpen?.(true);
}

const DIGIT_CODES = ["Digit1", "Digit2", "Digit3", "Digit4", "Digit5", "Digit6"];

export default function CommandPalette() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    externalSetOpen = setOpen;
    return () => {
      externalSetOpen = null;
    };
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && !e.altKey && e.code === "KeyK") {
        e.preventDefault();
        setOpen((o) => !o);
        return;
      }

      const tag = (document.activeElement as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      if (e.shiftKey && !e.metaKey && !e.ctrlKey && !e.altKey && e.code === "KeyA") {
        e.preventDefault();
        setOpen(true);
        return;
      }

      if (!e.metaKey && !e.ctrlKey && !e.altKey && !e.shiftKey) {
        const idx = DIGIT_CODES.indexOf(e.code);
        if (idx >= 0 && idx < SHAPES.length) {
          e.preventDefault();
          addShape(SHAPES[idx].type);
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  function runCommand(run: () => void) {
    setOpen(false);
    run();
  }

  return (
    <Command.Dialog open={open} onOpenChange={setOpen} label="Command palette">
      <Command.Input placeholder="Search commands…" />
      <Command.List>
        <Command.Empty>No results.</Command.Empty>
        {COMMANDS.map(({ id, label, icon: Icon, hotkey, keywords, run }) => (
          <Command.Item key={id} keywords={keywords} onSelect={() => runCommand(run)}>
            <Icon size={14} />
            <span className="cmdk-item-label">{label}</span>
            {hotkey && <kbd>{hotkey}</kbd>}
          </Command.Item>
        ))}
      </Command.List>
    </Command.Dialog>
  );
}
