import {
  Box,
  Boxes,
  Circle,
  Cone,
  Cylinder,
  Pill,
  Torus,
  type LucideIcon,
} from "lucide-react";
import {
  useSceneStore,
  wouldExceedDepth,
  type ShapeType,
} from "../store/sceneStore";
import { showToast } from "./toast";

export const SHAPES: { type: ShapeType; label: string; icon: LucideIcon }[] = [
  { type: "box", label: "Box", icon: Box },
  { type: "sphere", label: "Sphere", icon: Circle },
  { type: "torus", label: "Torus", icon: Torus },
  { type: "cylinder", label: "Cylinder", icon: Cylinder },
  { type: "capsule", label: "Capsule", icon: Pill },
  { type: "cone", label: "Cone", icon: Cone },
];

export function addShape(type: ShapeType) {
  const { addShapeToContainer, selectedContainerId } = useSceneStore.getState();
  if (!addShapeToContainer(selectedContainerId, type)) {
    showToast("Scene too complex (max 256 operations)");
  }
}

export function addGroup() {
  const state = useSceneStore.getState();
  if (!state.addGroupToContainer(state.selectedContainerId)) {
    showToast(
      wouldExceedDepth(state.root, state.selectedContainerId)
        ? "Nesting too deep (max 16 levels)"
        : "Scene too complex (max 256 operations)",
    );
  }
}

export type Command = {
  id: string;
  label: string;
  icon: LucideIcon;
  hotkey?: string;
  keywords?: string[];
  run: () => void;
};

export const COMMANDS: Command[] = [
  ...SHAPES.map(({ type, label, icon }, i) => ({
    id: `add-${type}`,
    label: `Add ${label}`,
    icon,
    hotkey: String(i + 1),
    run: () => addShape(type),
  })),
  {
    id: "add-group",
    label: "Add Object (group)",
    icon: Boxes,
    keywords: ["group", "container", "folder", "nest"],
    run: addGroup,
  },
];
