import {
  Box,
  Boxes,
  Circle,
  Cone,
  Cylinder,
  Diamond,
  Egg,
  Frame,
  Gem,
  Globe,
  Hexagon,
  IceCreamCone,
  Link,
  Moon,
  Pill,
  Pizza,
  Pyramid,
  Rainbow,
  Salad,
  Squircle,
  Torus,
  Triangle,
  type LucideIcon,
} from "lucide-react";
import {
  useSceneStore,
  wouldExceedDepth,
  shapeLabel,
  type ShapeType,
} from "../store/sceneStore";
import { showToast } from "./toast";

export const SHAPE_ICONS: Record<ShapeType, LucideIcon> = {
  sphere: Circle,
  box: Box,
  torus: Torus,
  cylinder: Cylinder,
  capsule: Pill,
  cone: Cone,
  roundedBox: Squircle,
  boxFrame: Frame,
  cappedTorus: Rainbow,
  link: Link,
  hexPrism: Hexagon,
  triPrism: Triangle,
  roundedCylinder: Cylinder,
  roundCone: IceCreamCone,
  solidAngle: Pizza,
  cutSphere: Moon,
  cutHollowSphere: Salad,
  deathStar: Globe,
  rhombus: Diamond,
  octahedron: Gem,
  pyramid: Pyramid,
  vesica: Egg,
};

function shapeEntry(type: ShapeType) {
  return { type, label: shapeLabel(type), icon: SHAPE_ICONS[type] };
}

/** Toolbar shapes — digit hotkeys 1–6. */
export const SHAPES: { type: ShapeType; label: string; icon: LucideIcon }[] = (
  ["box", "sphere", "torus", "cylinder", "capsule", "cone"] as ShapeType[]
).map(shapeEntry);

/** Remaining shapes — command palette only. */
export const EXTRA_SHAPES: typeof SHAPES = (
  [
    "roundedBox",
    "boxFrame",
    "cappedTorus",
    "link",
    "hexPrism",
    "triPrism",
    "roundedCylinder",
    "roundCone",
    "solidAngle",
    "cutSphere",
    "cutHollowSphere",
    "deathStar",
    "rhombus",
    "octahedron",
    "pyramid",
    "vesica",
  ] as ShapeType[]
).map(shapeEntry);

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
  ...EXTRA_SHAPES.map(({ type, label, icon }) => ({
    id: `add-${type}`,
    label: `Add ${label}`,
    icon,
    keywords: ["shape", "primitive"],
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
