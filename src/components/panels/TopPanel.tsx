import { useSceneStore, MAX_SCENE_SHAPES, type ShapeType } from "../../store/sceneStore";

const SHAPES: { type: ShapeType; label: string; icon: string }[] = [
  { type: "box", label: "Box", icon: "□" },
  { type: "sphere", label: "Sphere", icon: "○" },
  { type: "torus", label: "Torus", icon: "◎" },
  { type: "cylinder", label: "Cylinder", icon: "⌀" },
  { type: "capsule", label: "Capsule", icon: "⬬" },
  { type: "cone", label: "Cone", icon: "△" },
];

export default function TopPanel() {
  const addShape = useSceneStore((s) => s.addShape);
  const shapeCount = useSceneStore((s) => s.shapes.length);
  const isFull = shapeCount >= MAX_SCENE_SHAPES;

  return (
    <div className="panel panel-top">
      <span className="panel-label">Shapes</span>
      <div className="top-panel-shapes">
        {SHAPES.map(({ type, label, icon }) => (
          <button
            key={type}
            className="shape-btn"
            onClick={() => addShape(type)}
            disabled={isFull}
            title={isFull ? "Scene is full (max 8 shapes)" : `Add ${label}`}
          >
            <span className="shape-btn-icon">{icon}</span>
            <span className="shape-btn-label">{label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
