import { useSceneStore } from "../../store/sceneStore";

const TYPE_ICON: Record<string, string> = {
  sphere: "○",
  box: "□",
  torus: "◎",
  cylinder: "⌀",
  capsule: "⬬",
  cone: "△",
};

export default function LeftPanel() {
  const shapes = useSceneStore((s) => s.shapes);
  const selectedId = useSceneStore((s) => s.selectedId);
  const selectShape = useSceneStore((s) => s.selectShape);
  const removeShape = useSceneStore((s) => s.removeShape);

  return (
    <div className="panel panel-left">
      <div className="panel-header">Scene</div>
      <div className="scene-list">
        {shapes.length === 0 && (
          <p className="scene-empty">No shapes yet.<br />Add one from the top bar.</p>
        )}
        {shapes.map((shape) => (
          <div
            key={shape.id}
            className={`scene-item${selectedId === shape.id ? " scene-item--selected" : ""}`}
            onClick={() => selectShape(shape.id)}
          >
            <span className="scene-item-icon">{TYPE_ICON[shape.type]}</span>
            <span className="scene-item-name">{shape.name}</span>
            <button
              className="scene-item-remove"
              onClick={(e) => {
                e.stopPropagation();
                removeShape(shape.id);
              }}
              title="Remove shape"
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
