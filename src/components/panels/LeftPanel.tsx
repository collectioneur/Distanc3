import { useState } from "react";
import {
  useSceneStore,
  type CsgNode,
  type OpType,
  type SceneObject,
} from "../../store/sceneStore";

const SHAPE_ICON: Record<string, string> = {
  sphere: "○",
  box: "□",
  torus: "◎",
  cylinder: "⌀",
  capsule: "⬬",
  cone: "△",
};

const OP_ICON: Record<OpType, string> = {
  union: "∪",
  subtract: "∖",
  intersect: "∩",
  sUnion: "~∪",
  sSubtract: "~∖",
  sIntersect: "~∩",
};

const OP_LABEL: Record<OpType, string> = {
  union: "Union",
  subtract: "Subtract",
  intersect: "Intersect",
  sUnion: "Smooth Union",
  sSubtract: "Smooth Subtract",
  sIntersect: "Smooth Intersect",
};

// ── Recursive node renderer ───────────────────────────────────────────────────

function NodeRow({
  node,
  objectId,
  depth,
}: {
  node: CsgNode;
  objectId: string;
  depth: number;
}) {
  const selectedNodeId = useSceneStore((s) => s.selectedNodeId);
  const selectNodeInObject = useSceneStore((s) => s.selectNodeInObject);
  const removeNode = useSceneStore((s) => s.removeNode);

  const paddingLeft = 8 + depth * 16;
  const isSelected = selectedNodeId === node.id;

  if (node.kind === "shape") {
    return (
      <div
        className={`scene-item${isSelected ? " scene-item--selected" : ""}`}
        style={{ paddingLeft }}
        onClick={() => selectNodeInObject(objectId, node.id)}
      >
        <span className="scene-item-icon">{SHAPE_ICON[node.shapeType]}</span>
        <span className="scene-item-name">{node.name}</span>
        <button
          className="scene-item-remove"
          title="Remove shape"
          onClick={(e) => {
            e.stopPropagation();
            removeNode(objectId, node.id);
          }}
        >
          ×
        </button>
      </div>
    );
  }

  // OpNode: show op row first (pre-order), then left and right children
  return (
    <>
      <div
        className={`scene-item scene-item--op${isSelected ? " scene-item--selected" : ""}`}
        style={{ paddingLeft }}
        onClick={() => selectNodeInObject(objectId, node.id)}
        title={OP_LABEL[node.op]}
      >
        <span className="scene-item-icon scene-item-icon--op">{OP_ICON[node.op]}</span>
        <span className="scene-item-name scene-item-name--op">{OP_LABEL[node.op]}</span>
      </div>
      <NodeRow node={node.left} objectId={objectId} depth={depth + 1} />
      <NodeRow node={node.right} objectId={objectId} depth={depth + 1} />
    </>
  );
}

// ── Object row ────────────────────────────────────────────────────────────────

function ObjectRow({ obj }: { obj: SceneObject }) {
  const [collapsed, setCollapsed] = useState(false);
  const selectedObjectId = useSceneStore((s) => s.selectedObjectId);
  const selectedNodeId = useSceneStore((s) => s.selectedNodeId);
  const selectObject = useSceneStore((s) => s.selectObject);
  const removeObject = useSceneStore((s) => s.removeObject);

  const isActive = selectedObjectId === obj.id;
  const hasSelection = isActive && selectedNodeId === null;

  return (
    <div className="scene-object">
      <div
        className={`scene-object-header${hasSelection ? " scene-item--selected" : ""}`}
        onClick={() => selectObject(obj.id)}
      >
        <button
          className="scene-object-collapse"
          onClick={(e) => {
            e.stopPropagation();
            setCollapsed((c) => !c);
          }}
          title={collapsed ? "Expand" : "Collapse"}
        >
          {collapsed ? "▶" : "▼"}
        </button>
        <span className="scene-object-name">{obj.name}</span>
        <button
          className="scene-item-remove"
          title="Remove object"
          onClick={(e) => {
            e.stopPropagation();
            removeObject(obj.id);
          }}
        >
          ×
        </button>
      </div>

      {!collapsed && obj.root && (
        <div className="scene-object-nodes">
          <NodeRow node={obj.root} objectId={obj.id} depth={0} />
        </div>
      )}

      {!collapsed && !obj.root && (
        <p className="scene-object-empty">Empty — add a shape from the top bar.</p>
      )}
    </div>
  );
}

// ── Panel ─────────────────────────────────────────────────────────────────────

export default function LeftPanel() {
  const objects = useSceneStore((s) => s.objects);
  const addObject = useSceneStore((s) => s.addObject);

  return (
    <div className="panel panel-left">
      <div className="panel-header">Scene</div>
      <div className="scene-list">
        {objects.length === 0 && (
          <p className="scene-empty">
            No objects yet.
            <br />
            Click <strong>+ Object</strong> below.
          </p>
        )}
        {objects.map((obj) => (
          <ObjectRow key={obj.id} obj={obj} />
        ))}
      </div>
      <div className="scene-add-object">
        <button
          className="add-object-btn"
          onClick={addObject}
          disabled={objects.length >= 8}
          title={objects.length >= 8 ? "Max 8 objects" : "Add new object"}
        >
          + Object
        </button>
      </div>
    </div>
  );
}
