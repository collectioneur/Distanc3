import { useSceneStore, findNodeInTree, temporalStore, type OpType, type ShapeNode } from "../../store/sceneStore";

// ── Shared numeric input ──────────────────────────────────────────────────────

function NumericInput({
  label,
  value,
  min,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="prop-row">
      <label className="prop-label">{label}</label>
      <input
        className="prop-input"
        type="number"
        value={value}
        min={min}
        step={step ?? 0.1}
        onFocus={() => temporalStore.getState().pause()}
        onBlur={() => temporalStore.getState().resume()}
        onChange={(e) => {
          const n = parseFloat(e.target.value);
          if (!isNaN(n)) onChange(n);
        }}
      />
    </div>
  );
}

// ── Shape size field definitions ──────────────────────────────────────────────

interface SizeField {
  label: string;
  paramIndex: 0 | 1 | 2;
  min: number;
  step: number;
}

const SIZE_FIELDS: Record<string, SizeField[]> = {
  sphere: [{ label: "Radius", paramIndex: 0, min: 0.01, step: 0.05 }],
  box: [
    { label: "Width", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Height", paramIndex: 1, min: 0.01, step: 0.05 },
    { label: "Depth", paramIndex: 2, min: 0.01, step: 0.05 },
  ],
  torus: [
    { label: "Major R", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Minor r", paramIndex: 1, min: 0.01, step: 0.05 },
  ],
  cylinder: [
    { label: "Radius", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Half Height", paramIndex: 1, min: 0.01, step: 0.05 },
  ],
  capsule: [
    { label: "Radius", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Half Height", paramIndex: 1, min: 0.01, step: 0.05 },
  ],
  cone: [
    { label: "Radius", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Half Height", paramIndex: 1, min: 0.01, step: 0.05 },
  ],
};

// ── Op type configuration ─────────────────────────────────────────────────────

const OP_OPTIONS: { value: OpType; label: string; icon: string }[] = [
  { value: "union", label: "Union", icon: "∪" },
  { value: "subtract", label: "Subtract", icon: "∖" },
  { value: "intersect", label: "Intersect", icon: "∩" },
  { value: "sUnion", label: "Smooth Union", icon: "~∪" },
  { value: "sSubtract", label: "Smooth Subtract", icon: "~∖" },
  { value: "sIntersect", label: "Smooth Intersect", icon: "~∩" },
];

const SMOOTH_OPS: Set<OpType> = new Set(["sUnion", "sSubtract", "sIntersect"]);

// ── Sub-panels ────────────────────────────────────────────────────────────────

function ObjectProperties() {
  const objects = useSceneStore((s) => s.objects);
  const selectedObjectId = useSceneStore((s) => s.selectedObjectId);
  const updateObjectName = useSceneStore((s) => s.updateObjectName);

  const obj = objects.find((o) => o.id === selectedObjectId) ?? null;
  if (!obj) return null;

  return (
    <div className="props-content">
      <div className="props-section-title">Object</div>
      <div className="prop-row">
        <label className="prop-label">Name</label>
        <input
          className="prop-input"
          type="text"
          value={obj.name}
          onFocus={() => temporalStore.getState().pause()}
          onBlur={() => temporalStore.getState().resume()}
          onChange={(e) => updateObjectName(obj.id, e.target.value)}
        />
      </div>
    </div>
  );
}

function ShapeProperties({ shape, objectId }: { shape: ShapeNode; objectId: string }) {
  const updateShapeNode = useSceneStore((s) => s.updateShapeNode);

  const setPosition = (axis: 0 | 1 | 2, v: number) => {
    const pos: [number, number, number] = [...shape.position];
    pos[axis] = v;
    updateShapeNode(objectId, shape.id, { position: pos });
  };

  const setRotation = (axis: 0 | 1 | 2, v: number) => {
    const rot: [number, number, number] = [...(shape.rotation ?? [0, 0, 0])];
    rot[axis] = v;
    updateShapeNode(objectId, shape.id, { rotation: rot });
  };

  const setParam = (idx: 0 | 1 | 2, v: number) => {
    const params: [number, number, number, number] = [...shape.params];
    params[idx] = v;
    updateShapeNode(objectId, shape.id, { params });
  };

  const sizeFields = SIZE_FIELDS[shape.shapeType] ?? [];

  return (
    <div className="props-content">
      <div className="props-section-title">Position</div>
      <NumericInput label="X" value={shape.position[0]} step={0.1} onChange={(v) => setPosition(0, v)} />
      <NumericInput label="Y" value={shape.position[1]} step={0.1} onChange={(v) => setPosition(1, v)} />
      <NumericInput label="Z" value={shape.position[2]} step={0.1} onChange={(v) => setPosition(2, v)} />

      <div className="props-section-title" style={{ marginTop: "16px" }}>Rotation (°)</div>
      <NumericInput label="Rx" value={(shape.rotation ?? [0, 0, 0])[0]} step={1} onChange={(v) => setRotation(0, v)} />
      <NumericInput label="Ry" value={(shape.rotation ?? [0, 0, 0])[1]} step={1} onChange={(v) => setRotation(1, v)} />
      <NumericInput label="Rz" value={(shape.rotation ?? [0, 0, 0])[2]} step={1} onChange={(v) => setRotation(2, v)} />

      <div className="props-section-title" style={{ marginTop: "16px" }}>Size</div>
      {sizeFields.map((f) => (
        <NumericInput
          key={f.label}
          label={f.label}
          value={shape.params[f.paramIndex]}
          min={f.min}
          step={f.step}
          onChange={(v) => setParam(f.paramIndex, v)}
        />
      ))}
    </div>
  );
}

function OpProperties({
  nodeId,
  objectId,
}: {
  nodeId: string;
  objectId: string;
}) {
  const objects = useSceneStore((s) => s.objects);
  const updateOpNode = useSceneStore((s) => s.updateOpNode);

  const obj = objects.find((o) => o.id === objectId) ?? null;
  const node = findNodeInTree(obj?.root ?? null, nodeId);
  if (!node || node.kind !== "op") return null;

  const isSmooth = SMOOTH_OPS.has(node.op);

  return (
    <div className="props-content">
      <div className="props-section-title">Operation</div>
      <div className="prop-row">
        <label className="prop-label">Type</label>
        <select
          className="prop-input prop-select"
          value={node.op}
          onChange={(e) =>
            updateOpNode(objectId, nodeId, { op: e.target.value as OpType })
          }
        >
          {OP_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.icon} {opt.label}
            </option>
          ))}
        </select>
      </div>

      {isSmooth && (
        <NumericInput
          label="Smooth K"
          value={node.smoothK}
          min={0.001}
          step={0.01}
          onChange={(v) => updateOpNode(objectId, nodeId, { smoothK: v })}
        />
      )}

      <p className="props-op-hint">
        {node.op === "union" && "Merges both shapes."}
        {node.op === "subtract" && "Subtracts right from left."}
        {node.op === "intersect" && "Keeps only the overlap."}
        {node.op === "sUnion" && "Smoothly blends both shapes."}
        {node.op === "sSubtract" && "Smoothly subtracts right from left."}
        {node.op === "sIntersect" && "Smoothly intersects both shapes."}
      </p>
    </div>
  );
}

// ── Panel ─────────────────────────────────────────────────────────────────────

export default function RightPanel() {
  const objects = useSceneStore((s) => s.objects);
  const selectedObjectId = useSceneStore((s) => s.selectedObjectId);
  const selectedNodeId = useSceneStore((s) => s.selectedNodeId);

  const selectedObject = objects.find((o) => o.id === selectedObjectId) ?? null;
  const selectedNode = selectedObject
    ? findNodeInTree(selectedObject.root, selectedNodeId)
    : null;

  let title = "";
  let content: React.ReactNode = null;

  if (selectedNode) {
    if (selectedNode.kind === "shape") {
      title = selectedNode.name;
      content = (
        <ShapeProperties shape={selectedNode} objectId={selectedObjectId!} />
      );
    } else {
      title = "Operation";
      content = (
        <OpProperties nodeId={selectedNode.id} objectId={selectedObjectId!} />
      );
    }
  } else if (selectedObject) {
    title = selectedObject.name;
    content = <ObjectProperties />;
  }

  return (
    <div className="panel panel-right">
      <div className="panel-header">Properties</div>
      {content ? (
        <>
          <div className="props-shape-name">{title}</div>
          {content}
        </>
      ) : (
        <p className="scene-empty">
          Select an object or node
          <br />
          to edit its properties.
        </p>
      )}
    </div>
  );
}
