import { useState } from "react";
import {
  useSceneStore,
  temporalStore,
  findItem,
  type ObjectGroup,
  type OpType,
  type ShapeLayer,
} from "../../store/sceneStore";
import { roundTo } from "../../utils/round";
import CodeView from "./CodeView";

// ── Shared numeric input ──────────────────────────────────────────────────────

function NumericInput({
  label,
  value,
  min,
  max,
  step,
  decimals,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  decimals?: number;
  onChange: (v: number) => void;
}) {
  const display = decimals != null ? roundTo(value, decimals) : value;

  return (
    <div className="prop-row">
      <label className="prop-label">{label}</label>
      <input
        className="prop-input"
        type="number"
        value={display}
        min={min}
        max={max}
        step={step ?? 0.1}
        onFocus={() => temporalStore.getState().pause()}
        onBlur={() => temporalStore.getState().resume()}
        onChange={(e) => {
          const n = parseFloat(e.target.value);
          if (!isNaN(n)) onChange(decimals != null ? roundTo(n, decimals) : n);
        }}
      />
    </div>
  );
}

// ── Shape size field definitions ──────────────────────────────────────────────

interface SizeField {
  label: string;
  paramIndex: 0 | 1 | 2 | 3;
  min: number;
  step: number;
  max?: number;
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
  roundedBox: [
    { label: "Width", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Height", paramIndex: 1, min: 0.01, step: 0.05 },
    { label: "Depth", paramIndex: 2, min: 0.01, step: 0.05 },
    { label: "Round r", paramIndex: 3, min: 0, step: 0.01 },
  ],
  boxFrame: [
    { label: "Width", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Height", paramIndex: 1, min: 0.01, step: 0.05 },
    { label: "Depth", paramIndex: 2, min: 0.01, step: 0.05 },
    { label: "Edge", paramIndex: 3, min: 0.005, step: 0.01 },
  ],
  cappedTorus: [
    { label: "Major R", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Minor r", paramIndex: 1, min: 0.01, step: 0.05 },
    { label: "Angle °", paramIndex: 2, min: 1, max: 180, step: 5 },
  ],
  link: [
    { label: "Half Length", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Major R", paramIndex: 1, min: 0.01, step: 0.05 },
    { label: "Minor r", paramIndex: 2, min: 0.01, step: 0.01 },
  ],
  hexPrism: [
    { label: "Radius", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Half Height", paramIndex: 1, min: 0.01, step: 0.05 },
  ],
  triPrism: [
    { label: "Radius", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Half Height", paramIndex: 1, min: 0.01, step: 0.05 },
  ],
  roundedCylinder: [
    { label: "Radius", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Round r", paramIndex: 1, min: 0.005, step: 0.01 },
    { label: "Half Height", paramIndex: 2, min: 0.01, step: 0.05 },
  ],
  roundCone: [
    { label: "Bottom R", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Top R", paramIndex: 1, min: 0.01, step: 0.05 },
    { label: "Height", paramIndex: 2, min: 0.01, step: 0.05 },
  ],
  solidAngle: [
    { label: "Angle °", paramIndex: 0, min: 1, max: 179, step: 5 },
    { label: "Radius", paramIndex: 1, min: 0.01, step: 0.05 },
  ],
  cutSphere: [
    { label: "Radius", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Cut Height", paramIndex: 1, min: -10, step: 0.05 },
  ],
  cutHollowSphere: [
    { label: "Radius", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Cut Height", paramIndex: 1, min: -10, step: 0.05 },
    { label: "Thickness", paramIndex: 2, min: 0.005, step: 0.01 },
  ],
  deathStar: [
    { label: "Sphere R", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Carve R", paramIndex: 1, min: 0.01, step: 0.05 },
    { label: "Offset", paramIndex: 2, min: 0.01, step: 0.05 },
  ],
  rhombus: [
    { label: "Half Diag A", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Half Diag B", paramIndex: 1, min: 0.01, step: 0.05 },
    { label: "Half Height", paramIndex: 2, min: 0.01, step: 0.05 },
    { label: "Round r", paramIndex: 3, min: 0, step: 0.01 },
  ],
  octahedron: [{ label: "Size", paramIndex: 0, min: 0.01, step: 0.05 }],
  pyramid: [
    { label: "Base", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Height", paramIndex: 1, min: 0.01, step: 0.05 },
  ],
  vesica: [
    { label: "Radius", paramIndex: 0, min: 0.01, step: 0.05 },
    { label: "Offset", paramIndex: 1, min: 0.005, step: 0.05 },
  ],
};

const OP_OPTIONS: { value: OpType; label: string; icon: string }[] = [
  { value: "union", label: "Union", icon: "∪" },
  { value: "subtract", label: "Subtract", icon: "∖" },
  { value: "intersect", label: "Intersect", icon: "∩" },
  { value: "sUnion", label: "Smooth Union", icon: "~∪" },
  { value: "sSubtract", label: "Smooth Subtract", icon: "~∖" },
  { value: "sIntersect", label: "Smooth Intersect", icon: "~∩" },
];

const SMOOTH_OPS: Set<OpType> = new Set(["sUnion", "sSubtract", "sIntersect"]);

const OP_HINT: Record<OpType, string> = {
  union: "Merges with the previous result.",
  subtract: "Subtracts from the previous result.",
  intersect: "Keeps only the overlap with the previous result.",
  sUnion: "Smoothly blends with the previous result.",
  sSubtract: "Smoothly subtracts from the previous result.",
  sIntersect: "Smoothly intersects with the previous result.",
};

function OperationFields({
  op,
  smoothK,
  showOp,
  onOpChange,
  onSmoothKChange,
}: {
  op: OpType;
  smoothK: number;
  showOp: boolean;
  onOpChange: (op: OpType) => void;
  onSmoothKChange: (k: number) => void;
}) {
  if (!showOp) return null;
  const isSmooth = SMOOTH_OPS.has(op);

  return (
    <>
      <div className="props-section-title">Operation</div>
      <div className="prop-row">
        <label className="prop-label">Type</label>
        <select
          className="prop-input prop-select"
          value={op}
          onChange={(e) => onOpChange(e.target.value as OpType)}
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
          value={smoothK}
          min={0.001}
          step={0.01}
          onChange={onSmoothKChange}
        />
      )}

      <p className="props-op-hint">{OP_HINT[op]}</p>
    </>
  );
}

function RootProperties() {
  const root = useSceneStore((s) => s.root);
  const updateRootName = useSceneStore((s) => s.updateRootName);

  return (
    <div className="props-content">
      <div className="props-section-title">Scene</div>
      <div className="prop-row">
        <label className="prop-label">Name</label>
        <input
          className="prop-input"
          type="text"
          value={root.name}
          onFocus={() => temporalStore.getState().pause()}
          onBlur={() => temporalStore.getState().resume()}
          onChange={(e) => updateRootName(e.target.value)}
        />
      </div>
    </div>
  );
}

function GroupProperties({ group, itemIndex }: { group: ObjectGroup; itemIndex: number }) {
  const updateGroup = useSceneStore((s) => s.updateGroup);
  const showOp = itemIndex > 0;

  const setPosition = (axis: 0 | 1 | 2, v: number) => {
    const pos: [number, number, number] = [...group.position];
    pos[axis] = v;
    updateGroup(group.id, { position: pos });
  };

  const setRotation = (axis: 0 | 1 | 2, v: number) => {
    const rot: [number, number, number] = [...group.rotation];
    rot[axis] = v;
    updateGroup(group.id, { rotation: rot });
  };

  const setScale = (axis: 0 | 1 | 2, v: number) => {
    const scale: [number, number, number] = [...group.scale];
    scale[axis] = v;
    updateGroup(group.id, { scale });
  };

  return (
    <div className="props-content">
      <div className="props-section-title">Object</div>
      <div className="prop-row">
        <label className="prop-label">Name</label>
        <input
          className="prop-input"
          type="text"
          value={group.name}
          onFocus={() => temporalStore.getState().pause()}
          onBlur={() => temporalStore.getState().resume()}
          onChange={(e) => updateGroup(group.id, { name: e.target.value })}
        />
      </div>

      <OperationFields
        op={group.op}
        smoothK={group.smoothK}
        showOp={showOp}
        onOpChange={(op) => updateGroup(group.id, { op })}
        onSmoothKChange={(smoothK) => updateGroup(group.id, { smoothK })}
      />

      <div className="props-section-title" style={{ marginTop: showOp ? "16px" : undefined }}>
        Position
      </div>
      <NumericInput label="X" value={group.position[0]} step={0.1} decimals={2} onChange={(v) => setPosition(0, v)} />
      <NumericInput label="Y" value={group.position[1]} step={0.1} decimals={2} onChange={(v) => setPosition(1, v)} />
      <NumericInput label="Z" value={group.position[2]} step={0.1} decimals={2} onChange={(v) => setPosition(2, v)} />

      <div className="props-section-title" style={{ marginTop: "16px" }}>Rotation (°)</div>
      <NumericInput label="Rx" value={group.rotation[0]} step={1} decimals={2} onChange={(v) => setRotation(0, v)} />
      <NumericInput label="Ry" value={group.rotation[1]} step={1} decimals={2} onChange={(v) => setRotation(1, v)} />
      <NumericInput label="Rz" value={group.rotation[2]} step={1} decimals={2} onChange={(v) => setRotation(2, v)} />

      <div className="props-section-title" style={{ marginTop: "16px" }}>Scale</div>
      <NumericInput label="Sx" value={group.scale[0]} min={0.01} step={0.1} decimals={2} onChange={(v) => setScale(0, v)} />
      <NumericInput label="Sy" value={group.scale[1]} min={0.01} step={0.1} decimals={2} onChange={(v) => setScale(1, v)} />
      <NumericInput label="Sz" value={group.scale[2]} min={0.01} step={0.1} decimals={2} onChange={(v) => setScale(2, v)} />
    </div>
  );
}

function LayerProperties({
  layer,
  containerId,
  itemIndex,
}: {
  layer: ShapeLayer;
  containerId: string;
  itemIndex: number;
}) {
  const updateLayer = useSceneStore((s) => s.updateLayer);

  const setPosition = (axis: 0 | 1 | 2, v: number) => {
    const pos: [number, number, number] = [...layer.position];
    pos[axis] = v;
    updateLayer(containerId, layer.id, { position: pos });
  };

  const setRotation = (axis: 0 | 1 | 2, v: number) => {
    const rot: [number, number, number] = [...layer.rotation];
    rot[axis] = v;
    updateLayer(containerId, layer.id, { rotation: rot });
  };

  const setParam = (idx: 0 | 1 | 2 | 3, v: number) => {
    const params: [number, number, number, number] = [...layer.params];
    params[idx] = v;
    updateLayer(containerId, layer.id, { params });
  };

  const setScale = (axis: 0 | 1 | 2, v: number) => {
    const scale: [number, number, number] = [...layer.scale];
    scale[axis] = v;
    updateLayer(containerId, layer.id, { scale });
  };

  const sizeFields = SIZE_FIELDS[layer.shapeType] ?? [];
  const showOp = itemIndex > 0;

  return (
    <div className="props-content">
      <OperationFields
        op={layer.op}
        smoothK={layer.smoothK}
        showOp={showOp}
        onOpChange={(op) => updateLayer(containerId, layer.id, { op })}
        onSmoothKChange={(smoothK) => updateLayer(containerId, layer.id, { smoothK })}
      />

      <div className="props-section-title" style={{ marginTop: showOp ? "16px" : undefined }}>
        Position
      </div>
      <NumericInput label="X" value={layer.position[0]} step={0.1} decimals={2} onChange={(v) => setPosition(0, v)} />
      <NumericInput label="Y" value={layer.position[1]} step={0.1} decimals={2} onChange={(v) => setPosition(1, v)} />
      <NumericInput label="Z" value={layer.position[2]} step={0.1} decimals={2} onChange={(v) => setPosition(2, v)} />

      <div className="props-section-title" style={{ marginTop: "16px" }}>Rotation (°)</div>
      <NumericInput label="Rx" value={layer.rotation[0]} step={1} decimals={2} onChange={(v) => setRotation(0, v)} />
      <NumericInput label="Ry" value={layer.rotation[1]} step={1} decimals={2} onChange={(v) => setRotation(1, v)} />
      <NumericInput label="Rz" value={layer.rotation[2]} step={1} decimals={2} onChange={(v) => setRotation(2, v)} />

      <div className="props-section-title" style={{ marginTop: "16px" }}>Scale</div>
      <NumericInput label="Sx" value={layer.scale[0]} min={0.01} step={0.1} decimals={2} onChange={(v) => setScale(0, v)} />
      <NumericInput label="Sy" value={layer.scale[1]} min={0.01} step={0.1} decimals={2} onChange={(v) => setScale(1, v)} />
      <NumericInput label="Sz" value={layer.scale[2]} min={0.01} step={0.1} decimals={2} onChange={(v) => setScale(2, v)} />

      <div className="props-section-title" style={{ marginTop: "16px" }}>Size</div>
      {sizeFields.map((f) => (
        <NumericInput
          key={f.label}
          label={f.label}
          value={layer.params[f.paramIndex]}
          min={f.min}
          max={f.max}
          step={f.step}
          decimals={2}
          onChange={(v) => setParam(f.paramIndex, v)}
        />
      ))}
    </div>
  );
}

export default function RightPanel() {
  const root = useSceneStore((s) => s.root);
  const selectedItemId = useSceneStore((s) => s.selectedItemId);
  const rootSelected = useSceneStore((s) => s.rootSelected);
  const [tab, setTab] = useState<"props" | "code">("props");

  const found = selectedItemId ? findItem(root, selectedItemId) : null;

  let title = root.name;
  let content: React.ReactNode = null;

  if (found?.item.kind === "layer") {
    title = found.item.name;
    content = (
      <LayerProperties
        layer={found.item}
        containerId={found.container.id}
        itemIndex={found.index}
      />
    );
  } else if (found?.item.kind === "group") {
    title = found.item.name;
    content = <GroupProperties group={found.item} itemIndex={found.index} />;
  } else if (rootSelected) {
    content = <RootProperties />;
  } else {
    title = "Properties";
    content = (
      <p className="scene-object-empty">Nothing selected — click an object in the viewport or scene tree.</p>
    );
  }

  return (
    <div className="panel panel-right">
      <div className="panel-header panel-header--tabs">
        <button
          className={`panel-tab${tab === "props" ? " panel-tab--active" : ""}`}
          onClick={() => setTab("props")}
        >
          Properties
        </button>
        <div className="panel-tab-divider" />
        <button
          className={`panel-tab${tab === "code" ? " panel-tab--active" : ""}`}
          onClick={() => setTab("code")}
        >
          Code
        </button>
      </div>
      {tab === "props" ? (
        <>
          <div className="props-shape-name">{title}</div>
          {content}
        </>
      ) : (
        <CodeView />
      )}
    </div>
  );
}
