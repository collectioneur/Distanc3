import { useSceneStore, type ShapeInstance } from "../../store/sceneStore";

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
        onChange={(e) => {
          const n = parseFloat(e.target.value);
          if (!isNaN(n)) onChange(n);
        }}
      />
    </div>
  );
}

function ShapeProperties({ shape }: { shape: ShapeInstance }) {
  const updateShape = useSceneStore((s) => s.updateShape);

  const setPosition = (axis: 0 | 1 | 2, v: number) => {
    const pos: [number, number, number] = [...shape.position];
    pos[axis] = v;
    updateShape(shape.id, { position: pos });
  };

  const setParam = (idx: 0 | 1 | 2, v: number) => {
    const params: [number, number, number, number] = [...shape.params];
    params[idx] = v;
    updateShape(shape.id, { params });
  };

  const sizeFields = SIZE_FIELDS[shape.type] ?? [];

  return (
    <div className="props-content">
      <div className="props-section-title">Position</div>
      <NumericInput label="X" value={shape.position[0]} step={0.1} onChange={(v) => setPosition(0, v)} />
      <NumericInput label="Y" value={shape.position[1]} step={0.1} onChange={(v) => setPosition(1, v)} />
      <NumericInput label="Z" value={shape.position[2]} step={0.1} onChange={(v) => setPosition(2, v)} />

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

export default function RightPanel() {
  const shapes = useSceneStore((s) => s.shapes);
  const selectedId = useSceneStore((s) => s.selectedId);
  const selectedShape = shapes.find((s) => s.id === selectedId) ?? null;

  return (
    <div className="panel panel-right">
      <div className="panel-header">Properties</div>
      {selectedShape ? (
        <>
          <div className="props-shape-name">{selectedShape.name}</div>
          <ShapeProperties shape={selectedShape} />
        </>
      ) : (
        <p className="scene-empty">Select a shape<br />to edit its properties.</p>
      )}
    </div>
  );
}
