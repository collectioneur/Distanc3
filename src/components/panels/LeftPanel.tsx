import { useState } from "react";
import {
  useSceneStore,
  findContainer,
  wouldExceedCap,
  wouldExceedDepth,
  type ObjectGroup,
  type OpType,
  type SceneItem,
  type ShapeLayer,
} from "../../store/sceneStore";
import { showToast } from "../../utils/toast";

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

function LayerRow({
  layer,
  containerId,
  showOp,
  depth,
}: {
  layer: ShapeLayer;
  containerId: string;
  showOp: boolean;
  depth: number;
}) {
  const selectedItemId = useSceneStore((s) => s.selectedItemId);
  const selectItem = useSceneStore((s) => s.selectItem);
  const removeItem = useSceneStore((s) => s.removeItem);

  const isSelected = selectedItemId === layer.id;

  return (
    <div
      className={`scene-item${isSelected ? " scene-item--selected" : ""}`}
      style={{ paddingLeft: 8 + depth * 12 }}
      onClick={() => selectItem(containerId, layer.id)}
    >
      <span className="scene-item-icon">{SHAPE_ICON[layer.shapeType]}</span>
      <span className="scene-item-name">{layer.name}</span>
      {showOp && (
        <span className="scene-item-icon scene-item-icon--op" title={layer.op}>
          {OP_ICON[layer.op]}
        </span>
      )}
      <button
        className="scene-item-remove"
        title="Remove shape"
        onClick={(e) => {
          e.stopPropagation();
          removeItem(containerId, layer.id);
        }}
      >
        ×
      </button>
    </div>
  );
}

function GroupRow({
  group,
  parentContainerId,
  showOp,
  depth,
}: {
  group: ObjectGroup;
  parentContainerId: string;
  showOp: boolean;
  depth: number;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const selectedItemId = useSceneStore((s) => s.selectedItemId);
  const selectItem = useSceneStore((s) => s.selectItem);
  const removeItem = useSceneStore((s) => s.removeItem);

  const isSelected = selectedItemId === group.id;

  return (
    <div className="scene-object" style={{ marginLeft: depth * 12 }}>
      <div
        className={`scene-object-header scene-item${isSelected ? " scene-item--selected" : ""}`}
        style={{ paddingLeft: 8 }}
        onClick={() => selectItem(parentContainerId, group.id)}
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
        <span className="scene-item-icon">◫</span>
        <span className="scene-object-name">{group.name}</span>
        {showOp && (
          <span className="scene-item-icon scene-item-icon--op" title={group.op}>
            {OP_ICON[group.op]}
          </span>
        )}
        <button
          className="scene-item-remove"
          title="Remove object"
          onClick={(e) => {
            e.stopPropagation();
            removeItem(parentContainerId, group.id);
          }}
        >
          ×
        </button>
      </div>

      {!collapsed && group.items.length > 0 && (
        <div className="scene-object-nodes">
          <ItemList containerId={group.id} items={group.items} depth={depth + 1} />
        </div>
      )}

      {!collapsed && group.items.length === 0 && (
        <p className="scene-object-empty" style={{ paddingLeft: 8 + (depth + 1) * 12 }}>
          Empty — add shapes inside this object.
        </p>
      )}
    </div>
  );
}

function ItemList({
  containerId,
  items,
  depth,
}: {
  containerId: string;
  items: SceneItem[];
  depth: number;
}) {
  return (
    <>
      {items.map((item, index) =>
        item.kind === "layer" ? (
          <LayerRow
            key={item.id}
            layer={item}
            containerId={containerId}
            showOp={index > 0}
            depth={depth}
          />
        ) : (
          <GroupRow
            key={item.id}
            group={item}
            parentContainerId={containerId}
            showOp={index > 0}
            depth={depth}
          />
        ),
      )}
    </>
  );
}

function SceneTree() {
  const [collapsed, setCollapsed] = useState(false);
  const root = useSceneStore((s) => s.root);
  const selectedItemId = useSceneStore((s) => s.selectedItemId);
  const selectRoot = useSceneStore((s) => s.selectRoot);

  const hasRootSelection = selectedItemId === null;

  return (
    <div className="scene-object">
      <div
        className={`scene-object-header${hasRootSelection ? " scene-item--selected" : ""}`}
        onClick={selectRoot}
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
        <span className="scene-object-name">{root.name}</span>
      </div>

      {!collapsed && root.items.length > 0 && (
        <div className="scene-object-nodes">
          <ItemList containerId={root.id} items={root.items} depth={0} />
        </div>
      )}

      {!collapsed && root.items.length === 0 && (
        <p className="scene-object-empty">Empty — add a shape from the top bar.</p>
      )}
    </div>
  );
}

export default function LeftPanel() {
  const root = useSceneStore((s) => s.root);
  const selectedContainerId = useSceneStore((s) => s.selectedContainerId);
  const addGroupToContainer = useSceneStore((s) => s.addGroupToContainer);

  const container = findContainer(root, selectedContainerId);
  const groupExtra = container && container.items.length > 0 ? 1 : 0;
  const capExceeded = wouldExceedCap(root, groupExtra);
  const depthExceeded = wouldExceedDepth(root, selectedContainerId);

  function handleAddGroup() {
    if (depthExceeded) {
      showToast("Nesting too deep (max 16 levels)");
      return;
    }
    const ok = addGroupToContainer(selectedContainerId);
    if (!ok) {
      if (wouldExceedDepth(root, selectedContainerId)) {
        showToast("Nesting too deep (max 16 levels)");
      } else {
        showToast("Scene too complex (max 256 operations)");
      }
    }
  }

  return (
    <div className="panel panel-left">
      <div className="panel-header">Scene</div>
      <div className="scene-list">
        <SceneTree />
      </div>
      <div className="scene-add-object">
        <button
          className="add-object-btn"
          onClick={handleAddGroup}
          disabled={capExceeded || depthExceeded}
          title={
            depthExceeded
              ? "Nesting too deep (max 16 levels)"
              : capExceeded
                ? "Scene too complex (max 256 operations)"
                : "Add nested object to selected container"
          }
        >
          + Object
        </button>
      </div>
    </div>
  );
}
