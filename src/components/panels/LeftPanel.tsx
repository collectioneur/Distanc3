import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Tree, type DragPreviewProps, type NodeRendererProps } from "react-arborist";
import { useTreeApi } from "react-arborist/dist/module/context.js";
import {
  arboristChildrenAccessor,
  canDropAt,
  rootItemsToArborist,
  type ArboristNode,
} from "../../scene/arboristAdapter";
import { useElementSize } from "../../hooks/useElementSize";
import {
  useSceneStore,
  findContainer,
  findItem,
  wouldExceedCap,
  wouldExceedDepth,
  type OpType,
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

function SceneNode({
  node,
  style,
  dragHandle,
  preview,
}: NodeRendererProps<ArboristNode>) {
  const root = useSceneStore((s) => s.root);
  const removeItem = useSceneStore((s) => s.removeItem);
  const data = node.data;
  const showOp = node.childIndex > 0;
  const isGroup = data.kind === "group";

  const containerId = useMemo(() => {
    const found = findItem(root, data.id);
    return found?.container.id ?? root.id;
  }, [root, data.id]);

  if (node.isEditing && !preview) {
    return (
      <div ref={dragHandle} style={style} className="scene-item scene-item--editing">
        <NodeRenameInput node={node} />
      </div>
    );
  }

  return (
    <div
      ref={preview ? undefined : dragHandle}
      style={style}
      className={`scene-item${!preview && node.isSelected ? " scene-item--selected" : ""}`}
      onClick={preview ? undefined : node.handleClick}
    >
      {isGroup && (
        <button
          type="button"
          className="scene-object-collapse"
          onClick={(e) => {
            e.stopPropagation();
            node.toggle();
          }}
          title={node.isOpen ? "Collapse" : "Expand"}
        >
          {node.isOpen ? "▼" : "▶"}
        </button>
      )}
      {!isGroup && <span className="scene-item-icon scene-item-icon--spacer" />}
      <span className="scene-item-icon">
        {isGroup ? "◫" : SHAPE_ICON[data.shapeType ?? "sphere"]}
      </span>
      <span className="scene-item-name">{data.name}</span>
      {showOp && (
        <span className="scene-item-icon scene-item-icon--op" title={data.op}>
          {OP_ICON[data.op]}
        </span>
      )}
      {!preview && (
        <button
          type="button"
          className="scene-item-remove"
          title={isGroup ? "Remove object" : "Remove shape"}
          onClick={(e) => {
            e.stopPropagation();
            removeItem(containerId, data.id);
          }}
        >
          ×
        </button>
      )}
    </div>
  );
}

/** Portal to body: panel transform/backdrop-filter break position:fixed for the default preview. */
function SceneDragPreview({ offset, id, isDragging }: DragPreviewProps) {
  const tree = useTreeApi<ArboristNode>();

  if (!isDragging || !offset || !id) return null;

  const node = tree.get(id);
  if (!node) return null;

  const previewWidth = typeof tree.width === "number" ? tree.width : 218;

  return createPortal(
    <div
      className="scene-drag-preview-layer"
      style={{
        position: "fixed",
        left: 0,
        top: 0,
        zIndex: 10000,
        pointerEvents: "none",
        transform: `translate(${offset.x}px, ${offset.y}px)`,
        width: previewWidth,
      }}
    >
      <SceneNode
        preview
        node={node}
        tree={tree}
        style={{
          paddingLeft: node.level * tree.indent,
          opacity: 0.45,
          width: "100%",
        }}
      />
    </div>,
    document.body,
  );
}

function NodeRenameInput({ node }: { node: NodeRendererProps<ArboristNode>["node"] }) {
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  return (
    <input
      ref={inputRef}
      className="scene-item-rename-input"
      defaultValue={node.data.name}
      onFocus={(e) => e.currentTarget.select()}
      onBlur={() => node.reset()}
      onKeyDown={(e) => {
        if (e.key === "Escape") node.reset();
        if (e.key === "Enter") node.submit(inputRef.current?.value ?? "");
      }}
      onClick={(e) => e.stopPropagation()}
    />
  );
}

function SceneRootHeader({
  collapsed,
  onToggleCollapse,
}: {
  collapsed: boolean;
  onToggleCollapse: () => void;
}) {
  const root = useSceneStore((s) => s.root);
  const selectedItemId = useSceneStore((s) => s.selectedItemId);
  const rootSelected = useSceneStore((s) => s.rootSelected);
  const selectRoot = useSceneStore((s) => s.selectRoot);
  const hasRootSelection = rootSelected && selectedItemId === null;

  return (
    <div className="scene-object scene-object--root">
      <div
        className={`scene-object-header scene-item${hasRootSelection ? " scene-item--selected" : ""}`}
        style={{ paddingLeft: 8 }}
        onClick={selectRoot}
      >
        <button
          type="button"
          className="scene-object-collapse"
          onClick={(e) => {
            e.stopPropagation();
            onToggleCollapse();
          }}
          title={collapsed ? "Expand" : "Collapse"}
        >
          {collapsed ? "▶" : "▼"}
        </button>
        <span className="scene-object-name">{root.name}</span>
      </div>
    </div>
  );
}

export default function LeftPanel() {
  const [sceneCollapsed, setSceneCollapsed] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);
  const { width, height } = useElementSize(listRef);

  const root = useSceneStore((s) => s.root);
  const selectedItemId = useSceneStore((s) => s.selectedItemId);
  const selectedContainerId = useSceneStore((s) => s.selectedContainerId);
  const addGroupToContainer = useSceneStore((s) => s.addGroupToContainer);
  const moveItems = useSceneStore((s) => s.moveItems);
  const renameItem = useSceneStore((s) => s.renameItem);
  const selectItem = useSceneStore((s) => s.selectItem);

  const arboristData = useMemo(() => rootItemsToArborist(root.items), [root.items]);

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

  function syncSelection(nodes: { id: string; data: ArboristNode }[]) {
    if (nodes.length === 0) return;
    const id = nodes[0].id;
    const found = findItem(root, id);
    if (!found) return;
    const containerId =
      found.item.kind === "group" ? found.item.id : found.container.id;
    selectItem(containerId, id);
  }

  return (
    <div className="panel panel-left">
      <div className="panel-header">Scene</div>
      <SceneRootHeader
        collapsed={sceneCollapsed}
        onToggleCollapse={() => setSceneCollapsed((c) => !c)}
      />
      <div className="scene-list" ref={listRef}>
        {!sceneCollapsed && root.items.length === 0 && (
          <p className="scene-object-empty">Empty — add a shape from the top bar.</p>
        )}
        {!sceneCollapsed && root.items.length > 0 && width > 0 && height > 0 && (
          <Tree<ArboristNode>
            data={arboristData}
            width={width}
            height={height}
            indent={12}
            rowHeight={32}
            openByDefault
            dndRootElement={document.body}
            renderDragPreview={SceneDragPreview}
            selection={selectedItemId ?? undefined}
            disableMultiSelection
            idAccessor="id"
            childrenAccessor={arboristChildrenAccessor}
            onMove={({ dragIds, parentId, index }) => {
              moveItems(dragIds, parentId, index);
            }}
            onRename={({ id, name }) => {
              renameItem(id, name);
            }}
            onSelect={(nodes) => syncSelection(nodes)}
            onActivate={(node) => syncSelection([node])}
            disableDrop={({ parentNode, dragNodes }) => {
              const dragIds = dragNodes.map((n) => n.id);
              const parentData = parentNode.isRoot
                ? null
                : (parentNode.data as ArboristNode);
              return !canDropAt(
                root,
                dragIds,
                parentNode.isRoot ? null : parentNode.id,
                parentData,
                parentNode.isRoot,
              );
            }}
          >
            {SceneNode}
          </Tree>
        )}
      </div>
      <div className="scene-add-object">
        <button
          type="button"
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
