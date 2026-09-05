import {
  findContainer,
  findItem,
  getContainerDepth,
  maxGroupDepth,
  MAX_TRANSFORM_DEPTH,
  wouldExceedDepth,
  type ObjectGroup,
  type OpType,
  type SceneItem,
  type SceneRoot,
  type ShapeType,
} from "../store/sceneStore";

export type ArboristNode = {
  id: string;
  kind: "layer" | "group" | "root";
  name: string;
  shapeType?: ShapeType;
  op?: OpType;
  children?: ArboristNode[];
};

export function sceneItemToArborist(item: SceneItem): ArboristNode {
  if (item.kind === "layer") {
    return {
      id: item.id,
      kind: "layer",
      name: item.name,
      shapeType: item.shapeType,
      op: item.op,
    };
  }
  return {
    id: item.id,
    kind: "group",
    name: item.name,
    op: item.op,
    children:
      item.items.length > 0 ? item.items.map(sceneItemToArborist) : [],
  };
}

export function rootItemsToArborist(items: SceneItem[]): ArboristNode[] {
  return items.map(sceneItemToArborist);
}

export function sceneRootToArborist(root: SceneRoot): ArboristNode[] {
  return [
    {
      id: root.id,
      kind: "root",
      name: root.name,
      children: root.items.map(sceneItemToArborist),
    },
  ];
}

export function arboristChildrenAccessor(node: ArboristNode): readonly ArboristNode[] | null {
  if (node.kind === "group" || node.kind === "root") return node.children ?? [];
  return null;
}

/** True if `nodeId` is in the subtree rooted at `ancestorId` (excluding ancestor itself). */
export function isDescendant(root: SceneRoot, ancestorId: string, nodeId: string): boolean {
  if (ancestorId === nodeId) return false;

  const ancestor = findItem(root, ancestorId);
  if (!ancestor || ancestor.item.kind !== "group") return false;

  function search(items: SceneItem[]): boolean {
    for (const item of items) {
      if (item.id === nodeId) return true;
      if (item.kind === "group" && search(item.items)) return true;
    }
    return false;
  }

  return search(ancestor.item.items);
}

function removeItemFromRoot(
  root: SceneRoot,
  itemId: string,
): { root: SceneRoot; item: SceneItem } | null {
  const found = findItem(root, itemId);
  if (!found) return null;

  const newRoot = mapContainerItems(root, found.container.id, (items) =>
    items.filter((item) => item.id !== itemId),
  );
  return { root: newRoot, item: found.item };
}

function mapContainerItems(
  root: SceneRoot,
  containerId: string,
  mapItems: (items: SceneItem[]) => SceneItem[],
): SceneRoot {
  if (root.id === containerId) {
    return { ...root, items: mapItems(root.items) };
  }
  return {
    ...root,
    items: root.items.map((item) =>
      item.kind === "group" ? mapGroupItems(item, containerId, mapItems) : item,
    ),
  };
}

function mapGroupItems(
  group: ObjectGroup,
  containerId: string,
  mapItems: (items: SceneItem[]) => SceneItem[],
): ObjectGroup {
  if (group.id === containerId) {
    return { ...group, items: mapItems(group.items) };
  }
  return {
    ...group,
    items: group.items.map((item) =>
      item.kind === "group" ? mapGroupItems(item, containerId, mapItems) : item,
    ),
  };
}

function insertIntoContainer(
  root: SceneRoot,
  containerId: string,
  index: number,
  item: SceneItem,
): SceneRoot {
  return mapContainerItems(root, containerId, (items) => {
    const next = [...items];
    next.splice(index, 0, item);
    return next;
  });
}

function validateMoveDepth(
  root: SceneRoot,
  targetParentId: string,
  item: SceneItem,
): boolean {
  if (wouldExceedDepth(root, targetParentId)) return false;
  if (item.kind === "layer") return true;

  const parentDepth = getContainerDepth(root, targetParentId);
  if (parentDepth < 0) return false;
  const groupContainerDepth = parentDepth + 1;
  const innerMax = maxGroupDepth(item.items);
  return groupContainerDepth + innerMax <= MAX_TRANSFORM_DEPTH;
}

export type MoveRejectReason = "cycle" | "depth" | "invalid_parent" | "not_found";

export function applyMoveToRoot(
  root: SceneRoot,
  dragIds: string[],
  parentId: string | null,
  index: number,
): { root: SceneRoot } | { reason: MoveRejectReason } {
  const targetParentId = parentId ?? root.id;

  if (!findContainer(root, targetParentId)) {
    return { reason: "invalid_parent" };
  }

  for (const dragId of dragIds) {
    if (dragId === targetParentId) return { reason: "cycle" };
    if (isDescendant(root, dragId, targetParentId)) return { reason: "cycle" };
  }

  let newRoot = root;
  const moved: SceneItem[] = [];
  const sourceIndices: { parentId: string; index: number }[] = [];

  for (const dragId of dragIds) {
    const found = findItem(newRoot, dragId);
    if (!found) return { reason: "not_found" };
    sourceIndices.push({ parentId: found.container.id, index: found.index });

    const removed = removeItemFromRoot(newRoot, dragId);
    if (!removed) return { reason: "not_found" };
    newRoot = removed.root;
    moved.push(removed.item);
  }

  let insertIndex = index;
  if (
    dragIds.length === 1 &&
    sourceIndices[0].parentId === targetParentId &&
    sourceIndices[0].index < index
  ) {
    insertIndex = index - 1;
  }

  for (let i = 0; i < moved.length; i++) {
    const item = moved[i];
    if (!validateMoveDepth(newRoot, targetParentId, item)) {
      return { reason: "depth" };
    }
    newRoot = insertIntoContainer(newRoot, targetParentId, insertIndex + i, item);
  }

  return { root: newRoot };
}

export function canDropAt(
  root: SceneRoot,
  dragIds: string[],
  parentId: string | null,
  parentData: ArboristNode | null,
  isRootParent: boolean,
): boolean {
  const targetParentId = parentId ?? root.id;

  if (isRootParent) return false;
  if (parentData?.kind === "layer") return false;
  if (!findContainer(root, targetParentId)) return false;

  for (const dragId of dragIds) {
    if (dragId === targetParentId) return false;
    if (isDescendant(root, dragId, targetParentId)) return false;
  }

  if (wouldExceedDepth(root, targetParentId)) return false;

  for (const dragId of dragIds) {
    const found = findItem(root, dragId);
    if (!found) return false;
    if (!validateMoveDepth(root, targetParentId, found.item)) return false;
  }

  return true;
}
