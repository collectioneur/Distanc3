import { createShader } from "./shader";

/**
 * TypeGPU Runtime Inspector entry.
 * MCP (`inspect_typegpu` / module) and `npm run inspect:gpu` both hit this.
 * Construction stays inside `create` so attribution sees the real pipeline.
 */
export async function inspect({
  root,
}: {
  root: Parameters<typeof createShader>[0];
}) {
  return {
    kind: "render-pipeline" as const,
    create: () => createShader(root).pipeline,
  };
}
