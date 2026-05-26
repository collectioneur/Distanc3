import { useEffect, type RefObject } from "react";
import tgpu, { d } from "typegpu";
import { createShader, MAX_SHAPES, SHAPE_TYPE_INT } from "./shader";
import { useSceneStore } from "../store/sceneStore";
import { useRenderStore } from "../store/renderStore";

const emptyShapeEntry = {
  shapeType: 0,
  position: d.vec3f(0, 0, 0),
  params: d.vec4f(0, 0, 0, 0),
};

export function useGpu(canvasRef: RefObject<HTMLCanvasElement | null>) {
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let cancelled = false;
    let animFrameId = 0;
    let registeredCleanup: (() => void) | undefined;

    tgpu.init().then((root) => {
      if (cancelled) {
        root.destroy();
        return;
      }

      const context = root.configureContext({
        canvas,
        alphaMode: "premultiplied",
      });

      const {
        pipeline,
        timeUniform,
        aspectUniform,
        mouseUniform,
        shapesBuffer,
        shapeCountUniform,
        renderModeUniform,
      } = createShader(root);

      let isDragging = false;
      let dragStartX = 0;
      let dragStartY = 0;
      let rotX = 0.3;
      let rotY = -0.4;

      const onMouseDown = (e: MouseEvent) => {
        isDragging = true;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
      };

      const onMouseMove = (e: MouseEvent) => {
        if (!isDragging) return;
        const dx =
          ((e.clientX - dragStartX) / window.innerWidth) * Math.PI * 2;
        const dy =
          ((e.clientY - dragStartY) / window.innerHeight) * Math.PI * 2;
        mouseUniform.write(d.vec2f(rotX + dx, rotY + dy));
      };

      const onMouseUp = (e: MouseEvent) => {
        if (!isDragging) return;
        isDragging = false;
        rotX += ((e.clientX - dragStartX) / window.innerWidth) * Math.PI * 2;
        rotY += ((e.clientY - dragStartY) / window.innerHeight) * Math.PI * 2;
      };

      canvas.addEventListener("mousedown", onMouseDown);
      window.addEventListener("mousemove", onMouseMove);
      window.addEventListener("mouseup", onMouseUp);

      function updateSize() {
        const dpr = window.devicePixelRatio ?? 1;
        canvas!.width = canvas!.clientWidth * dpr;
        canvas!.height = canvas!.clientHeight * dpr;
        if (canvas!.height > 0) {
          aspectUniform.write(canvas!.width / canvas!.height);
        }
      }

      updateSize();

      const observer = new ResizeObserver(updateSize);
      observer.observe(canvas);

      const startTime = performance.now();

      function frame() {
        const { shapes } = useSceneStore.getState();

        const shapeEntries = shapes.map((s) => ({
          shapeType: SHAPE_TYPE_INT[s.type],
          position: d.vec3f(s.position[0], s.position[1], s.position[2]),
          params: d.vec4f(s.params[0], s.params[1], s.params[2], s.params[3]),
        }));

        const paddingCount = MAX_SHAPES - shapeEntries.length;
        const paddedShapes = [
          ...shapeEntries,
          ...Array.from({ length: paddingCount }, () => emptyShapeEntry),
        ];

        shapesBuffer.write(paddedShapes);
        shapeCountUniform.write(shapes.length);
        renderModeUniform.write(useRenderStore.getState().renderMode);

        timeUniform.write((performance.now() - startTime) / 1000);
        pipeline.withColorAttachment({ view: context }).draw(3);
        animFrameId = requestAnimationFrame(frame);
      }

      animFrameId = requestAnimationFrame(frame);

      registeredCleanup = () => {
        observer.disconnect();
        cancelAnimationFrame(animFrameId);
        canvas.removeEventListener("mousedown", onMouseDown);
        window.removeEventListener("mousemove", onMouseMove);
        window.removeEventListener("mouseup", onMouseUp);
        root.destroy();
      };
    });

    return () => {
      cancelled = true;
      cancelAnimationFrame(animFrameId);
      registeredCleanup?.();
    };
    // canvasRef is a stable ref object — safe to omit from deps
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}
