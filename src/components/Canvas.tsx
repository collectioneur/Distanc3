import { useRef } from "react";
import { useGpu } from "../gpu/useGpu";

export default function Canvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useGpu(canvasRef);
  return <canvas ref={canvasRef} />;
}
