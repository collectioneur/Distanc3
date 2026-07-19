import { useCallback, useRef, useState } from "react";
import { Signal } from "lucide-react";
import { useRenderStore } from "../../store/renderStore";
import {
  QUALITY_PRESETS,
  nearestStop,
  stopFraction,
} from "../../utils/quality";

function fractionFromClientX(clientX: number, rect: DOMRect): number {
  if (rect.width <= 0) return 0;
  return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
}

export default function QualityPanel() {
  const quality = useRenderStore((s) => s.quality);
  const setQuality = useRenderStore((s) => s.setQuality);
  const trackRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState(false);
  const [hovering, setHovering] = useState(false);
  const [dragFraction, setDragFraction] = useState<number | null>(null);

  const applyFraction = useCallback(
    (fraction: number) => {
      const nearest = nearestStop(fraction);
      if (nearest !== useRenderStore.getState().quality) {
        setQuality(nearest);
      }
    },
    [setQuality],
  );

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const rect = trackRef.current?.getBoundingClientRect();
    if (!rect) return;

    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(true);
    const fraction = fractionFromClientX(e.clientX, rect);
    setDragFraction(fraction);
    applyFraction(fraction);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragging) return;
    const rect = trackRef.current?.getBoundingClientRect();
    if (!rect) return;

    const fraction = fractionFromClientX(e.clientX, rect);
    setDragFraction(fraction);
    applyFraction(fraction);
  };

  const endDrag = () => {
    setDragging(false);
    setDragFraction(null);
  };

  const onPointerUp = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragging) return;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
    endDrag();
  };

  const displayFraction =
    dragFraction ?? stopFraction(quality, QUALITY_PRESETS.length);
  const displayPercent = displayFraction * 100;
  const activePreset = QUALITY_PRESETS[quality];
  const showTooltip = hovering || dragging;

  return (
    <div className="floating-pill floating-pill--top-left">
      <Signal size={14} className="floating-pill-icon" aria-hidden />
      <div
        ref={trackRef}
        className="quality-track"
        role="slider"
        aria-label="Render quality"
        aria-valuemin={0}
        aria-valuemax={QUALITY_PRESETS.length - 1}
        aria-valuenow={quality}
        aria-valuetext={activePreset.label}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerEnter={() => setHovering(true)}
        onPointerLeave={() => {
          setHovering(false);
        }}
      >
        <div className="quality-track-line" aria-hidden />
        <div
          className="quality-track-fill"
          style={{ width: `${displayPercent}%` }}
          aria-hidden
        />
        {QUALITY_PRESETS.map((preset, index) => (
          <span
            key={preset.id}
            className="quality-dot"
            style={{ left: `${stopFraction(index, QUALITY_PRESETS.length) * 100}%` }}
            aria-hidden
          />
        ))}
        <div
          className={`quality-thumb${dragging ? " quality-thumb--dragging" : ""}`}
          style={{ left: `${displayPercent}%` }}
          aria-hidden
        />
        <div
          className={`quality-tooltip${showTooltip ? " quality-tooltip--visible" : ""}`}
          style={{ left: `${displayPercent}%` }}
        >
          {activePreset.label}
        </div>
      </div>
    </div>
  );
}
