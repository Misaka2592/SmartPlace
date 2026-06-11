import { useEffect, useMemo, useRef, useState } from "react";

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export default function DragCanvas({ scene, position, onChangePosition }) {
  const stageRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [cursorPoint, setCursorPoint] = useState(null);

  const stageStyle = useMemo(() => {
    if (!scene) return {};
    return {
      width: "100%",
      aspectRatio: `${scene.background.width} / ${scene.background.height}`
    };
  }, [scene]);

  useEffect(() => {
    if (!dragging) return;
    const handlePointerUp = () => setDragging(false);
    window.addEventListener("pointerup", handlePointerUp);
    return () => window.removeEventListener("pointerup", handlePointerUp);
  }, [dragging]);

  if (!scene) {
    return <div className="canvas-empty">先上传素材，然后点击“加载画布”进入操作状态。</div>;
  }

  const bg = scene.background;
  const fg = scene.foreground;
  const init = position || scene.initial_state;

  function pointerToImageCoords(event) {
    const rect = stageRef.current.getBoundingClientRect();
    const scaleX = bg.width / rect.width;
    const scaleY = bg.height / rect.height;
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY
    };
  }

  function updateCursorPoint(event) {
    const rect = stageRef.current?.getBoundingClientRect();
    if (!rect) return;
    setCursorPoint({
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    });
  }

  function beginDrag(event) {
    const p = pointerToImageCoords(event);
    setDragging(true);
    setOffset({ x: p.x - init.x, y: p.y - init.y });
  }

  function handleMove(event) {
    updateCursorPoint(event);
    if (!dragging) return;
    const p = pointerToImageCoords(event);
    const nextX = clamp(Math.round(p.x - offset.x), 0, bg.width - fg.width);
    const nextY = clamp(Math.round(p.y - offset.y), 0, bg.height - fg.height);
    onChangePosition({ ...init, x: nextX, y: nextY });
  }

  const fgStyle = {
    left: `${(init.x / bg.width) * 100}%`,
    top: `${(init.y / bg.height) * 100}%`,
    width: `${(fg.width / bg.width) * 100}%`,
    height: `${(fg.height / bg.height) * 100}%`
  };

  const cursorStyle = cursorPoint
    ? {
        left: `${cursorPoint.x}px`,
        top: `${cursorPoint.y}px`
      }
    : undefined;

  return (
    <div className="canvas-shell">
      <div
        className="canvas-stage"
        ref={stageRef}
        style={stageStyle}
        onPointerMove={handleMove}
        onPointerEnter={updateCursorPoint}
        onPointerLeave={() => setCursorPoint(null)}
      >
        <img className="canvas-bg" src={bg.url} alt="background" draggable={false} />
        {cursorPoint && <div className="canvas-target-cursor" style={cursorStyle} />}
        <div className="canvas-fg-wrap" style={fgStyle} onPointerDown={beginDrag}>
          <img className="canvas-fg" src={fg.url} alt="foreground" draggable={false} />
          <div className="canvas-box" />
        </div>
      </div>
    </div>
  );
}
