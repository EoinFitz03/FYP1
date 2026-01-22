import "../styles/overlay.css";

export default function ResultOverlay({ result }) {
  if (!result) {
    return (
      <div className="overlayBox">
        <div><b>Person:</b> —</div>
        <div><b>Gesture:</b> —</div>
        <div><b>Status:</b> waiting…</div>
      </div>
    );
  }

  return (
    <div className="overlayBox">
      <div><b>Person:</b> {result.person} ({Math.round((result.face_conf ?? 0) * 100)}%)</div>
      <div><b>Gesture:</b> {result.gesture} ({Math.round((result.gesture_conf ?? 0) * 100)}%)</div>
      <div><b>Latency:</b> {result.latency_ms} ms</div>
    </div>
  );
}
