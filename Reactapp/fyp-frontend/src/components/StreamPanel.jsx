import { useEffect, useRef, useState } from "react";
import ResultOverlay from "./ResultOverlay";
import { useWebcam } from "../hooks/useWebcam";
import { useWS } from "../hooks/useWS";
import { useFrameLoop } from "../hooks/useFrameLoop";

import "../styles/live.css";

export default function StreamPanel({
  title,
  wsUrl = "ws://localhost:8000/ws",
  timed = false,
  durationSec = 15,
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const { ready: webcamReady, error: camError } = useWebcam(videoRef);
  const { connected, error: wsError, lastMessage, sendJson } = useWS(wsUrl);

  const [running, setRunning] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(durationSec);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState("");

  // EXACT backend frame format
  useFrameLoop({
    enabled: running && connected && webcamReady,
    videoRef,
    canvasRef,
    sendJson,
    intervalMs: 350,
    frameOptions: { width: 640, height: 360, quality: 0.6 },
    messageBuilder: (b64) => ({ type: "frame", data: b64 }),
  });

  // Robust result parsing (supports both payload and non-payload formats)
  useEffect(() => {
    if (!lastMessage) return;

    if (lastMessage.type === "result") {
      setResult(lastMessage.payload ?? lastMessage);
    }
  }, [lastMessage]);

  useEffect(() => {
    if (wsError) setErr("WebSocket error (is backend running on :8000?)");
  }, [wsError]);

  useEffect(() => {
    if (camError) setErr(camError);
  }, [camError]);

  // Timed session countdown (only when timed=true)
  useEffect(() => {
    if (!timed || !running) return;

    if (secondsLeft <= 0) {
      setRunning(false);
      return;
    }

    const t = setTimeout(() => setSecondsLeft((s) => s - 1), 1000);
    return () => clearTimeout(t);
  }, [timed, running, secondsLeft]);

  const start = () => {
    setErr("");
    if (timed) setSecondsLeft(durationSec);
    setRunning(true);
  };

  const stop = () => setRunning(false);

  const statusClass = connected
    ? "statusPill statusConnected"
    : "statusPill statusDisconnected";

  return (
    <div className="gridTwoCol">
      <div className="card">
        <div className="cardHeader">
          <h2 className="h2NoMargin">{title}</h2>
          <div className={statusClass}>
            {connected ? "Connected" : "Disconnected"}
          </div>
        </div>

        <div className="videoWrap">
          <video ref={videoRef} autoPlay playsInline muted className="video" />
          <ResultOverlay result={result} />
        </div>

        <canvas ref={canvasRef} style={{ display: "none" }} />

        {err ? <p className="errorText">{err}</p> : null}
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Controls</h3>

        <button
          onClick={running ? stop : start}
          className="primaryBtn"
          disabled={!connected || !webcamReady}
        >
          {running ? "Stop" : timed ? `Start ${durationSec}s Session` : "Start"}
        </button>

        {timed ? (
          <div style={{ marginTop: 14 }} className="smallText">
            <div>
              <b>Seconds left:</b> {running ? secondsLeft : "—"}
            </div>
          </div>
        ) : null}

        <div style={{ marginTop: 14 }}>
          <h4 className="sectionTitle">Current Result</h4>
          <div className="smallText">
            <div>
              <b>Person:</b> {result?.person ?? "—"}
            </div>
            <div>
              <b>Face conf:</b> {result?.face_conf ?? 0}
            </div>
            <div>
              <b>Gesture:</b> {result?.gesture ?? "—"}
            </div>
            <div>
              <b>Gesture conf:</b> {result?.gesture_conf ?? 0}
            </div>
            <div>
              <b>Latency:</b> {result?.latency_ms ?? "—"} ms
            </div>
            <div>
              <b>Distance:</b> {result?.distance ?? "—"}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
