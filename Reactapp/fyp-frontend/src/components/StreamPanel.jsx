import { useEffect, useRef, useState } from "react";
import ResultOverlay from "./ResultOverlay";
import { useWebcam } from "../hooks/useWebcam";
import { useWS } from "../hooks/useWS";
import { useFrameLoop } from "../hooks/useFrameLoop";

import "../styles/live.css";

// Normalise gesture text so "ThumbsUp", "thumbs_up", "thumbs up" all match.
const norm = (s) =>
  String(s || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "")
    .replace(/_/g, "");

export default function StreamPanel({
  title,
  wsUrl = "ws://localhost:8000/ws",
  timed = false,
  durationSec = 15,

  // Door simulation props (used by Simulation)
  doorSim = false,

  // Defaults match your backend values
  openGesture = "thumbs_up",
  closeGesture = "open_palm",
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const { ready: webcamReady, error: camError } = useWebcam(videoRef);
  const { connected, error: wsError, lastMessage, sendJson } = useWS(wsUrl);

  const [running, setRunning] = useState(false);
  const [secondsLeft, setSecondsLeft] = useState(durationSec);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState("");

  // --- NEW: Training UI state ---
  const [trainLabel, setTrainLabel] = useState("open_palm");
  const [trainSamples, setTrainSamples] = useState(200);
  const [trainStatus, setTrainStatus] = useState({
    active: false,
    label: null,
    count: 0,
    target: 0,
    saved: false,
    finished: false,
    error: null,
  });

  // Door state for simulation text
  const [doorState, setDoorState] = useState("CLOSED"); // CLOSED | OPENING | OPEN | CLOSING
  const doorTimeoutRef = useRef(null);

  // Send frames when running
  useFrameLoop({
    enabled: running && connected && webcamReady,
    videoRef,
    canvasRef,
    sendJson,
    intervalMs: 350,
    frameOptions: { width: 640, height: 360, quality: 0.6 },
    // EXACT backend format
    messageBuilder: (b64) => ({ type: "frame", data: b64 }),
  });

  // Parse results from backend
  useEffect(() => {
    if (!lastMessage) return;

    if (lastMessage.type === "result") {
      setResult(lastMessage.payload ?? lastMessage);
      return;
    }

    // --- NEW: Training status messages from backend ---
    if (lastMessage.type === "train_status") {
      const p = lastMessage.payload ?? {};
      setTrainStatus((prev) => ({
        ...prev,
        ...p,
      }));
      return;
    }
  }, [lastMessage]);

  // Errors
  useEffect(() => {
    if (wsError) setErr("WebSocket error (is backend running on :8000?)");
  }, [wsError]);

  useEffect(() => {
    if (camError) setErr(camError);
  }, [camError]);

  // Timed session countdown (simulation mode)
  useEffect(() => {
    if (!timed || !running) return;

    if (secondsLeft <= 0) {
      setRunning(false);
      return;
    }

    const t = setTimeout(() => setSecondsLeft((s) => s - 1), 1000);
    return () => clearTimeout(t);
  }, [timed, running, secondsLeft]);

  // Door logic driven by gestures (only when enabled)
  useEffect(() => {
    if (!doorSim) return;
    if (!running) return;
    if (!result) return;

    // Support both possible keys
    const rawGesture = result.gesture ?? result.hand_gesture ?? "";
    const g = norm(rawGesture);
    if (!g) return;

    // Clear any previous transition timer
    if (doorTimeoutRef.current) {
      clearTimeout(doorTimeoutRef.current);
      doorTimeoutRef.current = null;
    }

    const openKey = norm(openGesture);
    const closeKey = norm(closeGesture);

    // Thumbs up => OPEN
    if (g === openKey) {
      setDoorState("OPENING");
      doorTimeoutRef.current = setTimeout(() => {
        setDoorState("OPEN");
        doorTimeoutRef.current = null;
      }, 700);
      return;
    }

    // Open palm => CLOSE
    if (g === closeKey) {
      setDoorState("CLOSING");
      doorTimeoutRef.current = setTimeout(() => {
        setDoorState("CLOSED");
        doorTimeoutRef.current = null;
      }, 700);
      return;
    }
  }, [doorSim, running, result, openGesture, closeGesture]);

  // Reset door when session stops
  useEffect(() => {
    if (!doorSim) return;

    if (!running) {
      if (doorTimeoutRef.current) {
        clearTimeout(doorTimeoutRef.current);
        doorTimeoutRef.current = null;
      }
      setDoorState("CLOSED");
    }
  }, [doorSim, running]);

  const start = () => {
    setErr("");
    if (timed) setSecondsLeft(durationSec);
    setRunning(true);
  };

  const stop = () => setRunning(false);

  // --- NEW: Training button actions ---
  const startTraining = () => {
    setErr("");
    const label = String(trainLabel || "").trim();
    const num = Number(trainSamples) || 200;

    if (!label) {
      setErr("Training label is empty (e.g. open_palm)");
      return;
    }

    const ok = sendJson({
      type: "train_start",
      label,
      num_samples: num,
    });

    if (!ok) setErr("Could not send train_start (is WS connected?)");
  };

  const stopTraining = () => {
    const ok = sendJson({ type: "train_stop" });
    if (!ok) setErr("Could not send train_stop (is WS connected?)");
  };

  const statusClass = connected
    ? "statusPill statusConnected"
    : "statusPill statusDisconnected";

  const doorText =
    doorState === "OPEN"
      ? "DOOR OPEN ✅"
      : doorState === "OPENING"
      ? "DOOR OPENING..."
      : doorState === "CLOSING"
      ? "DOOR CLOSING..."
      : "DOOR CLOSED 🔒";

  const trainText = trainStatus.active
    ? `RECORDING: ${trainStatus.label} (${trainStatus.count}/${trainStatus.target})`
    : trainStatus.finished
    ? `FINISHED: ${trainStatus.count}/${trainStatus.target}`
    : "NOT RECORDING";

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

        {/* =======================
            NEW: Training Controls
           ======================= */}
        <div style={{ marginTop: 18 }}>
          <h4 className="sectionTitle">Training (Dataset Capture)</h4>

          <div className="smallText" style={{ marginBottom: 10, opacity: 0.9 }}>
            <div>
              <b>Status:</b> {trainText}
            </div>
            <div style={{ opacity: 0.85 }}>
              Records MediaPipe hand points into <b>backend/dataset/gestures.csv</b>
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
            <label className="smallText" style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <b>Label</b>
              <input
                value={trainLabel}
                onChange={(e) => setTrainLabel(e.target.value)}
                placeholder="open_palm"
                style={{ padding: "6px 8px", minWidth: 140 }}
                disabled={!connected}
              />
            </label>

            <label className="smallText" style={{ display: "flex", gap: 6, alignItems: "center" }}>
              <b>Samples</b>
              <input
                type="number"
                value={trainSamples}
                onChange={(e) => setTrainSamples(e.target.value)}
                min={10}
                max={5000}
                style={{ padding: "6px 8px", width: 90 }}
                disabled={!connected}
              />
            </label>
          </div>

          <div style={{ display: "flex", gap: 10, marginTop: 10, flexWrap: "wrap" }}>
            <button
              onClick={startTraining}
              className="primaryBtn"
              disabled={!connected || !running || !webcamReady}
              title={!running ? "Start the session first so frames are sending" : ""}
            >
              Start Training
            </button>

            <button
              onClick={stopTraining}
              className="secondaryBtn"
              disabled={!connected}
            >
              Stop Training
            </button>
          </div>

          {!running ? (
            <div className="smallText" style={{ marginTop: 8, opacity: 0.8 }}>
              Tip: click <b>Start</b> first (so frames are streaming), then click <b>Start Training</b>.
            </div>
          ) : null}
        </div>

        {doorSim ? (
          <div style={{ marginTop: 16 }}>
            <h4 className="sectionTitle">Door State</h4>
            <div style={{ fontSize: 18, fontWeight: 700 }}>{doorText}</div>

            <div className="smallText" style={{ marginTop: 8, opacity: 0.85 }}>
              <div>
                <b>Detected gesture:</b>{" "}
                {result?.gesture ?? result?.hand_gesture ?? "—"}
              </div>
              <div>
                <b>thumbs_up</b> = Open
              </div>
              <div>
                <b>open_palm</b> = Close
              </div>
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