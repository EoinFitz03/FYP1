import { useEffect, useRef, useState } from "react";
import { useWebcam } from "../hooks/useWebcam";
import { useWS } from "../hooks/useWS";
import { useFrameLoop } from "../hooks/useFrameLoop";
import { buildEnrolStart, buildEnrolCancel, buildFrameMessage } from "../services/protocol";

import "../styles/live.css";

export default function Enroll() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const wsUrl = "ws://localhost:8000/ws";

  const { ready: webcamReady, error: camError } = useWebcam(videoRef);
  const { connected, error: wsError, lastMessage, sendJson } = useWS(wsUrl);

  const [err, setErr] = useState("");
  const [running, setRunning] = useState(false);

  const [name, setName] = useState("");
  const [target, setTarget] = useState(10);

  const [enrol, setEnrol] = useState({
    active: false,
    captured: 0,
    target: 0,
    done: false,
    error: null,
  });

  // Send frames ONLY while enrol is running (same behaviour as old code)
  useFrameLoop({
    enabled: running && connected && webcamReady,
    videoRef,
    canvasRef,
    sendJson,
    intervalMs: 350,
    frameOptions: { width: 640, height: 360, quality: 0.6 },
    messageBuilder: (b64) => buildFrameMessage(b64), // { type:"frame", data:b64 }
  });

  // Handle backend enrol status messages (THIS is the key!)
  useEffect(() => {
    if (!lastMessage) return;

    try {
      // lastMessage is already parsed JSON in useWS
      const msg = lastMessage;

      if (msg.type === "enrol_status") {
        const p = msg.payload || {};

        setEnrol({
          active: !!p.active,
          captured: p.captured ?? 0,
          target: p.target ?? 0,
          done: !!p.done,
          error: p.error ?? null,
        });

        // Auto-stop sending frames once finished (same as old)
        if (p.done) setRunning(false);
      }
    } catch {
      // ignore
    }
  }, [lastMessage]);

  useEffect(() => {
    if (wsError) setErr("WebSocket error (is backend running on :8000?)");
  }, [wsError]);

  useEffect(() => {
    if (camError) setErr(camError);
  }, [camError]);

  const startEnroll = () => {
    setErr("");

    const cleanName = name.trim();
    const n = Math.max(3, Math.min(30, Number(target) || 10));

    setEnrol({ active: false, captured: 0, target: 0, done: false, error: null });

    // Send enrol_start EXACTLY like old backend expects
    sendJson(buildEnrolStart(cleanName, n));

    // Start streaming frames
    setRunning(true);
  };

  const cancelEnroll = () => {
    // Tell backend to cancel, stop frames
    sendJson(buildEnrolCancel());
    setRunning(false);
  };

  const statusClass = connected ? "statusPill statusConnected" : "statusPill statusDisconnected";

  return (
    <div className="gridTwoCol">
      <div className="card">
        <div className="cardHeader">
          <h2 className="h2NoMargin">Enroll</h2>
          <div className={statusClass}>{connected ? "Connected" : "Disconnected"}</div>
        </div>

        <div className="videoWrap">
          <video ref={videoRef} autoPlay playsInline muted className="video" />
        </div>

        <canvas ref={canvasRef} style={{ display: "none" }} />
        {err ? <p className="errorText">{err}</p> : null}
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Enroll Controls</h3>

        <div style={{ display: "grid", gap: 10, maxWidth: 360 }}>
          <label className="smallText">
            <b>Name</b>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Eoin"
              style={{
                width: "100%",
                marginTop: 6,
                padding: 10,
                borderRadius: 10,
                border: "1px solid #222",
              }}
              disabled={running}
            />
          </label>

          <label className="smallText">
            <b>Samples</b>
            <input
              type="number"
              value={target}
              onChange={(e) => setTarget(e.target.value)}
              min={3}
              max={30}
              style={{
                width: "100%",
                marginTop: 6,
                padding: 10,
                borderRadius: 10,
                border: "1px solid #222",
              }}
              disabled={running}
            />
          </label>

          <button
            onClick={startEnroll}
            className="primaryBtn"
            disabled={running || !name.trim() || !connected || !webcamReady}
          >
            Start Enroll
          </button>

          <button
            onClick={cancelEnroll}
            className="primaryBtn"
            disabled={!running && !connected}
          >
            Cancel / Stop
          </button>
        </div>

        <div style={{ marginTop: 14 }}>
          <h4 className="sectionTitle">Status</h4>
          <div className="smallText">
            <div><b>Active:</b> {enrol.active ? "Yes" : "No"}</div>
            <div><b>Captured:</b> {enrol.captured} / {enrol.target || "—"}</div>
            <div><b>Done:</b> {enrol.done ? "Yes" : "No"}</div>
            <div><b>Error:</b> {enrol.error ?? "—"}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
