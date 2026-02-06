import { useEffect, useMemo, useRef, useState } from "react";
import { createWS } from "../services/wsClient";

import "../styles/live.css";

export default function Enroll() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const timerRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState("");

  const [name, setName] = useState("");
  const [target, setTarget] = useState(10);

  const [enrol, setEnrol] = useState({
    active: false,
    captured: 0,
    target: 0,
    done: false,
    error: null,
  });

  const wsUrl = useMemo(() => "ws://localhost:8000/ws", []);

  useEffect(() => {
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch {
        setErr("Could not access webcam. Check permissions.");
      }
    })();

    return () => {
      const v = videoRef.current;
      const s = v?.srcObject;
      if (s && typeof s.getTracks === "function") s.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const connectWS = () => {
    if (wsRef.current) return;

    wsRef.current = createWS(wsUrl, {
      onOpen: () => setConnected(true),
      onClose: () => setConnected(false),
      onError: () => setErr("WebSocket error (is backend running on :8000?)"),
      onMessage: (data) => {
        try {
          const msg = JSON.parse(data);
          if (msg.type === "enrol_status") {
            setEnrol({
              active: !!msg.payload?.active,
              captured: msg.payload?.captured ?? 0,
              target: msg.payload?.target ?? 0,
              done: !!msg.payload?.done,
              error: msg.payload?.error ?? null,
            });

            // Auto stop sending frames once finished
            if (msg.payload?.done) stopSendingFramesOnly();
          }
        } catch {
          // ignore
        }
      },
    });
  };

  const disconnectWS = () => {
    if (wsRef.current) wsRef.current.close();
    wsRef.current = null;
    setConnected(false);
  };

  const sendFrame = () => {
    const ws = wsRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!ws || ws.readyState !== 1 || !video || !canvas) return;
    if (video.readyState < 2) return;

    const w = 640;
    const h = 360;

    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, w, h);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.6);
    const base64 = dataUrl.split(",")[1];

    ws.send(JSON.stringify({ type: "frame", data: base64 }));
  };

  const startSendingFrames = () => {
    if (timerRef.current) return;

    setRunning(true);
    timerRef.current = setInterval(() => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== 1) return;
      sendFrame();
    }, 350);
  };

  const stopSendingFramesOnly = () => {
    setRunning(false);
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
  };

  const startEnroll = () => {
    setErr("");
    setEnrol({ active: false, captured: 0, target: 0, done: false, error: null });

    connectWS();

    const ws = wsRef.current;
    if (!ws) return;

    const cleanName = name.trim();
    const n = Math.max(3, Math.min(30, Number(target) || 10));

    ws.send(JSON.stringify({ type: "enrol_start", name: cleanName, num_samples: n }));
    startSendingFrames();
  };

  const cancelEnroll = () => {
    const ws = wsRef.current;
    if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: "enrol_cancel" }));
    stopSendingFramesOnly();
    disconnectWS();
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
              style={{ width: "100%", marginTop: 6, padding: 10, borderRadius: 10, border: "1px solid #222" }}
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
              style={{ width: "100%", marginTop: 6, padding: 10, borderRadius: 10, border: "1px solid #222" }}
              disabled={running}
            />
          </label>

          <button onClick={startEnroll} className="primaryBtn" disabled={running || !name.trim()}>
            Start Enroll
          </button>

          <button onClick={cancelEnroll} className="primaryBtn" disabled={!running && !connected}>
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
