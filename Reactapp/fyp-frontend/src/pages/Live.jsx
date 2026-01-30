import { useEffect, useMemo, useRef, useState } from "react";
import { createWS } from "../services/wsClient";
import ResultOverlay from "../components/ResultOverlay";

import "../styles/live.css";

export default function Live() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const timerRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState("");

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
          if (msg.type === "result") setResult(msg.payload);
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

  const start = () => {
    setErr("");
    connectWS();
    setRunning(true);

    // ~2.8 FPS (stable). If latency stays stable you can try 300ms later.
    timerRef.current = setInterval(() => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== 1) return;
      sendFrame();
    }, 350);
  };

  const stop = () => {
    setRunning(false);
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    disconnectWS();
  };

  const statusClass = connected ? "statusPill statusConnected" : "statusPill statusDisconnected";

  return (
    <div className="gridTwoCol">
      <div className="card">
        <div className="cardHeader">
          <h2 className="h2NoMargin">Live</h2>
          <div className={statusClass}>{connected ? "Connected" : "Disconnected"}</div>
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

        <button onClick={running ? stop : start} className="primaryBtn">
          {running ? "Stop" : "Start"}
        </button>

        <div style={{ marginTop: 14 }}>
          <h4 className="sectionTitle">Current Result</h4>
          <div className="smallText">
            <div><b>Person:</b> {result?.person ?? "—"}</div>
            <div><b>Face conf:</b> {result?.face_conf ?? 0}</div>
            <div><b>Gesture:</b> {result?.gesture ?? "—"}</div>
            <div><b>Gesture conf:</b> {result?.gesture_conf ?? 0}</div>
            <div><b>Latency:</b> {result?.latency_ms ?? "—"} ms</div>
            <div><b>Distance:</b> {result?.distance ?? "—"}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
