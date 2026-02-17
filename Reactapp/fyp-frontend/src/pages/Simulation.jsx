import { useEffect, useMemo, useRef, useState } from "react";
import { createWS } from "../services/wsClient";
import "../styles/live.css";

export default function Simulation() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const timerRef = useRef(null);
  const countdownRef = useRef(null);

  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [err, setErr] = useState("");

  const [secondsLeft, setSecondsLeft] = useState(0);

  const [result, setResult] = useState({
    person: "—",
    face_conf: 0,
    gesture: "—",
    gesture_conf: 0,
    latency_ms: 0,
  });

  const wsUrl = useMemo(() => "ws://localhost:8000/ws", []);

  // --- Webcam ---
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

  // --- WS helpers ---
  const connectWS = () => {
    if (wsRef.current) return;

    wsRef.current = createWS(wsUrl, {
      onOpen: () => setConnected(true),
      onClose: () => setConnected(false),
      onError: () => setErr("WebSocket error (is backend running on :8000?)"),
      onMessage: (data) => {
        try {
          const msg = JSON.parse(data);
          if (msg.type === "result" && msg.payload) {
            setResult({
              person: msg.payload.person ?? "—",
              face_conf: msg.payload.face_conf ?? 0,
              gesture: msg.payload.gesture ?? "—",
              gesture_conf: msg.payload.gesture_conf ?? 0,
              latency_ms: msg.payload.latency_ms ?? 0,
            });
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

  // --- Frame sender (same approach as Live/Enroll) ---
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

  const stopSession = () => {
    setRunning(false);
    setSecondsLeft(0);

    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;

    if (countdownRef.current) clearInterval(countdownRef.current);
    countdownRef.current = null;
  };

  // --- Doorbell session (15s) ---
  const pressDoorbell = () => {
    setErr("");
    connectWS();

    // Start 15 second window
    const totalSeconds = 15;
    setSecondsLeft(totalSeconds);
    setRunning(true);

    // Send frames at a reasonable rate (keep your backend happy)
    const frameIntervalMs = 350;

    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = setInterval(() => {
      sendFrame();
    }, frameIntervalMs);

    // Countdown timer
    if (countdownRef.current) clearInterval(countdownRef.current);
    countdownRef.current = setInterval(() => {
      setSecondsLeft((s) => {
        if (s <= 1) {
          stopSession();
          return 0;
        }
        return s - 1;
      });
    }, 1000);
  };

  // Cleanup
  useEffect(() => {
    return () => {
      stopSession();
      disconnectWS();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const statusClass = connected ? "statusPill statusConnected" : "statusPill statusDisconnected";
  const sessionText = running ? `SCANNING… (${secondsLeft}s left)` : "IDLE (Press Doorbell)";

  return (
    <div className="gridTwoCol">
      <div className="card">
        <div className="cardHeader">
          <h2 className="h2NoMargin">Simulation (Doorbell Session)</h2>
          <div className={statusClass}>{connected ? "Connected" : "Disconnected"}</div>
        </div>

        <div className="videoWrap">
          <video ref={videoRef} autoPlay playsInline muted className="video" />
        </div>

        <canvas ref={canvasRef} style={{ display: "none" }} />

        <div style={{ marginTop: 10 }} className="smallText">
          <b>Status:</b> {sessionText}
        </div>

        {err ? <p className="errorText">{err}</p> : null}

        <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
          <button className="primaryBtn" onClick={pressDoorbell} disabled={running}>
            Press Doorbell (15s)
          </button>

          <button className="primaryBtn" onClick={stopSession} disabled={!running}>
            Stop
          </button>
        </div>
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Output</h3>
        <div className="smallText" style={{ display: "grid", gap: 6 }}>
          <div><b>Person:</b> {result.person}</div>
          <div><b>Face conf:</b> {result.face_conf}</div>
          <div><b>Gesture:</b> {result.gesture}</div>
          <div><b>Gesture conf:</b> {result.gesture_conf}</div>
          <div><b>Latency (ms):</b> {result.latency_ms}</div>
        </div>

        <div style={{ marginTop: 14 }} className="smallText">
          <b>Door:</b> {running ? "CLOSED (Scanning)" : "CLOSED (Idle)"}
        </div>
      </div>
    </div>
  );
}
