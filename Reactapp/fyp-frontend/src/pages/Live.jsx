import { useEffect, useRef, useState } from "react";
import { createWS } from "../services/wsClient";
import ResultOverlay from "../components/ResultOverlay";

import "../styles/live.css";

export default function Live() {
  // Refs hold live objects that should NOT trigger re-renders when they change.
  const videoRef = useRef(null); // points at <video> element so I can assign srcObject (webcam stream) 
  const canvasRef = useRef(null); // points at a hidden canvas to screen sht video 
  const wsRef = useRef(null); // stores live webb socket 
  const timerRef = useRef(null); 

  // React state drives the UI.
  const [connected, setConnected] = useState(false); // controls the connected/disconnected state
  const [running, setRunning] = useState(false); // control start stop button for sending frames 
  const [result, setResult] = useState(null); //the latest result on teh right side 
  const [err, setErr] = useState("");

  // The backend WebSocket URL. This is the "network connection point" to your FastAPI server.
  // ws://localhost:8000/ws means: connect to my own machine on port 8000 at /ws
  const wsUrl = "ws://localhost:8000/ws";

  // This runs once when the page loads.
  // getUserMedia asks Chrome for permission and returns a live MediaStream if allowed.
  // chrome ask permission to use camera and it allows it to strean using videoref 
  useEffect(() => {
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
          audio: false,
        });

        // Attach the webcam MediaStream to the <video> element.
        // This is what makes the live webcam appear on the webpage.
        if (videoRef.current) videoRef.current.srcObject = stream; 
      } catch {
        setErr("Could not access webcam. Check permissions.");
      }
    })();

    // Cleanup runs when you leave the page / component unmounts.
    // Stops the webcam tracks so the camera turns off properly.
    // this avoids bugs of camera staying on whenit should not be 
    return () => {
      const v = videoRef.current;
      const s = v?.srcObject;
      if (s && typeof s.getTracks === "function") s.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // This creates a persistent connection to the FastAPI /ws endpoint.
  const connectWS = () => {
    if (wsRef.current) return;

    wsRef.current = createWS(wsUrl, {
      // Backend connection opened successfully
      onOpen: () => setConnected(true),

      // Backend connection closed (backend stopped, network issues, stop button)
      onClose: () => setConnected(false),

      // WebSocket errors (common: backend not running)
      onError: () => setErr("WebSocket error (is backend running on :8000?)"),

      // Messages sent FROM the backend arrive here
      // You parse JSON and store the latest result for UI + overlay.
      onMessage: (data) => {
        try {
          const msg = JSON.parse(data);
          if (msg.type === "result") setResult(msg.payload);
        } catch {
          // Ignore bad JSON, basically ignore a json that is missing a comma or a typo error 
        }
      },
    });
  };

  // Close the WebSocket connection and update UI state
  const disconnectWS = () => {
    if (wsRef.current) wsRef.current.close();
    wsRef.current = null;
    setConnected(false);
  };

/**
 *  Takes a quick screenshot of the live webcam video 
 * it draws the current frame onto a hidden canvas, then converts it to a JPEG
 * That JPEG is sent to teh websocket to the backend ,where it gets decoded and processed 
 * and teh backend sends bac a json result 
 */
  const sendFrame = () => {
    const ws = wsRef.current;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    // Safety checks must have an open socket + valid video/canvas
    if (!ws || ws.readyState !== 1 || !video || !canvas) return;

    // Ensure the video has enough data for a frame (readyState >= 2 means frame data is available)
    if (video.readyState < 2) return;

    // Resize the snapshot to a fixed size for performance and consistent processing
    const w = 640;
    const h = 360;

    // Match canvas size to the snapshot size
    canvas.width = w;
    canvas.height = h;

    // Draw current video frame into the canvas
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, w, h); //copies the frame inot the canvas 

    // Convert that canvas image to a compressed JPEG data URL
    // 0.6 is the JPEG quality 
    const dataUrl = canvas.toDataURL("image/jpeg", 0.6);

    const base64 = dataUrl.split(",")[1];
    ws.send(JSON.stringify({ type: "frame", data: base64 }));  //Extracts the base 64 and sends it 
  };

  // Start button:
  // - clears errors
  // - connects WebSocket
  // - begins repeating frame sending every 350ms (~2.8 FPS)
  const start = () => {
    setErr("");
    connectWS();
    setRunning(true);

    // Interval: on each tick, check socket is still open, then send a frame
    timerRef.current = setInterval(() => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== 1) return;
      sendFrame();
    }, 350);
  };

  // Stop button:
  // - stops the interval
  // - closes WebSocket
  // - updates UI state
  const stop = () => {
    setRunning(false);
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    disconnectWS();
  };

  // Small helper for the connected/disconnected pill styling
  const statusClass = connected ? "statusPill statusConnected" : "statusPill statusDisconnected";

  return (
    <div className="gridTwoCol">
      <div className="card">
        <div className="cardHeader">
          <h2 className="h2NoMargin">Live</h2>

          {/* Shows whether the WebSocket to backend is connected */}
          <div className={statusClass}>{connected ? "Connected" : "Disconnected"}</div>
        </div>

        <div className="videoWrap">
          {/* This <video> shows the live webcam stream (srcObject = MediaStream) */}
          <video ref={videoRef} autoPlay playsInline muted className="video" />

          {/* Overlay draws backend results on top of the video */}
          <ResultOverlay result={result} />
        </div>

        {/* Hidden canvas used only for frame capture and JPEG encoding */}
        <canvas ref={canvasRef} style={{ display: "none" }} />

        {/* Display errors like webcam permission denied or backend not running */}
        {err ? <p className="errorText">{err}</p> : null}
      </div>

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Controls</h3>

        {/* Starts/stops the frame sending loop */}
        <button onClick={running ? stop : start} className="primaryBtn">
          {running ? "Stop" : "Start"}
        </button>

        <div style={{ marginTop: 14 }}>
          <h4 className="sectionTitle">Current Result</h4>

          {/* Shows the latest recognition payload received from backend */}
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
