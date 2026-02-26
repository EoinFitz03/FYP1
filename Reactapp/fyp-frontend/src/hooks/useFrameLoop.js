import { useEffect, useRef } from "react"; // useRef does not trigger re-enders, useEffect runs code when the component renders / dependencies change, and can clean up when it unmounts or changes.
import { captureFrameBase64 } from "../utils/captureFrame"; //Imports the helper that does the video → canvas → base64 JPEG

/**
 * Frame sender loop:
 * - captures base64 JPEG from video+canvas
 * - sends via sendJson
 */
export function useFrameLoop({
  enabled, //if false do not run timer 
  videoRef, // webcame stream 
  canvasRef, //snapshot
  sendJson,
  intervalMs = 350,
  frameOptions,
  messageBuilder, // (base64) => payload
}) {
  const timerRef = useRef(null);

  useEffect(() => {  
    if (timerRef.current) {  // prevents duplicates and prevents multiple intervals running 
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (!enabled) return; // if enabled is fasle exit

    timerRef.current = setInterval(() => {
      const videoEl = videoRef?.current; // read current information from website 
      const canvasEl = canvasRef?.current;
      if (!videoEl || !canvasEl) return; // safety check if there is nothing 

      const base64 = captureFrameBase64(videoEl, canvasEl, frameOptions);
      if (!base64) return; // converts the base to JPEG

      const payload = messageBuilder
        ? messageBuilder(base64)
        : { type: "frame", data: base64 };

      sendJson(payload); //sends JSON to the websocket 
    }, intervalMs); 

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [enabled, videoRef, canvasRef, sendJson, intervalMs, frameOptions, messageBuilder]);
}
