import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Webcam hook:
 * - Starts on mount, stops on unmount
 * - Default constraints: video: true 
 */
export function useWebcam(videoRef, constraints) {
  const streamRef = useRef(null); // Stores the MediaStream returned by the webcam.

  const [ready, setReady] = useState(false); 
  const [error, setError] = useState("");

  const stop = useCallback(() => {
    setReady(false);

    const stream = streamRef.current; // get sthe current stream from ref 
    if (stream) {
      stream.getTracks().forEach((t) => t.stop()); // sjuts down camera correctly 
      streamRef.current = null; // clears current ref 
    }

    if (videoRef?.current) videoRef.current.srcObject = null; //prevents pointing to an old stream 
  }, [videoRef]);

  const start = useCallback(async () => {
    setError("");
    setReady(false);

    try {
      stop(); // if there was s stream running stop it 

      const defaultConstraints = {
        video: true, // gives the webcam access
        audio: false,
      };

      const stream = await navigator.mediaDevices.getUserMedia( // ask browser for perimission to use mediaStream 
        constraints ?? defaultConstraints
      );

      streamRef.current = stream;

      if (videoRef?.current) {
        videoRef.current.srcObject = stream; //Attaches the stream to the <video> element this is what makes the live webcam appear on screen
        const p = videoRef.current.play(); //chrome requires .play to play it  
        if (p && typeof p.catch === "function") p.catch(() => {});
      }

      setReady(true); //webcam is ready if we got this far 
    } catch (e) { //error handling of anything above fails
      console.error("Webcam start failed:", e);
      setError(
        e?.name === "NotAllowedError"
          ? "Camera permission denied"
          : "Could not access camera"
      );
      stop();
    }
  }, [constraints, stop, videoRef]);

  useEffect(() => {
    start();
    return () => stop(); 
  }, [start, stop]); // turns camera ona and off

  return { ready, error, start, stop, stream: streamRef.current };
}
