import { useCallback, useEffect, useRef, useState } from "react";

/**
 * WebSocket hook:
 * - Connect on mount, cleanup on unmount
 * - lastMessage contains parsed JSON
 */
export function useWS(url) { // websocket URL
  const wsRef = useRef(null);
  const connectingRef = useRef(false); // stops it trying to connect multiple times 

  const [connected, setConnected] = useState(false); // socket open/closed
  const [error, setError] = useState("");
  const [lastMessage, setLastMessage] = useState(null);

  const disconnect = useCallback(() => {
    const ws = wsRef.current;
    wsRef.current = null;
    connectingRef.current = false; // resets connection 

    if (ws) {
      try {
        ws.onopen = null;
        ws.onclose = null;
        ws.onerror = null;
        ws.onmessage = null;
        ws.close(1000, "component unmount"); // normal close 
      } catch (_) {} //avoids browser crashing 
    }

    setConnected(false);
  }, []);

  const connect = useCallback(() => { //connects websocket 
    if (connectingRef.current) return; //prevensy multiple connetion attempts 

    const existing = wsRef.current;
    if ( // if websocket is already trying ot connect or open do not try to do these again 
      existing &&
      (existing.readyState === WebSocket.OPEN ||
        existing.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    setError("");
    connectingRef.current = true;

    let ws;
    try {
      ws = new WebSocket(url); //try create new webseocket and if not show error
    } catch (e) {
      connectingRef.current = false;
      setError("Could not create WebSocket");
      setConnected(false);
      return;
    }

    wsRef.current = ws; // save so other functions can use like sendJSON

    ws.onopen = () => {
      connectingRef.current = false;
      setConnected(true);
    }; //updates UI 

    ws.onclose = (ev) => {
      connectingRef.current = false;
      setConnected(false);
      console.log("WS closed:", { code: ev.code, reason: ev.reason });
      if (ev.code !== 1000 && ev.code !== 1001) {
        setError(`WebSocket closed (code ${ev.code})`);
      }
    }; // this if for any enexpected drops liek refresh or wifi drop 

    ws.onerror = () => {
      setError("WebSocket error");
      setConnected(false);
    };

    ws.onmessage = (event) => {
      try {
        setLastMessage(JSON.parse(event.data));
      } catch {
        setLastMessage({ raw: event.data });
      } //sends jsons to backend so it can read them 
    };
  }, [url]);

  const sendJson = useCallback((obj) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return false; // sends if socket exist and is open

    try {
      ws.send(JSON.stringify(obj)); //actually sends the object 
      return true;
    } catch (e) {
      console.error("WS send failed:", e);
      return false;
    }
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return {
    connected,
    error,
    lastMessage,
    sendJson,
    reconnect: () => {
      disconnect();
      connect();
    },
  };
}
