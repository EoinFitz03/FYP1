export function createWS(url, { onOpen, onClose, onError, onMessage } = {}) {
  const ws = new WebSocket(url);

  ws.onopen = () => onOpen?.();
  ws.onclose = () => onClose?.();
  ws.onerror = (e) => onError?.(e);
  ws.onmessage = (evt) => onMessage?.(evt.data);

  return ws;
}
