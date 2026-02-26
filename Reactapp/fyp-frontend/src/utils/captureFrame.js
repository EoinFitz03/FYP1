// src/utils/captureFrame.js
// Captures current <video> frame into <canvas> and returns base64 JPEG (NO data URL header).

export function captureFrameBase64(videoEl, canvasEl, opts = {}) {
  const { width = 640, height = 360, quality = 0.6 } = opts;

  if (!videoEl || !canvasEl) return null;
  if (videoEl.readyState < 2) return null; // HAVE_CURRENT_DATA

  canvasEl.width = width;
  canvasEl.height = height;

  const ctx = canvasEl.getContext("2d");
  if (!ctx) return null;

  ctx.drawImage(videoEl, 0, 0, width, height);

  const dataUrl = canvasEl.toDataURL("image/jpeg", quality);
  const parts = dataUrl.split(",");
  return parts[1] || null;
}
