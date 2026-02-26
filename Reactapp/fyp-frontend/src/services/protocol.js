// src/services/protocol.js
// Single source of truth for frontend <-> backend message formats.

export function buildFrameMessage(base64) {
  return { type: "frame", data: base64 };
}

export function isResultMessage(msg) {
  return msg && msg.type === "result" && msg.payload;
}

export function getResultPayload(msg) {
  return msg?.payload ?? null;
}

// Enrol messages (match backend app.py: "enrol_start", "enrol_cancel")
export function buildEnrolStart(name, numSamples = 10) {
  return { type: "enrol_start", name, num_samples: numSamples };
}

export function buildEnrolCancel() {
  return { type: "enrol_cancel" };
}
