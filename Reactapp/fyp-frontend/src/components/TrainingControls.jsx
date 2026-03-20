import { useState } from "react";

// Simple UI to send train_start / train_stop over an existing sendJson function
export default function TrainingControls({ connected, sendJson }) {
  const [label, setLabel] = useState("open_palm");
  const [numSamples, setNumSamples] = useState(200);

  const start = () => {
    if (!sendJson) return;
    sendJson({
      type: "train_start",
      label: String(label || "").trim(),
      num_samples: Number(numSamples) || 200,
    });
  };

  const stop = () => {
    if (!sendJson) return;
    sendJson({ type: "train_stop" });
  };

  return (
    <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
      <strong>Training</strong>

      <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
        Label:
        <input
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="open_palm"
          style={{ padding: "6px 8px", minWidth: 140 }}
          disabled={!connected}
        />
      </label>

      <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
        Samples:
        <input
          type="number"
          value={numSamples}
          onChange={(e) => setNumSamples(e.target.value)}
          style={{ padding: "6px 8px", width: 90 }}
          min={10}
          max={5000}
          disabled={!connected}
        />
      </label>

      <button onClick={start} disabled={!connected} style={{ padding: "6px 10px" }}>
        Start Training
      </button>

      <button onClick={stop} disabled={!connected} style={{ padding: "6px 10px" }}>
        Stop Training
      </button>

      {!connected && <span style={{ opacity: 0.7 }}>Connect first</span>}
    </div>
  );
}