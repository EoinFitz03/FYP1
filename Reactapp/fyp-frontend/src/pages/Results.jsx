import { useEffect, useState } from "react";
import "../styles/live.css";

const STORAGE_KEY = "doorbell_recent_results";

export default function Results() {
  const [items, setItems] = useState([]);

  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      setItems(Array.isArray(saved) ? saved : []);
    } catch {
      setItems([]);
    }
  }, []);

  const clearResults = () => {
    localStorage.removeItem(STORAGE_KEY);
    setItems([]);
  };

  return (
    <div className="card">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        <h2 style={{ marginTop: 0, marginBottom: 0 }}>Recent Door Activity</h2>

        <button onClick={clearResults} className="secondaryBtn">
          Clear
        </button>
      </div>

      <p className="smallText" style={{ marginTop: 10 }}>
        Shows the most recent people detected at the door.
      </p>

      {items.length === 0 ? (
        <p>No recent activity yet.</p>
      ) : (
        <div style={{ marginTop: 14, display: "grid", gap: 10 }}>
          {items.map((item) => (
            <div
              key={item.id}
              style={{
                border: "1px solid #ddd",
                borderRadius: 10,
                padding: 12,
                background: "#fff",
              }}
            >
              <div>
                <b>Person:</b> {item.person}
              </div>
              <div>
                <b>Type:</b> {item.person === "Unknown" ? "Unknown person" : "Known person"}
              </div>
              <div>
                <b>Time:</b> {item.time}
              </div>
              {typeof item.face_conf === "number" ? (
                <div>
                  <b>Confidence:</b> {Math.round(item.face_conf * 100)}%
                </div>
              ) : null}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}