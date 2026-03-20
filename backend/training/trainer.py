# backend/training/trainer.py
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # .../backend
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "gestures.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.pkl")


def train_gesture_model(
    dataset_path: str = DATASET_PATH,
    model_path: str = MODEL_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 200,
    max_depth=None,
    min_samples_leaf: int = 1,
):
    """
    Trains a RandomForestClassifier on MediaPipe hand landmark CSV.

    CSV expected columns:
      label, hand, x0,y0,z0 ... x20,y20,z20

    Returns:
      model, metrics_dict
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    # Optional 'hand' column (Left/Right/Unknown). Use it if present.
    y = df["label"].astype(str)

    drop_cols = ["label"]
    if "hand" in df.columns:
        # Encode hand as simple numeric feature (optional but can help)
        hand_map = {"Left": 0, "Right": 1, "Unknown": 2}
        df["hand_enc"] = df["hand"].map(hand_map).fillna(2).astype(int)
        drop_cols.append("hand")

    X = df.drop(columns=drop_cols)

    # Basic sanity check: expect at least 63 landmark columns
    if X.shape[1] < 63:
        raise ValueError(f"Expected landmark columns; got only {X.shape[1]} feature columns.")

    # Train/test split (stratify keeps label proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    labels = sorted(y.unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": X.columns.tolist(),
            "labels": labels,
        },
        model_path
    )

    metrics = {
        "accuracy": acc,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "saved_model_path": model_path,
        "num_rows": int(df.shape[0]),
        "num_features": int(X.shape[1]),
    }
    return model, metrics


if __name__ == "__main__":
    model, metrics = train_gesture_model()
    print("\n=== Gesture Training Results ===")
    print(f"Rows: {metrics['num_rows']}")
    print(f"Features: {metrics['num_features']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("\nLabels:", metrics["labels"])
    print("\nConfusion Matrix (rows=true, cols=pred):")
    for row in metrics["confusion_matrix"]:
        print(row)
    print("\nClassification Report:\n")
    print(metrics["classification_report"])
    print("\nSaved model to:", metrics["saved_model_path"])