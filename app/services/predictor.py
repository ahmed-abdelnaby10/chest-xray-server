from typing import Dict, Any, List
import numpy as np


def postprocess_binary_sigmoid(pred: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    # pred shape: (N,1) or (N,)
    p1 = float(np.squeeze(pred))
    idx = 1 if p1 >= 0.5 else 0
    prob = p1 if idx == 1 else (1.0 - p1)
    return {"label": labels[idx], "probability": round(prob, 4)}


def postprocess_softmax(pred: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    probs = pred[0].astype(float).tolist()
    idx = int(np.argmax(probs))
    return {
        "label": labels[idx],
        "probability": round(float(probs[idx]), 4),
        "probs": {labels[i]: round(float(p), 4) for i, p in enumerate(probs)},
    }


def predict_one(model, labels: List[str], arr, head: str) -> Dict[str, Any]:
    out = model.predict(arr, verbose=0)
    if head == "softmax":
        return postprocess_softmax(out, labels)
    # default sigmoid (binary)
    return postprocess_binary_sigmoid(out, labels)
