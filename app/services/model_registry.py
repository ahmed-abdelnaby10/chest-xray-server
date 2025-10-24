import json, os
from typing import Any, Dict, List, Tuple
from fastapi import HTTPException
from ..config import MODEL_REGISTRY
import tensorflow as tf

_model_cache: Dict[str, Dict[str, Any]] = {}


def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def load_model_bundle(name: str) -> Dict[str, Any]:
    if name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model '{name}'. Available: {available_models()}",
        )
    if name in _model_cache:
        return _model_cache[name]

    spec = MODEL_REGISTRY[name]
    model_path = spec["model_path"]
    labels_path = spec["labels_path"]
    img_size = spec["img_size"]
    head = spec.get("head", "sigmoid")

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=500, detail=f"Model file not found: {model_path}"
        )
    if not os.path.exists(labels_path):
        raise HTTPException(
            status_code=500, detail=f"Labels file not found: {labels_path}"
        )

    model = tf.keras.models.load_model(model_path, safe_mode=False, compile=False)
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)["class_names"]

    bundle = {"model": model, "labels": labels, "img_size": img_size, "head": head}
    _model_cache[name] = bundle
    return bundle
