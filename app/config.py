import os
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Register all models here.
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mobile_image_net": {
        "model_path": os.path.join(MODELS_DIR, "mobile_image_net.keras"),
        "labels_path": os.path.join(MODELS_DIR, "labels.json"),
        "img_size": (224, 224),  # (H, W)
        "head": "sigmoid",  # or "softmax"
    },
    "efficient_net_b3": {
        "model_path": os.path.join(MODELS_DIR, "efficient_net_b3.keras"),
        "labels_path": os.path.join(MODELS_DIR, "labels.json"),
        "img_size": (300, 300),
        "head": "sigmoid",
    },
}

DEFAULT_MODEL_NAME = "mobile_image_net"

# Allowed MIME types for uploads
ALLOWED_MIME = {"image/jpeg", "image/png", "image/jpg", "image/webp", "image/avif"}

# Server
PORT = int(os.getenv("PORT", "7860"))
