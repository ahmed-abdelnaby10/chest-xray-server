from typing import Tuple
import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert → RGB, resize (BILINEAR), to float32 in 0..255, expand dims to (1,H,W,3).
    If your model expects 0..1 or z-score normalization, adjust here.
    """
    img = image.convert("RGB").resize(target_size, Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)
    x = x[None, ...]  # (1, H, W, 3)
    return x
