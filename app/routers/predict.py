from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List
from io import BytesIO
from PIL import Image
import numpy as np

from ..schemas import PredictSingleResp, PredictBatchResp, PredictItem
from ..config import DEFAULT_MODEL_NAME, ALLOWED_MIME
from ..services.model_registry import load_model_bundle
from ..services.preprocess import preprocess_image
from ..services.predictor import predict_one

router = APIRouter(prefix="", tags=["predict"])

@router.post("/predict", response_model=PredictSingleResp)
async def predict(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL_NAME, description="Model name")
):
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(status_code=415, detail=f"Unsupported type: {file.content_type}")

    try:
        img = Image.open(BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    bundle = load_model_bundle(model)
    arr = preprocess_image(img, target_size=bundle["img_size"])
    out = predict_one(bundle["model"], bundle["labels"], arr, head=bundle["head"])
    return {"model": model, "label": out["label"], "probability": out["probability"]}

@router.post("/predict/batch", response_model=PredictBatchResp)
async def predict_batch(
    files: List[UploadFile] = File(...),
    model: str = Query(DEFAULT_MODEL_NAME, description="Model name")
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    bundle = load_model_bundle(model)
    arrays, names = [], []

    # Read + preprocess
    for f in files:
        if f.content_type not in ALLOWED_MIME:
            raise HTTPException(status_code=415, detail=f"Unsupported type: {f.filename} ({f.content_type})")
        try:
            img = Image.open(BytesIO(await f.read()))
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid image: {f.filename}")
        arrays.append(preprocess_image(img, target_size=bundle["img_size"]))
        names.append(f.filename)

    # Try batched inference; fallback to per-image
    results = []
    try:
        batch = np.vstack(arrays)  # (N,H,W,3)
        preds = bundle["model"].predict(batch, verbose=0)
        if bundle["head"] == "softmax":
            # softmax per-row
            for fname, row in zip(names, preds):
                idx = int(np.argmax(row))
                prob = float(row[idx])
                results.append(PredictItem(file=fname, label=bundle["labels"][idx], probability=round(prob, 4)))
        else:
            # sigmoid
            p1s = np.squeeze(preds).tolist()
            if isinstance(p1s, float): p1s = [p1s]
            for fname, p1 in zip(names, p1s):
                p1 = float(p1)
                idx = 1 if p1 >= 0.5 else 0
                prob = p1 if idx == 1 else (1.0 - p1)
                results.append(PredictItem(file=fname, label=bundle["labels"][idx], probability=round(prob, 4)))
    except Exception:
        # Fallback per-image
        for fname, arr in zip(names, arrays):
            out = predict_one(bundle["model"], bundle["labels"], arr, head=bundle["head"])
            results.append(PredictItem(file=fname, label=out["label"], probability=out["probability"]))

    return {"model": model, "count": len(results), "results": results}
