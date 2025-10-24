from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import predict
from .schemas import HealthResp
from .services.model_registry import available_models

app = FastAPI(title="Image Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResp)
def health():
    return {"ok": True, "models": available_models()}

app.include_router(predict.router)
