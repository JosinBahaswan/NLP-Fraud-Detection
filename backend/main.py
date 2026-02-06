from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Training Data"
MODEL_PATH = DATA_DIR / "knn_model.pkl"
VECTORIZER_PATH = DATA_DIR / "tfidf_vectorizer.pkl"


class TextInput(BaseModel):
    text: str = Field(..., min_length=3, max_length=5_000, description="Teks yang akan dicek")


class PredictionResponse(BaseModel):
    is_fraud: bool
    label: str
    probability: Optional[float] = Field(None, description="Probabilitas prediksi jika tersedia")


def load_assets():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise FileNotFoundError("Model atau vectorizer tidak ditemukan. Pastikan file .pkl ada di folder Training Data.")

    model_obj = joblib.load(MODEL_PATH)
    vectorizer_obj = joblib.load(VECTORIZER_PATH)
    return model_obj, vectorizer_obj


model, vectorizer = load_assets()

app = FastAPI(title="Fraud Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def predict_label(text: str) -> PredictionResponse:
    transformed = vectorizer.transform([text])
    label_int = int(model.predict(transformed)[0])

    proba: Optional[float] = None
    if hasattr(model, "predict_proba"):
        try:
            proba = float(model.predict_proba(transformed)[0][label_int])
        except Exception:
            proba = None

    is_fraud = bool(label_int == 1)
    label = "Fraud" if is_fraud else "Bukan Fraud"
    return PredictionResponse(is_fraud=is_fraud, label=label, probability=proba)


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "vectorizer": VECTORIZER_PATH.name}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TextInput):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Teks tidak boleh kosong")
    try:
        return predict_label(payload.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Gagal memproses prediksi: {exc}") from exc


@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"Unhandled error: {exc}"})

