import uvicorn
from schemas.SampleSchema import SampleSchema
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import io

import joblib
import pandas as pd

app = FastAPI()

# ---- Model loading (fix 1) ----
# Use env var from docker-compose (MODEL_DIR: /models) and correct filename.
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "best_model_pipeline.joblib")
model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)

try:
    model = joblib.load(model_path)
except FileNotFoundError as e:
    # Give a clear error if the file isn't where we expect.
    raise RuntimeError(
        f"Model file not found at '{model_path}'. "
        "Check docker-compose MODEL_DIR / Dockerfile COPY paths."
    ) from e

@app.get("/health", tags=["health"])
def health_check():
    return {"status": "healthy"}

# ---- Single sample predict (fix 2) ----
@app.post("/predict", tags=["predict"])
def predict(sample: SampleSchema):
    """
    Accepts a single JSON sample matching SampleSchema and returns a prediction.
    """
    # Pydantic v2: use model_dump(), not dict(sample)
    row = sample.model_dump()
    df = pd.DataFrame([row])
    try:
        pred = model.predict(df)
    except Exception as e:
        # Surface model/data errors as 422 to make debugging easier in CI logs
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")
    # Ensure JSON-serializable
    return {"prediction": (pred[0].item() if hasattr(pred[0], "item") else pred[0])}

# ---- Batch predict (fix 3) ----
@app.post("/predict_batch", tags=["predict"])
def predict_batch(file: UploadFile):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only CSV files are allowed.")

    try:
        # Let pandas detect delimiter; supports ; or , automatically with engine='python'
        # Also rewind file pointer just in case.
        file.file.seek(0)
        df = pd.read_csv(file.file, sep=None, engine="python")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to read CSV: {e}")

    try:
        predictions = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {e}")

    # Convert to plain Python types for JSON
    preds = [p.item() if hasattr(p, "item") else p for p in predictions]
    return {"predictions": preds}

if __name__ == "__main__":
    # Keep as 0.0.0.0:8000 so docker-compose can map it; your local health-check uses :8000.
    uvicorn.run(app, host="0.0.0.0", port=8000)
