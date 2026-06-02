import os
import io
import sqlite3
from datetime import datetime

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from loguru import logger
from starlette.responses import Response
import tensorflow as tf

# chargement du modèle CNN MNIST au démarrage de l'API
MODEL_PATH = os.getenv("MODEL_PATH", "models/cnn_mnist.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# Chemin vers la base SQLite pour stocker les corrections utilisateur
DB_PATH = os.getenv("DB_PATH", "data/corrections.db")

def init_db():
    # création de la table corrections
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction INTEGER,
            correction INTEGER,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# métriques Prometheus exposées sur /metrics
prediction_counter = Counter("predictions_total", "Nombre total de prédictions", ["predicted_class"])
correction_counter = Counter("corrections_total", "Nombre total de corrections", ["predicted_class", "corrected_class"])
prediction_latency = Histogram("prediction_latency_seconds", "Latence des prédictions")

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "FastIA MNIST API"}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "fastapi"}

@app.get("/metrics")
async def metrics():
    # endpoint scrappé par Prometheus
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # lire et préparer l'image reçue depuis Streamlit
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # prédiction via le modèle CNN
    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    prediction_counter.labels(predicted_class=str(predicted_class)).inc()
    logger.info(f"[PREDICT] classe={predicted_class} confiance={confidence:.2f}")

    return {"prediction": predicted_class, "confidence": round(confidence, 2)}

@app.post("/correct")
async def correct(prediction: int = Form(...), correction: int = Form(...)):
    # stockage de la correction en SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO corrections (prediction, correction, timestamp) VALUES (?, ?, ?)",
        (prediction, correction, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    # maj des métriques Prometheus
    correction_counter.labels(
        predicted_class=str(prediction),
        corrected_class=str(correction)
    ).inc()
    logger.info(f"[CORRECT] prédit={prediction} corrigé={correction}")

    return {"message": "Correction enregistrée", "prediction": prediction, "correction": correction}