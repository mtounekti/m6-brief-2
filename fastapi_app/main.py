import os
import io
import sqlite3
from datetime import datetime
import optuna

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from loguru import logger
from starlette.responses import Response
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# chargement du modèle CNN MNIST au démarrage de l'API
MODEL_PATH = os.getenv("MODEL_PATH", "models/cnn_mnist.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# Chemin vers la base SQLite pour stocker les corrections utilisateur
DB_PATH = os.getenv("DB_PATH", "data/corrections.db")
CORRECTION_REPEAT_FACTOR = int(os.getenv("CORRECTION_REPEAT_FACTOR", "20"))


def init_db():
    # Prépare le dossier et la table des corrections.
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction INTEGER,
            correction INTEGER,
            image BLOB,
            timestamp TEXT
        )
    """)

    # Migre les anciennes bases sans colonne image.
    cursor.execute("PRAGMA table_info(corrections)")
    columns = [column[1] for column in cursor.fetchall()]
    if "image" not in columns:
        cursor.execute("ALTER TABLE corrections ADD COLUMN image BLOB")

    # Table d'historique des corrections supprimées lors du nettoyage
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS error_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            correction_id INTEGER,
            prediction INTEGER,
            correction INTEGER,
            reason TEXT,
            timestamp TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()

# métriques Prometheus exposées sur /metrics
prediction_counter = Counter(
    "predictions_total", "Nombre total de prédictions", ["predicted_class"])
correction_counter = Counter("corrections_total", "Nombre total de corrections", [
                             "predicted_class", "corrected_class"])
prediction_latency = Histogram(
    "prediction_latency_seconds", "Latence des prédictions")

app = FastAPI()


@app.get("/", summary="Racine", description="Vérifie que l'API est en ligne.")
async def root():
    return {"message": "FastIA MNIST API"}


@app.get("/health", summary="Santé", description="Retourne le statut de l'API, utilisé par Uptime Kuma.")
async def health():
    return {"status": "ok", "service": "fastapi"}


@app.get("/metrics", summary="Métriques Prometheus", description="Expose les métriques de l'API au format Prometheus.")
async def metrics():
    # endpoint scrappé par Prometheus
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/errors", summary="Historique des erreurs", description="Retourne l'historique des corrections supprimées lors du nettoyage.")
async def get_errors():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM error_log ORDER BY timestamp DESC LIMIT 100")
    rows = cursor.fetchall()
    conn.close()
    return {"errors": [
        {"id": r[0], "correction_id": r[1], "prediction": r[2], "correction": r[3], "reason": r[4], "timestamp": r[5]}
        for r in rows
    ]}


@app.post("/predict", summary="Prédiction", description="Reçoit une image PNG d'un chiffre manuscrit et retourne la classe prédite (0-9) ainsi que le score de confiance.")
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
    logger.info(
        f"[PREDICT] classe={predicted_class} confiance={confidence:.2f}")

    return {"prediction": predicted_class, "confidence": round(confidence, 2)}


@app.post("/correct", summary="Correction", description="Reçoit la prédiction originale, la correction et l'image utilisateur pour le réentraînement.")
async def correct(
    prediction: int = Form(...),
    correction: int = Form(...),
    file: UploadFile = File(...),
):
    # Stocke le couple image corrigée / label utilisateur.
    contents = await file.read()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO corrections (prediction, correction, image, timestamp) VALUES (?, ?, ?, ?)",
        (prediction, correction, contents, datetime.now().isoformat())
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


def load_correction_dataset(corrections):
    # Convertit les images corrigées en tensors MNIST.
    X_corrections = []
    y_corrections = []

    for image_blob, correction in corrections:
        try:
            img = Image.open(io.BytesIO(image_blob)).convert(
                "L").resize((28, 28))
            img_array = np.array(img).astype("float32") / 255.0
            X_corrections.append(img_array.reshape(28, 28, 1))
            y_corrections.append(correction)
        except Exception as error:
            logger.warning(f"[RETRAIN] Correction ignorée: {error}")

    if not X_corrections:
        return None, None

    return np.array(X_corrections), to_categorical(np.array(y_corrections), 10)


@app.post("/retrain", summary="Fine-tuning", description="Fine-tuning du modèle existant sur les nouvelles corrections utilisateur")
async def retrain():
    global model
    logger.info("[RETRAIN] Démarrage du fine-tuning")

    # chargement des corrections depuis SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image, correction FROM corrections WHERE image IS NOT NULL")
    corrections = cursor.fetchall()
    conn.close()
    logger.info(f"[RETRAIN] {len(corrections)} corrections chargées depuis SQLite")

    X_corrections, y_corrections_cat = load_correction_dataset(corrections)
    if X_corrections is None:
        logger.warning("[RETRAIN] Aucune correction exploitable, fine-tuning annulé")
        return {"message": "Aucune correction exploitable"}

    # chargement du jeu de test MNIST pour évaluation
    (_, _), (X_test, y_test) = mnist.load_data()
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
    y_test_cat = to_categorical(y_test, 10)

    # score avant fine-tuning
    _, accuracy_before = model.evaluate(X_test, y_test_cat, verbose=0)
    logger.info(f"[RETRAIN] Accuracy avant : {accuracy_before:.4f}")

    # fine-tuning : on gèle les couches convolutives, on réentraîne seulement les couches denses
    for layer in model.layers[:-3]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(
        X_corrections,
        y_corrections_cat,
        epochs=10,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # on dégèle toutes les couches pour les prochains fine-tunings
    for layer in model.layers:
        layer.trainable = True

    # score après fine-tuning
    _, accuracy_after = model.evaluate(X_test, y_test_cat, verbose=0)
    logger.info(f"[RETRAIN] Accuracy après : {accuracy_after:.4f}")

    # save new model
    model.save(MODEL_PATH)
    logger.info(f"[RETRAIN] Modèle sauvegardé dans {MODEL_PATH}")

    return {
        "accuracy_before": round(accuracy_before, 4),
        "accuracy_after": round(accuracy_after, 4),
        "corrections_used": len(corrections)
    }

@app.get("/corrections", summary="Corrections", description="Retourne les corrections enregistrées.")
async def get_corrections():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, prediction, correction, timestamp FROM corrections ORDER BY timestamp DESC LIMIT 100")
    rows = cursor.fetchall()
    conn.close()
    return {"corrections": [
        {"id": r[0], "prediction": r[1], "correction": r[2], "timestamp": r[3]}
        for r in rows
    ]}