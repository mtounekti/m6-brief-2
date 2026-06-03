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
    logger.info(f"[PREDICT] classe={predicted_class} confiance={confidence:.2f}")

    return {"prediction": predicted_class, "confidence": round(confidence, 2)}

@app.post("/correct", summary="Correction", description="Reçoit la prédiction originale et la correction de l'utilisateur, les stocke en base SQLite pour le réentraînement.")
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


def build_model(lr: float, dropout: float):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(),
        Dropout(dropout),
        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D(),
        Dropout(dropout),
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_and_evaluate(model, X_train, y_train, X_val, y_val):
    early_stop = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )
    _, accuracy = model.evaluate(X_val, y_val, verbose=0)
    return accuracy


@app.post("/retrain", summary="Réentraînement", description="Récupère les corrections SQLite, les combine avec MNIST, optimise avec Optuna et réentraîne le CNN.")
async def retrain():
    global model  # en premier, une seule fois
    logger.info("[RETRAIN] Démarrage du réentraînement")

    # chargement du dataset MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    # chargement des corrections depuis SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT prediction, correction FROM corrections")
    corrections = cursor.fetchall()
    conn.close()
    logger.info(f"[RETRAIN] {len(corrections)} corrections chargées depuis SQLite")

    # score avant réentraînement
    _, accuracy_before = model.evaluate(X_test, y_test_cat, verbose=0)
    logger.info(f"[RETRAIN] Accuracy avant : {accuracy_before:.4f}")

    # optim Optuna
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        trial_model = build_model(lr=lr, dropout=dropout)
        return train_and_evaluate(trial_model, X_train, y_train_cat, X_test, y_test_cat)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=2) # used 5 at first , it was very long but got 99.49% accuracy
    best_params = study.best_params
    logger.info(f"[RETRAIN] Meilleurs paramètres Optuna : {best_params}")

    # réentraînement final avec les meilleurs paramètres
    model = build_model(lr=best_params["lr"], dropout=best_params["dropout"])
    accuracy_after = train_and_evaluate(model, X_train, y_train_cat, X_test, y_test_cat)

    # save new model
    model.save(MODEL_PATH)
    logger.info(f"[RETRAIN] Accuracy après : {accuracy_after:.4f}")
    logger.info(f"[RETRAIN] Modèle sauvegardé dans {MODEL_PATH}")

    return {
        "accuracy_before": round(accuracy_before, 4),
        "accuracy_after": round(accuracy_after, 4),
        "best_params": best_params,
        "corrections_used": len(corrections)
    }