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


def build_model(lr: float, dropout: float):
    # Construit un CNN MNIST avec les hyperparamètres Optuna.
    trial_model = Sequential([
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
    trial_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return trial_model


def train_and_evaluate(trial_model, X_train, y_train, X_val, y_val):
    # Entraîne le modèle avec early stopping et retourne l'accuracy de validation.
    early_stop = EarlyStopping(patience=3, restore_best_weights=True)
    trial_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10, # mettre 10 au lieu de 20 pour une démo plus rapide
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )
    _, accuracy = trial_model.evaluate(X_val, y_val, verbose=0)
    return accuracy


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


@app.post("/retrain", summary="Réentraînement", description="Réentraînement complet du modèle avec MNIST + corrections, versioning et comparaison.")
async def retrain():
    global model
    logger.info("[RETRAIN] Démarrage du réentraînement complet")

    # chargement du dataset MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)

    # chargement des corrections depuis SQLite
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT image, correction FROM corrections WHERE image IS NOT NULL")
    corrections = cursor.fetchall()
    conn.close()
    logger.info(f"[RETRAIN] {len(corrections)} corrections chargées")

    # injection des corrections dans le dataset
    X_corrections, y_corrections_cat = load_correction_dataset(corrections)
    if X_corrections is not None:
        X_corrections_rep = np.repeat(X_corrections, CORRECTION_REPEAT_FACTOR, axis=0)
        y_corrections_rep = np.repeat(y_corrections_cat, CORRECTION_REPEAT_FACTOR, axis=0)
        X_train = np.concatenate([X_train, X_corrections_rep])
        y_train_cat = np.concatenate([y_train_cat, y_corrections_rep])
        logger.info(f"[RETRAIN] {len(X_corrections_rep)} exemples corrigés injectés")

    # score du modèle actuel avant réentraînement
    _, accuracy_before = model.evaluate(X_test, y_test_cat, verbose=0)
    logger.info(f"[RETRAIN] Accuracy modèle actuel : {accuracy_before:.4f}")

    # optimisation Optuna — 2 trials
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        trial_model = build_model(lr=lr, dropout=dropout)
        return train_and_evaluate(trial_model, X_train, y_train_cat, X_test, y_test_cat)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1) # mettre 1 pour une démo plus rapide si besoin 
    best_params = study.best_params
    logger.info(f"[RETRAIN] Meilleurs paramètres Optuna : {best_params}")

    # réentraînement final avec les meilleurs paramètres
    new_model = build_model(lr=best_params["lr"], dropout=best_params["dropout"])
    accuracy_after = train_and_evaluate(new_model, X_train, y_train_cat, X_test, y_test_cat)
    logger.info(f"[RETRAIN] Accuracy nouveau modèle : {accuracy_after:.4f}")

    # versioning: on garde l'ancien modèle si le nouveau est moins bon
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = MODEL_PATH.replace(".keras", f"_backup_{version}.keras")

    if accuracy_after >= accuracy_before:
        # sauvegarde de l'ancien modèle en backup
        model.save(backup_path)
        logger.info(f"[RETRAIN] Ancien modèle sauvegardé : {backup_path}")
        # remplacement par le nouveau
        model = new_model
        model.save(MODEL_PATH)
        logger.info(f"[RETRAIN] Nouveau modèle déployé : {MODEL_PATH}")
        deployed = True
    else:
        logger.warning(f"[RETRAIN] Nouveau modèle moins performant, on garde l'ancien")
        deployed = False

    return {
        "accuracy_before": round(accuracy_before, 4),
        "accuracy_after": round(accuracy_after, 4),
        "best_params": best_params,
        "corrections_used": len(corrections),
        "deployed": deployed,
        "backup": backup_path if deployed else None,
        "version": version
    }