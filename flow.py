import os
import sqlite3
import requests

from datetime import datetime, timedelta, timezone
from typing import Any
from prefect import flow, task, get_run_logger

DB_PATH = os.getenv("DB_PATH", "data/corrections.db")
API_URL = os.getenv("API_URL", "http://localhost:8080").rstrip("/")
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "5"))
LOCK_TIMEOUT_SECONDS = int(os.getenv("LOCK_TIMEOUT_SECONDS", "3600"))
FLOW_STATE_KEY = "last_retrained_correction_id"
RETRAIN_LOCK_KEY = "retraining_in_progress"


# Initialise la table de suivi du flow.
@task
def init_flow_state(db_path: str):
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS flow_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()


# Nettoie les corrections incohérentes ou corrompues.
@task
def clean_corrections(db_path: str) -> int:
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()

    # Supprime les corrections sans image
    cursor.execute("DELETE FROM corrections WHERE image IS NULL")
    deleted_null = cursor.rowcount

    # Supprime les doublons (même prediction, correction, timestamp)
    cursor.execute("""
        DELETE FROM corrections
        WHERE id NOT IN (
            SELECT MIN(id)
            FROM corrections
            GROUP BY prediction, correction, timestamp
        )
    """)
    deleted_duplicates = cursor.rowcount

    # Supprime les corrections incohérentes (prediction == correction)
    cursor.execute("DELETE FROM corrections WHERE prediction = correction")
    deleted_incoherent = cursor.rowcount

    conn.commit()
    conn.close()

    return deleted_null + deleted_duplicates + deleted_incoherent


# Récupère le dernier feedback déjà traité.
@task
def get_last_processed_id(db_path: str):
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()

    cursor.execute("SELECT value FROM flow_state WHERE key = ?",
                   (FLOW_STATE_KEY,))
    result = cursor.fetchone()
    conn.close()

    return int(result[0]) if result else 0


# Compte les nouveaux feedbacks exploitables.
@task
def get_new_correction_stats(db_path: str, last_processed_id: int):
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*), COALESCE(MAX(id), ?)
        FROM corrections
        WHERE id > ? AND image IS NOT NULL
        """,
                   (last_processed_id, last_processed_id)
                   )

    count, latest_id = cursor.fetchone()
    conn.close()

    return {
        "new_corrections_count": int(count),
        "latest_correction_id": int(latest_id)
    }


# Pose un verrou avant le réentraînement.
@task
def acquire_retraining_lock(db_path: str) -> bool:
    now = datetime.now(timezone.utc)
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()

    cursor.execute("BEGIN IMMEDIATE")
    cursor.execute("SELECT value FROM flow_state WHERE key = ?", (RETRAIN_LOCK_KEY,))
    result = cursor.fetchone()

    if result:
        locked_at = datetime.fromisoformat(result[0])
        if now - locked_at < timedelta(seconds=LOCK_TIMEOUT_SECONDS):
            conn.rollback()
            conn.close()
            return False

    cursor.execute(
        """
        INSERT INTO flow_state (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (RETRAIN_LOCK_KEY, now.isoformat()),
    )
    conn.commit()
    conn.close()

    return True


# Libère le verrou de réentraînement.
@task
def release_retraining_lock(db_path: str) -> None:
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM flow_state WHERE key = ?", (RETRAIN_LOCK_KEY,))
    conn.commit()
    conn.close()


# Déclenche le réentraînement via l'API.
@task
def trigger_retraining(api_url: str) -> dict[str, Any]:
    response = requests.post(f"{api_url}/retrain", timeout=1800)
    response.raise_for_status()
    return response.json()


# Sauvegarde le dernier feedback traité.
@task
def save_last_processed_id(db_path: str, correction_id: int) -> None:
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO flow_state (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (FLOW_STATE_KEY, str(correction_id)),
    )
    conn.commit()
    conn.close()


# Orchestre la décision et le déclenchement du réentraînement.
@flow(name="mnist-retraining-flow")
def mnist_retraining_flow() -> None:
    logger = get_run_logger()

    logger.info("Starting MNIST retraining flow")
    logger.info(f"Using database: {DB_PATH}")
    logger.info(f"Using API: {API_URL}")
    logger.info(f"Retraining threshold: {RETRAIN_THRESHOLD}")

    init_flow_state(DB_PATH)

    # Nettoyage des données avant analyse
    deleted = clean_corrections(DB_PATH)
    logger.info(f"Corrections nettoyées : {deleted}")

    last_processed_id = get_last_processed_id(DB_PATH)
    stats = get_new_correction_stats(DB_PATH, last_processed_id)

    new_corrections_count = stats["new_corrections_count"]
    latest_correction_id = stats["latest_correction_id"]

    logger.info(f"Last processed correction id: {last_processed_id}")
    logger.info(f"New corrections count: {new_corrections_count}")
    logger.info(f"Latest correction id: {latest_correction_id}")

    if new_corrections_count == 0:
        logger.info("No new corrections found. Skipping retraining.")
        return
    if new_corrections_count < RETRAIN_THRESHOLD:
        logger.info(
            f"Retraining skipped: {new_corrections_count} corrections found, "
            f"threshold is {RETRAIN_THRESHOLD}."
        )
        return

    logger.info(
        "Retraining threshold reached. Calling FastAPI /retrain endpoint.")
    lock_acquired = acquire_retraining_lock(DB_PATH)
    if not lock_acquired:
        logger.info("Another retraining is already running. Skipping this run.")
        return

    try:
        retrain_result = trigger_retraining(API_URL)
        logger.info(f"Retraining completed: {retrain_result}")
        save_last_processed_id(DB_PATH, latest_correction_id)
        logger.info(
            f"Flow state updated. Last processed correction id is now {latest_correction_id}."
        )
    finally:
        release_retraining_lock(DB_PATH)
        logger.info("Retraining lock released.")


if __name__ == "__main__":
    mnist_retraining_flow()