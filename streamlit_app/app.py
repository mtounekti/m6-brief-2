import io
import os

import numpy as np
import requests
import streamlit as st
from loguru import logger
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
API_URL = os.getenv("API_URL", "http://localhost:8080").rstrip("/")
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"


@st.cache_resource
def configure_logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(LOG_FILE, rotation="10 MB", retention="7 days", level="INFO")

    return logger


def preprocess_canvas_image(image_data: np.ndarray):
    image = Image.fromarray(image_data.astype("uint8"),
                            mode="RGBA").convert("L")
    image = image.resize((28, 28))

    return image


def request_prediction(image: Image.Image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    response = requests.post(f"{API_URL}/predict", files={"file": (
        "digit.png", image_bytes.getvalue(), "image/png")}, timeout=10)

    response.raise_for_status()
    return response.json()


# Envoie le feedback avec l'image corrigée.
def request_correction(prediction: int, correction: int, image: Image.Image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    response = requests.post(
        f"{API_URL}/correct",
        data={"prediction": prediction, "correction": correction},
        files={"file": ("corrected_digit.png", image_bytes.getvalue(), "image/png")},
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def canvas_has_drawing(image_data: np.ndarray) -> bool:
    if image_data is None:
        return False
    rgb_pixels = image_data[:, :, :3]
    white_pixels = np.sum(np.any(rgb_pixels > 30, axis=2))
    return white_pixels > 20


configure_logger()

st.set_page_config(
    page_title="MNIST Classifier",
    page_icon="🔢",
    layout="centered",
)

st.title("Classification MNIST")
st.write("Dessine un chiffre entre 0 et 9, puis lance la prédiction.")


canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=500,
    width=500,
    drawing_mode="freedraw",
    key="mnist_canvas",
)

has_drawing = canvas_has_drawing(canvas_result.image_data)

# Oublie la prédiction quand le canvas est vide.
if not has_drawing:
    st.session_state.pop("prediction", None)
    st.session_state.pop("confidence", None)
    st.session_state.pop("last_image", None)


if st.button("Prédire", type="primary"):
    if not has_drawing:
        st.warning("Dessine un chiffre avant de lancer la prédiction.")
        st.stop()

    image = preprocess_canvas_image(canvas_result.image_data)

    st.image(
        image, caption="Image envoyée au modèle, format MNIST (28x28 pixels)", width=140)

    try:
        logger.info("Sending MNIST image to API for prediction...")
        response = request_prediction(image)

        prediction = response.get("prediction")
        confidence = response.get("confidence")

        if prediction is None:
            st.error(
                "La résponse de l'API est invalide elle ne contient pas de prédiction.")
            logger.error("Invalid response from API: no prediction")
            st.stop()

        st.session_state["prediction"] = prediction
        st.session_state["confidence"] = confidence
        # Garde l'image liée à la correction utilisateur.
        st.session_state["last_image"] = image

        st.success(f"Chiffre prédit: {prediction}")

        if confidence is not None:
            st.write(f"Confiance: {confidence:.0%}")

    except requests.exceptions.Timeout:
        logger.error("API request timed out")
        st.error("La requête à l'API a pris trop de temps pour répondre.")

    except requests.exceptions.RequestException as e:
        logger.error(f"API connection error: {e}")
        st.error(f"Erreur lors de la connexion à l'API: {e}")

if "prediction" in st.session_state:
    st.divider()
    st.subheader("Correction")

    st.write(f"Chiffre prédit: {st.session_state['prediction']}")

    confidence = st.session_state.get("confidence")
    if confidence is not None:
        st.write(f"Confiance: {confidence:.0%}")

    is_correct = st.radio(
        "Est-ce que la prédiction est correcte ?", ["Oui", "Non"], horizontal=True)

    if is_correct == "Non":
        correction = st.selectbox(
            "Quel était le bon chiffre ?", list(range(10)))

        if st.button("Envoyer la correction"):
            try:
                response = request_correction(
                    st.session_state["prediction"],
                    correction,
                    st.session_state["last_image"],
                )
                logger.info(
                    f"Correction sent: prediction={st.session_state['prediction']}, correction={correction}")

                st.success(response.get("message", "Correction enregistrée."))

            except requests.exceptions.Timeout:
                logger.error("API request timed out")
                st.error("La requête à l'API a pris trop de temps pour répondre.")
            except requests.exceptions.RequestException as e:
                logger.error(f"API connection error: {e}")
                st.error(f"Erreur lors de la connexion à l'API: {e}")
    else:
        st.info("Aucune correction nécessaire.")

#errors enregistrées
st.divider()
st.subheader("Historique des corrections supprimées")

if st.button("Charger l'historique"):
    try:
        response = requests.get(f"{API_URL}/errors", timeout=10)
        response.raise_for_status()
        errors = response.json().get("errors", [])

        if not errors:
            st.info("Aucune correction supprimée pour l'instant.")
        else:
            st.dataframe(errors)

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la connexion à l'API: {e}")

# Corrections enregistrées
st.divider()
st.subheader("Corrections enregistrées")

if st.button("Charger les corrections"):
    try:
        response = requests.get(f"{API_URL}/corrections", timeout=10)
        response.raise_for_status()
        corrections = response.json().get("corrections", [])

        if not corrections:
            st.info("Aucune correction enregistrée pour l'instant.")
        else:
            st.dataframe(corrections)

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la connexion à l'API: {e}")