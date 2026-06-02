from loguru import logger
import streamlit as st
import requests
import os
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

def preprocess_canvas_image(image_data: np.ndarray):
    image = Image.fromarray(image_data.astype("uint8"), mode="RGBA").convert("L")
    image = image.resize((28, 28))
    
    pixels = np.array(image).astype("float32") / 255.0
    
    return pixels.tolist(), image

def request_predication(image_payload: list[list[float]]):
    response = requests.post(f"{api_url}/predict", json={"image": image_payload}, timeout=10)

    response.raise_for_status()
    return response.json()

# Récupération de l'URL de l'API depuis les variables d'environnement
api_url = f"http://api:{os.getenv('FASTAPI_PORT', '8080')}"
log_file = "logs/app.log"

# Configuration de Loguru pour sauvegarder les logs
logger.add(log_file, rotation="10 MB", retention="7 days", level="INFO")

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

if st.button("Prédire", type="primary"):
    if canvas_result.image_data is None:
        st.warning("Dessine un chiffre avant de lancer la prédiction.")
        st.stop()

    image_payload, image = preprocess_canvas_image(canvas_result.image_data)

    st.image(image, caption="Image envoyée au modèle, format MNIST (28x28 pixels)", width=140)

    try:
        logger.info("Sending MNIST image to API for prediction...")
        prediction = request_predication(image_payload)