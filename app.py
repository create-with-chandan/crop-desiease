from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("CROP_MODEL_PATH", BASE_DIR / "models" / "crop_disease.keras"))
LABELS_PATH = Path(os.environ.get("CROP_LABELS_PATH", BASE_DIR / "models" / "labels.json"))
DISEASE_INFO_PATH = Path(os.environ.get("CROP_DISEASE_INFO_PATH", BASE_DIR / "disease_info.json"))
IMAGE_SIZE = (224, 224)
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MIN_PLANT_SCORE = float(os.environ.get("CROP_MIN_PLANT_SCORE", "0.01"))
MIN_CONFIDENCE = float(os.environ.get("CROP_MIN_CONFIDENCE", "0.55"))

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_model: Any = None
_labels: list[str] = []
_model_error: str | None = None
_disease_info: dict[str, dict[str, str]] = {}


def load_artifacts() -> None:
    global _model, _labels, _model_error, _disease_info
    if DISEASE_INFO_PATH.is_file():
        try:
            with DISEASE_INFO_PATH.open("r", encoding="utf-8") as info_file:
                loaded_info = json.load(info_file)
            if isinstance(loaded_info, dict):
                _disease_info = {str(key).lower(): value for key, value in loaded_info.items() if isinstance(value, dict)}
        except (OSError, json.JSONDecodeError):
            _disease_info = {}
    if not MODEL_PATH.is_file() or not LABELS_PATH.is_file():
        _model_error = (
            "The TensorFlow model is not configured. Train a model with train.py "
            "and place crop_disease.keras and labels.json in models/."
        )
        return

    try:
        import tensorflow as tf

        with LABELS_PATH.open("r", encoding="utf-8") as labels_file:
            loaded_labels = json.load(labels_file)
        if not isinstance(loaded_labels, list) or not loaded_labels:
            raise ValueError("labels.json must contain a non-empty JSON list")
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        _labels = [str(label) for label in loaded_labels]
        _model_error = None
    except Exception as error:  # Startup should remain available for a useful error response.
        _model = None
        _labels = []
        _model_error = f"The TensorFlow model could not be loaded: {error}"


def plant_score(image: np.ndarray) -> float:
    """Estimate plant material using green, yellow, and brown vegetation colors."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([25, 35, 25]), np.array([100, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([15, 45, 35]), np.array([40, 255, 245]))
    brown = cv2.inRange(hsv, np.array([5, 35, 20]), np.array([25, 255, 210]))
    vegetation = cv2.bitwise_or(green, cv2.bitwise_or(yellow, brown))
    vegetation = cv2.morphologyEx(vegetation, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return float(np.count_nonzero(vegetation)) / float(vegetation.size)


def decode_image(raw: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("The uploaded file is not a readable image.")
    height, width = image.shape[:2]
    if width < 80 or height < 80:
        raise ValueError("Please upload a larger crop image, at least 80 by 80 pixels.")
    return image


def display_label(label: str) -> str:
    return label.replace("___", " - ").replace("__", " - ").replace("_", " ").strip()


def prediction_description(label: str) -> str:
    disease = display_label(label)
    if any(token in label.lower() for token in ("healthy", "normal", "no_disease")):
        return f"The trained CropAI classifier identified {disease}. No disease pattern was found in the supported classes."
    return (
        f"The trained CropAI classifier identified {disease}. Treat this as an AI-assisted "
        "screening result and confirm it with a local agronomist before applying treatment."
    )


def disease_information(label: str, healthy: bool) -> dict[str, str]:
    normalized = label.lower().replace("___", " ").replace("__", " ").replace("_", " ")
    matched = next(
        (value for key, value in _disease_info.items() if key != "default" and key in normalized),
        _disease_info.get("healthy" if healthy else "default", {}),
    )
    return {
        "cause": matched.get("cause", "Cause information is not available for this trained class."),
        "symptoms": matched.get("symptoms", "Review the image and confirm visible symptoms with an agronomist."),
        "action": matched.get("action", "Isolate affected plants and seek local agricultural guidance."),
        "food_safety": matched.get(
            "food_safety",
            "Do not eat visibly diseased plant parts. Wash produce and follow local food-safety guidance.",
        ),
    }


def error_response(message: str, status: int):
    return jsonify({"error": message}), status


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    return error_response("Image is too large. Please upload a file smaller than 8 MB.", 413)


@app.get("/api/health")
def health():
    return jsonify(
        {
            "status": "ok" if _model is not None else "model_not_ready",
            "model_loaded": _model is not None,
            "labels_loaded": len(_labels),
            "message": _model_error,
        }
    )


@app.post("/api/predict")
def predict():
    if _model is None:
        return error_response(_model_error or "The AI model is not ready.", 503)

    uploaded = request.files.get("image")
    if uploaded is None or not uploaded.filename:
        return error_response("Please upload a crop or plant image.", 400)
    if uploaded.mimetype not in {"image/jpeg", "image/png", "image/jpg"}:
        return error_response("Only JPG, JPEG, and PNG images are supported.", 415)

    try:
        image = decode_image(uploaded.read())
    except ValueError as error:
        return error_response(str(error), 422)

    score = plant_score(image)
    if score < MIN_PLANT_SCORE:
        return error_response(
            "This does not look like a crop or plant image. Upload a clear leaf, fruit, stem, or crop image.",
            422,
        )

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    model_input = np.expand_dims(resized.astype(np.float32), axis=0)

    try:
        probabilities = np.asarray(_model.predict(model_input, verbose=0))[0]
    except Exception as error:
        app.logger.exception("Model inference failed")
        return error_response(f"The AI model could not analyze this image: {error}", 500)

    if probabilities.ndim != 1 or probabilities.size != len(_labels):
        return error_response("The model output does not match labels.json.", 500)

    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    if confidence < MIN_CONFIDENCE:
        return error_response(
            "The image is a plant, but the model is not confident enough to identify a disease. "
            "Use a clearer leaf image or confirm it with an agronomist.",
            422,
        )

    label = _labels[index]
    plant = label.split("___", 1)[0].split("__", 1)[0].replace("_", " ").strip()
    healthy = any(token in label.lower() for token in ("healthy", "normal", "no_disease"))
    information = disease_information(label, healthy)
    return jsonify(
        {
            "disease": display_label(label),
            "condition": display_label(label),
            "status": "Healthy" if healthy else "Potential risk",
            "plant": plant or "Plant",
            "confidence": round(confidence, 4),
            "description": prediction_description(label),
            **information,
        }
    )


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


if __name__ == "__main__":
    load_artifacts()
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
else:
    load_artifacts()
