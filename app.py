from pathlib import Path

from flask import Flask, jsonify, render_template, request
from transformers import pipeline

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "emotion_model"

if not MODEL_DIR.is_dir():
    raise RuntimeError("Model not found. Run train_transformer.py first.")

classifier = pipeline("text-classification", model=str(MODEL_DIR), top_k=None)
app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Enter some text to analyze."}), 400
    if len(text) > 1000:
        return jsonify({"error": "Keep the text under 1,000 characters."}), 400

    scores = classifier(text.strip())[0]
    scores.sort(key=lambda result: result["score"], reverse=True)
    return jsonify({"scores": scores})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)