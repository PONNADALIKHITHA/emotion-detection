# Emotion Detection from Text

Detects 6 emotions (sadness, joy, love, anger, fear, surprise) using the `dair-ai/emotion` dataset.

## Setup
    pip install -r requirements.txt

## Run
    python baseline.py            # quick TF-IDF + Logistic Regression baseline
    python train_transformer.py   # fine-tune DistilBERT (GPU recommended)
    python predict.py "I am so excited about tomorrow!"
    python app.py                 # start the website at http://127.0.0.1:5000
