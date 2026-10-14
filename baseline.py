"""Baseline: TF-IDF + Logistic Regression (fast, no GPU needed)."""
import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

ds = load_dataset("dair-ai/emotion")  # sadness, joy, love, anger, fear, surprise
label_names = ds["train"].features["label"].names

model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=1000, C=5.0),
)
model.fit(ds["train"]["text"], ds["train"]["label"])

preds = model.predict(ds["test"]["text"])
print(classification_report(ds["test"]["label"], preds, target_names=label_names))

joblib.dump((model, label_names), "baseline_model.joblib")
print("Saved baseline_model.joblib")
