"""Run emotion detection on text using the fine-tuned model."""
import sys
from transformers import pipeline

clf = pipeline("text-classification", model="emotion_model", top_k=None)

def detect(text: str):
    scores = clf(text)[0]
    return sorted(scores, key=lambda s: s["score"], reverse=True)

if __name__ == "__main__":
    texts = sys.argv[1:] or [
        "I can't believe I finally got the job, I'm so happy!",
        "I feel so alone and nothing seems to matter anymore.",
        "That noise in the dark house terrified me.",
    ]
    for t in texts:
        results = detect(t)
        top = results[0]
        print(f"\n{t}\n -> {top['label']} ({top['score']:.2%})")
        for r in results[1:3]:
            print(f"    {r['label']}: {r['score']:.2%}")
