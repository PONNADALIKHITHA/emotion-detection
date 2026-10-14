from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).with_name("baseline_model.joblib")

st.set_page_config(page_title="Feeling Finder", page_icon="F", layout="centered")

st.markdown(
    """
    <style>
      .stApp {
        background-color: #f5f7f2;
        background-image: radial-gradient(#cbd8ce 0.7px, transparent 0.7px);
        background-size: 20px 20px;
      }
      .block-container { max-width: 850px; padding-top: 2.5rem; }
      h1, h2, h3 { color: #18352e; }
      h1 { font-family: Georgia, 'Times New Roman', serif; font-weight: 500; }
      [data-testid="stForm"] {
        border: 1px solid #dce5dd;
        border-radius: 8px;
        background: #fff;
        padding: 1.25rem;
      }
      [data-testid="stFormSubmitButton"] button {
        background: #176b52;
        border-color: #176b52;
        color: white;
      }
      [data-testid="stFormSubmitButton"] button:hover {
        background: #10543f;
        border-color: #10543f;
        color: white;
      }
      [data-testid="stMetric"] {
        border-left: 3px solid #176b52;
        padding: 0.25rem 0 0.25rem 0.75rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.caption("EMOTION CLASSIFIER  /  SIX LABELS")
st.title("What feeling is inside the words?")
st.write("Paste a thought, a note, or a sentence to see what the model notices.")


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "baseline_model.joblib is missing. Run `python baseline.py` and add the generated file to the repository."
        )
    return joblib.load(MODEL_PATH)


with st.form("emotion_input"):
    text = st.text_area(
        "Your text",
        max_chars=1000,
        height=170,
        placeholder="Type or paste something here...",
    )
    submitted = st.form_submit_button("Analyze text", type="primary", use_container_width=True)

if submitted:
    if not text.strip():
        st.warning("Enter some text to analyze.")
    else:
        try:
            model, labels = load_model()
            probabilities = model.predict_proba([text.strip()])[0]
            scores = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)
        except (FileNotFoundError, ValueError, OSError) as error:
            st.error(str(error))
        else:
            top_label, top_score = scores[0]
            st.subheader("Reading")
            st.metric("Strongest signal", top_label.capitalize(), f"{top_score:.1%}")

            st.markdown("#### Emotion scores")
            columns = st.columns(2)
            for index, (label, score) in enumerate(scores):
                with columns[index % 2]:
                    st.progress(float(score), text=f"{label.capitalize()} · {score:.1%}")

st.caption("TF-IDF + logistic regression · trained on the dair-ai/emotion dataset")