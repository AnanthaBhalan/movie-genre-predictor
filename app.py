import pickle
import re
import nltk
import streamlit as st
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(token for token in text.split() if token not in stop_words)


@st.cache_data
def load_model_and_vectorizer():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


def main():
    st.title("🎬 Movie Genre Predictor")
    st.subheader("Paste a movie plot summary and predict its genre.")

    plot_input = st.text_area("Movie Plot Summary", height=250)
    model, vectorizer = load_model_and_vectorizer()

    if st.button("Predict Genre"):
        if not plot_input.strip():
            st.error("Please enter a movie plot summary before predicting.")
            return

        cleaned_plot = clean_text(plot_input)
        features = vectorizer.transform([cleaned_plot])
        prediction = model.predict(features)
        st.success(f"Predicted genre: {prediction[0]}")


if __name__ == "__main__":
    main()
