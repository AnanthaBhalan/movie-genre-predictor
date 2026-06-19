import os
import re
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(token for token in text.split() if token not in stop_words)


if __name__ == "__main__":
    data_path = os.path.join("Genre Classification Dataset", "train_data.txt")
    df = pd.read_csv(
        data_path,
        sep=" ::: ",
        engine="python",
        header=None,
        names=["ID", "Title", "Genre", "Plot"],
    )

    df["Cleaned_Plot"] = df["Plot"].apply(clean_text)

    X = df["Cleaned_Plot"]
    y = df["Genre"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_vect = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vect, y)

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Saved vectorizer.pkl and model.pkl")
