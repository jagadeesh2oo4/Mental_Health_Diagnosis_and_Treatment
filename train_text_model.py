# train_text_model.py
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Config
DATA_PATH = os.path.join("data", "tweet_emotions.csv")  # update if needed
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "text_model.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(path):
    # expects CSV with at least two columns: 'text' and 'label'
    df = pd.read_csv(path)
    # if your CSV uses different column names, change here:
    if 'text' not in df.columns or 'label' not in df.columns:
        # try other sensible defaults
        if 'tweet' in df.columns:
            df = df.rename(columns={'tweet':'text'})
        elif 'content' in df.columns:
            df = df.rename(columns={'content':'text'})

        if 'sentiment' in df.columns:
            df = df.rename(columns={'sentiment':'label'})

    df = df.dropna(subset=['text','label'])
    return df

def train():
    df = load_data(DATA_PATH)
    X = df['text'].astype(str)
    y = df['label'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print("Classification report on test set:")
    print(classification_report(y_test, preds))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved text model to {MODEL_PATH}")

if __name__ == "__main__":
    train()
