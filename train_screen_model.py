import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# CLEAN COLUMN NAMES MAP
RENAME_MAP = {
    'who -bmi': 'who_bmi',
    'who-bmi': 'who_bmi',
    'phq_ score': 'phq_score',
    'phq score': 'phq_score',
    'ad-score': 'gad_score',
    'anxiet severity': 'anxiety_severity',
    'depression_ treatment': 'depression_treatment',
}

TARGET_COLS = [
    "depression_severity",
    "depressiveness",
    "suicidal",
    "depression_diagnosis",
    "depression_treatment",
    "anxiety_severity",
    "anxiousness",
    "anxiety_diagnosis",
    "anxiety_treatment",
    "sleepiness"
]

# AGE ADDED HERE
FEATURE_COLS = [
    "age",
    "school_year",
    "gender",
    "bmi",
    "who_bmi",
    "phq_score",
    "gad_score",
    "epworth_score"
]

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CSV_PATH = "data/depression_anxiety_data.csv"


def load_data():
    df = pd.read_csv(CSV_PATH)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns=RENAME_MAP)

    # Ensure age exists or throw error
    if "age" not in df.columns:
        raise ValueError("Dataset missing required column: 'age'")

    # Remove id column if exists
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Verify all required columns exist
    for col in FEATURE_COLS + TARGET_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    X = df[FEATURE_COLS].copy()
    Y = df[TARGET_COLS].copy()

    return X, Y


def encode_targets(Y):
    label_encoders = {}
    Y_encoded = pd.DataFrame()

    for col in TARGET_COLS:
        le = LabelEncoder()
        Y_encoded[col] = le.fit_transform(Y[col].astype(str))
        label_encoders[col] = list(le.classes_)

    with open(f"{MODEL_DIR}/screen_classes.json", "w") as f:
        json.dump(label_encoders, f, indent=4)

    return Y_encoded


def train():
    X, Y = load_data()
    Y_enc = encode_targets(Y)

    numeric_cols = ["age", "bmi", "phq_score", "gad_score", "epworth_score"]
    categorical_cols = ["school_year", "gender", "who_bmi"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", MultiOutputClassifier(RandomForestClassifier(n_estimators=250)))
    ])

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_enc, test_size=0.2, random_state=42
    )

    print("Training model...")
    model.fit(X_train, Y_train)

    joblib.dump(model, f"{MODEL_DIR}/screen_model.joblib")
    print("Saved: models/screen_model.joblib")

    print("Saved: models/screen_classes.json")


if __name__ == "__main__":
    train()
