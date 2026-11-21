#!/usr/bin/env python3
"""
Week 11 HPC assignment
Project: Diabetes risk classification using the Pima Indians Diabetes dataset
Student: Esther Kerubo

Loads a local diabetes CSV (preferred) or downloads the public dataset if
permitted, trains Logistic Regression and Random Forest classifiers, evaluates
training/test performance, and saves the best model and output files.
"""

import os
import glob
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_dataset():
    candidates = glob.glob("*.csv")
    if candidates:
        path = candidates[0]
        print(f"Loading local dataset: {path}")
        try:
            data = pd.read_csv(path)
            if "Outcome" not in data.columns and data.shape[1] == 9:
                data.columns = COLUMNS
            return data
        except Exception as exc:
            print(f"Could not read {path}: {exc}")

    print("No local CSV found. Attempting to download the public Pima dataset...")
    try:
        data = pd.read_csv(DATA_URL, header=None, names=COLUMNS)
        data.to_csv("pima_diabetes_dataset.csv", index=False)
        return data
    except Exception as exc:
        raise FileNotFoundError(
            "Dataset unavailable. Place a Pima diabetes CSV in the job "
            "working directory and submit again. Download attempt failed: "
            f"{exc}"
        )


def main():
    print("=" * 65)
    print("WEEK 11 HPC JOB: DIABETES RISK CLASSIFICATION")
    print("=" * 65)

    data = load_dataset()
    if "Outcome" not in data.columns:
        if data.shape[1] == 9:
            data.columns = COLUMNS
        else:
            raise ValueError("Expected an Outcome target column and 8 predictors.")

    feature_columns = [c for c in COLUMNS if c != "Outcome"]
    missing = [c for c in feature_columns + ["Outcome"] if c not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    data = data[feature_columns + ["Outcome"]].copy()
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Outcome"])
    data["Outcome"] = data["Outcome"].astype(int)
    data = data[data["Outcome"].isin([0, 1])]

    # In this dataset, zero in selected clinical measurements conventionally
    # represents a missing/invalid measurement; preserve zero for pregnancies.
    zero_as_missing = [
        "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
    ]
    data[zero_as_missing] = data[zero_as_missing].replace(0, np.nan)

    X = data[feature_columns]
    y = data["Outcome"]
    print(f"Usable observations: {len(data)}")
    print(f"Features: {feature_columns}")
    print(f"Class counts:\n{y.value_counts().sort_index().to_string()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Training observations: {len(X_train)}")
    print(f"Testing observations: {len(X_test)}")

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300,
                random_state=RANDOM_STATE,
                min_samples_leaf=2,
                n_jobs=2,
                class_weight="balanced"
            ))
        ])
    }

    rows = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        for split, x_part, y_part in [
            ("Training", X_train, y_train),
            ("Test", X_test, y_test)
        ]:
            prediction = model.predict(x_part)
            probabilities = model.predict_proba(x_part)[:, 1]
            rows.append({
                "Model": name,
                "Dataset": split,
                "Accuracy": accuracy_score(y_part, prediction),
                "Precision": precision_score(y_part, prediction, zero_division=0),
                "Recall": recall_score(y_part, prediction, zero_division=0),
                "F1": f1_score(y_part, prediction, zero_division=0),
                "ROC_AUC": roc_auc_score(y_part, probabilities)
            })

        if split == "Test":
            print(f"\n{name} test classification report:")
            print(classification_report(y_test, prediction, zero_division=0))
            print("Confusion matrix:")
            print(confusion_matrix(y_test, prediction))

    results = pd.DataFrame(rows)
    print("\nPERFORMANCE RESULTS")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    results.to_csv(OUTPUT_DIR / "model_performance.csv", index=False)

    test_rows = results[results["Dataset"] == "Test"]
    best_name = test_rows.loc[test_rows["ROC_AUC"].idxmax(), "Model"]
    best_model = models[best_name]
    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": feature_columns,
        "target_name": "Outcome",
        "project": "Pima Indians diabetes risk classification"
    }
    joblib.dump(artifact, OUTPUT_DIR / "diabetes_risk_best_model.joblib")

    # Reload model and demonstrate a prediction on a held-out, unseen row.
    reloaded = joblib.load(OUTPUT_DIR / "diabetes_risk_best_model.joblib")
    example = X_test.iloc[[0]]
    predicted_class = int(reloaded["model"].predict(example)[0])
    predicted_probability = float(reloaded["model"].predict_proba(example)[0, 1])

    print(f"\nBest model by test ROC-AUC: {best_name}")
    print(f"Example predicted class (0=no diabetes, 1=diabetes): {predicted_class}")
    print(f"Example predicted probability for class 1: {predicted_probability:.5f}")
    print(f"Outputs saved in: {OUTPUT_DIR.resolve()}")
    print("JOB COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
