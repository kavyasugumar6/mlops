import argparse
import logging
import os
from pathlib import Path
from typing import Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def generate_synthetic_stroke_data(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Create a synthetic stroke-like dataset with mixed categorical and numeric features.
    This keeps the example self-contained for local/offline runs.
    """
    rng = np.random.default_rng(random_state)

    data = pd.DataFrame(
        {
            "age": rng.normal(loc=55, scale=12, size=n_samples).clip(18, 90),
            "bmi": rng.normal(loc=28, scale=6, size=n_samples).clip(16, 48),
            "avg_glucose_level": rng.normal(loc=110, scale=35, size=n_samples).clip(60, 280),
            "hypertension": rng.binomial(1, 0.2, size=n_samples),
            "heart_disease": rng.binomial(1, 0.15, size=n_samples),
            "gender": rng.choice(["Male", "Female"], size=n_samples, p=[0.52, 0.48]),
            "ever_married": rng.choice(["Yes", "No"], size=n_samples, p=[0.7, 0.3]),
            "work_type": rng.choice(["Private", "Self-employed", "Govt_job", "Never_worked"], size=n_samples),
            "Residence_type": rng.choice(["Urban", "Rural"], size=n_samples),
            "smoking_status": rng.choice(["formerly smoked", "never smoked", "smokes", "Unknown"], size=n_samples),
        }
    )

    # Non-linear stroke risk inspired by known factors
    logits = (
        0.04 * (data["age"] - 45)
        + 0.06 * (data["bmi"] - 26)
        + 0.05 * (data["avg_glucose_level"] - 100)
        + 0.8 * data["hypertension"]
        + 0.6 * data["heart_disease"]
        + 0.4 * (data["smoking_status"].isin(["smokes", "formerly smoked"])).astype(int)
    )
    prob = 1 / (1 + np.exp(-logits / 50))
    data["stroke"] = rng.binomial(1, prob)
    return data


def build_pipeline(random_state: int) -> Pipeline:
    categorical_features = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]
    numeric_features = [
        "age",
        "bmi",
        "avg_glucose_level",
        "hypertension",
        "heart_disease",
    ]

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", numeric_transformer, numeric_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=220,
        max_depth=None,
        random_state=random_state,
        class_weight="balanced",
        min_samples_split=4,
    )

    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return clf


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Tuple[dict, np.ndarray]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics, y_pred


def train(args: argparse.Namespace) -> None:
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("stroke-risk")

    data = generate_synthetic_stroke_data(n_samples=args.samples, random_state=args.random_state)
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=args.random_state, stratify=data["stroke"])

    X_train = train_df.drop(columns=["stroke"])
    y_train = train_df["stroke"]
    X_test = test_df.drop(columns=["stroke"])
    y_test = test_df["stroke"]

    pipeline = build_pipeline(random_state=args.random_state)

    with mlflow.start_run(run_name="rf-stroke"):
        mlflow.log_params(
            {
                "n_estimators": pipeline.named_steps["model"].n_estimators,
                "min_samples_split": pipeline.named_steps["model"].min_samples_split,
                "class_weight": "balanced",
                "random_state": args.random_state,
                "samples": args.samples,
            }
        )

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics, _ = evaluate(y_test, y_prob, threshold=args.threshold)
        mlflow.log_metrics(metrics)

        signature = mlflow.models.infer_signature(X_train, pipeline.predict_proba(X_train))
        mlflow.sklearn.log_model(pipeline, artifact_path="model", signature=signature, input_example=X_train.iloc[:3])

        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifacts_dir / "model.pkl"
        sample_path = artifacts_dir / "reference_data.csv"

        joblib.dump(pipeline, model_path)
        train_df.to_csv(sample_path, index=False)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact(sample_path)

        LOGGER.info("Metrics: %s", metrics)
        LOGGER.info("Saved model to %s", model_path.resolve())
        LOGGER.info("Saved reference data to %s", sample_path.resolve())
        LOGGER.info("MLflow run: %s", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stroke prediction model with MLflow logging")
    parser.add_argument("--samples", type=int, default=1500, help="Number of synthetic samples to generate")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for evaluation")
    args = parser.parse_args()
    train(args)
