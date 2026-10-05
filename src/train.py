import argparse
import json
import sys
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import metrics

MODELS = ["logreg", "rf"]


def load_data(test_size=0.2, seed=42):
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def build_model(name="logreg", seed=42):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def evaluate(model, X, y):
    y_true = [int(v) for v in y]
    y_pred = [int(v) for v in model.predict(X)]
    return {
        "accuracy": round(metrics.accuracy(y_true, y_pred), 4),
        "precision": round(metrics.precision(y_true, y_pred), 4),
        "recall": round(metrics.recall(y_true, y_pred), 4),
        "f1": round(metrics.f1(y_true, y_pred), 4),
    }


def run(name="logreg", test_size=0.2, seed=42, out="artifacts"):
    X_train, X_test, y_train, y_test = load_data(test_size, seed)
    model = build_model(name, seed)
    model.fit(X_train, y_train)
    scores = evaluate(model, X_test, y_test)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model.joblib")
    result = {"model": name, "test_size": test_size, "seed": seed, **scores}
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a breast cancer classifier.")
    parser.add_argument("--model", choices=MODELS, default="logreg")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-accuracy", type=float, default=0.9)
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args(argv)

    result = run(args.model, args.test_size, args.seed, args.out)
    print(json.dumps(result, indent=2))

    if result["accuracy"] < args.min_accuracy:
        print(f"Accuracy {result['accuracy']} is below threshold {args.min_accuracy}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
