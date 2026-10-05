import json

import joblib
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src import report, train


@pytest.fixture(scope="module")
def data():
    return train.load_data()


def test_split_is_stratified(data):
    X_train, X_test, y_train, y_test = data
    assert len(X_train) + len(X_test) == 569
    assert abs(y_train.mean() - y_test.mean()) < 0.02


@pytest.mark.parametrize("name", train.MODELS)
def test_model_beats_threshold(data, name):
    X_train, X_test, y_train, y_test = data
    model = train.build_model(name).fit(X_train, y_train)
    assert train.evaluate(model, X_test, y_test)["accuracy"] >= 0.9


def test_metrics_match_sklearn(data):
    X_train, X_test, y_train, y_test = data
    model = train.build_model("logreg").fit(X_train, y_train)
    pred = model.predict(X_test)
    scores = train.evaluate(model, X_test, y_test)
    assert scores["accuracy"] == pytest.approx(accuracy_score(y_test, pred), abs=1e-4)
    assert scores["precision"] == pytest.approx(precision_score(y_test, pred), abs=1e-4)
    assert scores["recall"] == pytest.approx(recall_score(y_test, pred), abs=1e-4)
    assert scores["f1"] == pytest.approx(f1_score(y_test, pred), abs=1e-4)


def test_unknown_model():
    with pytest.raises(ValueError):
        train.build_model("svm")


def test_run_is_reproducible(tmp_path):
    a = train.run("rf", out=tmp_path / "a")
    b = train.run("rf", out=tmp_path / "b")
    assert a == b


def test_main_saves_artifacts(tmp_path):
    assert train.main(["--model", "logreg", "--out", str(tmp_path)]) == 0
    saved = json.loads((tmp_path / "metrics.json").read_text())
    model = joblib.load(tmp_path / "model.joblib")
    X_test = train.load_data()[1]
    assert saved["model"] == "logreg"
    assert set(model.predict(X_test)) <= {0, 1}


def test_main_fails_below_threshold(tmp_path):
    assert train.main(["--min-accuracy", "1.01", "--out", str(tmp_path)]) == 1


def test_report(tmp_path):
    for name in train.MODELS:
        train.run(name, out=tmp_path / name)
    results = report.load_results(tmp_path)
    assert results[0]["f1"] >= results[-1]["f1"]
    text = report.to_markdown(results)
    assert "Best model by F1" in text
    assert report.main([str(tmp_path)]) == 0


def test_report_empty(tmp_path):
    assert report.to_markdown(report.load_results(tmp_path)) == "No results found.\n"
