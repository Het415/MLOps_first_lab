# Lab 1: CI for an ML Training Pipeline with GitHub Actions

![Pytest](https://github.com/Het415/MLOps_first_lab/actions/workflows/pytest.yml/badge.svg)
![Unittest](https://github.com/Het415/MLOps_first_lab/actions/workflows/unittest.yml/badge.svg)
![Train](https://github.com/Het415/MLOps_first_lab/actions/workflows/train.yml/badge.svg)

The original lab uses GitHub Actions to test a simple calculator. This version applies the same CI ideas to a small ML pipeline: every push tests the code, trains two models, enforces a quality gate, and publishes a model comparison.

## What I changed from the original lab

| Original | This version |
|---|---|
| `calculator.py` with four arithmetic functions | `metrics.py`: hand-written mean, std, scaling, confusion matrix, accuracy, precision, recall, F1 |
| No data or model | Breast Cancer Wisconsin dataset (569 samples, 30 features), Logistic Regression and Random Forest |
| Tests check arithmetic | Tests check input validation, compare my metrics against scikit-learn, verify reproducibility and saved artifacts |
| Single Python version | Pytest matrix on Python 3.10, 3.11, 3.12 with a 90% coverage gate |
| Pass/fail only | Training job fails if accuracy drops below a threshold, uploads models as artifacts, and writes a comparison table to the Actions job summary |

## Structure

```
MLOps_first_lab/
├── .github/workflows/
│   ├── pytest.yml       # tests + coverage on 3 Python versions
│   ├── unittest.yml     # unittest suite
│   └── train.yml        # train both models, gate on accuracy, compare
├── src/
│   ├── metrics.py       # metric and scaling functions
│   ├── train.py         # data split, model training, evaluation, saving
│   └── report.py        # builds the model comparison table
├── test/
│   ├── test_pytest.py   # metrics tests
│   ├── test_unittest.py # metrics tests (unittest)
│   └── test_model.py    # pipeline tests
├── pytest.ini
└── requirements.txt
```

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9861 | 0.9861 | 0.9861 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 |

80/20 stratified split, seed 42.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

pytest -v --cov=src --cov-report=term-missing
python -m unittest discover -s test -p "test_unittest.py" -t . -v

python -m src.train --model logreg --out artifacts/logreg
python -m src.train --model rf --out artifacts/rf
python -m src.report artifacts
```

## CI pipeline

1. **Pytest** runs all tests with coverage on three Python versions and uploads JUnit reports.
2. **Unittest** runs the unittest suite.
3. **Train and Compare Models** trains both models in parallel, fails the run if test accuracy is below 0.9 (adjustable when triggered manually), uploads each `model.joblib` and `metrics.json`, then posts a ranked comparison table to the run summary.
