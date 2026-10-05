import math

import pytest

from src import metrics


@pytest.fixture
def labels():
    y_true = [1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [1, 0, 0, 1, 1, 0, 1, 0]
    return y_true, y_pred


@pytest.mark.parametrize(
    "values, expected",
    [([1, 2, 3, 4], 2.5), ([5], 5.0), ([-2, 2], 0.0), ((1.5, 2.5), 2.0)],
)
def test_mean(values, expected):
    assert metrics.mean(values) == pytest.approx(expected)


def test_std():
    assert metrics.std([2, 4, 4, 4, 5, 5, 7, 9]) == pytest.approx(2.0)
    assert metrics.std([3, 3, 3]) == 0.0


def test_min_max_scale():
    assert metrics.min_max_scale([10, 20, 30]) == pytest.approx([0.0, 0.5, 1.0])
    assert metrics.min_max_scale([7, 7]) == [0.0, 0.0]


def test_z_score():
    z = metrics.z_score([2, 4, 4, 4, 5, 5, 7, 9])
    assert z[0] == pytest.approx(-1.5)
    assert sum(z) == pytest.approx(0.0)
    assert metrics.z_score([1, 1, 1]) == [0.0, 0.0, 0.0]


def test_confusion(labels):
    assert metrics.confusion(*labels) == {"tp": 3, "fp": 1, "tn": 3, "fn": 1}


def test_classification_metrics(labels):
    assert metrics.accuracy(*labels) == pytest.approx(0.75)
    assert metrics.precision(*labels) == pytest.approx(0.75)
    assert metrics.recall(*labels) == pytest.approx(0.75)
    assert metrics.f1(*labels) == pytest.approx(0.75)


def test_zero_division_cases():
    assert metrics.precision([1, 1], [0, 0]) == 0.0
    assert metrics.recall([0, 0], [1, 1]) == 0.0
    assert metrics.f1([0, 0], [0, 0]) == 0.0


@pytest.mark.parametrize("bad", [[], "abc", [1, "2"], [True, 2], None])
def test_bad_values(bad):
    with pytest.raises(ValueError):
        metrics.mean(bad)


@pytest.mark.parametrize(
    "y_true, y_pred",
    [([1, 0], [1]), ([], []), ([1, 2], [1, 0]), ([True, 0], [1, 0])],
)
def test_bad_labels(y_true, y_pred):
    with pytest.raises(ValueError):
        metrics.accuracy(y_true, y_pred)


def test_no_nan():
    assert not math.isnan(metrics.f1([1, 0, 1], [0, 0, 0]))
