import math


def _check_values(values):
    if not isinstance(values, (list, tuple)) or len(values) == 0:
        raise ValueError("Input must be a non-empty list or tuple.")
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("All values must be numbers.")


def _check_labels(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_true) == 0:
        raise ValueError("Inputs must not be empty.")
    for v in list(y_true) + list(y_pred):
        if isinstance(v, bool) or v not in (0, 1):
            raise ValueError("Labels must be 0 or 1.")


def mean(values):
    _check_values(values)
    return sum(values) / len(values)


def std(values):
    _check_values(values)
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def min_max_scale(values):
    _check_values(values)
    lo, hi = min(values), max(values)
    if lo == hi:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def z_score(values):
    _check_values(values)
    m, s = mean(values), std(values)
    if s == 0:
        return [0.0] * len(values)
    return [(v - m) / s for v in values]


def confusion(y_true, y_pred):
    _check_labels(y_true, y_pred)
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def accuracy(y_true, y_pred):
    c = confusion(y_true, y_pred)
    return (c["tp"] + c["tn"]) / len(y_true)


def precision(y_true, y_pred):
    c = confusion(y_true, y_pred)
    total = c["tp"] + c["fp"]
    return c["tp"] / total if total else 0.0


def recall(y_true, y_pred):
    c = confusion(y_true, y_pred)
    total = c["tp"] + c["fn"]
    return c["tp"] / total if total else 0.0


def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) else 0.0
