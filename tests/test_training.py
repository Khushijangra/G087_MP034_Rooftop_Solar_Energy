import numpy as np
import pytest

from src.solar.train import select_best_run


def test_selects_lowest_mae():
    runs = [
        ("run1", "Linear", 4.69, "model1"),
        ("run2", "Ridge", 4.15, "model2"),
        ("run3", "Ridge2", 4.50, "model3"),
    ]

    best = select_best_run(runs)

    assert best[0] == "run2"
    assert best[2] == 4.15


def test_empty_runs_raise_error():
    with pytest.raises(ValueError):
        select_best_run([])


def test_invalid_mae_raises_error():
    runs = [
        ("run1", "Linear", 4.5, "model1"),
        ("run2", "Ridge", np.nan, "model2"),
    ]

    with pytest.raises(ValueError):
        select_best_run(runs)