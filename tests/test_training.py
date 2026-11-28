from src.solar.train import get_data_sha256
from pathlib import Path
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
def test_dataset_hash_is_64_characters(tmp_path):
    data_file = tmp_path / "train.csv"
    data_file.write_text("solar,power\n1,100\n")

    result = get_data_sha256(str(data_file))

    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


def test_same_dataset_produces_same_hash(tmp_path):
    data_file = tmp_path / "train.csv"
    data_file.write_text("solar,power\n1,100\n")

    first_hash = get_data_sha256(str(data_file))
    second_hash = get_data_sha256(str(data_file))

    assert first_hash == second_hash


def test_modified_dataset_produces_different_hash(tmp_path):
    data_file = tmp_path / "train.csv"
    data_file.write_text("solar,power\n1,100\n")

    first_hash = get_data_sha256(str(data_file))

    data_file.write_text("solar,power\n2,200\n")
    second_hash = get_data_sha256(str(data_file))

    assert first_hash != second_hash