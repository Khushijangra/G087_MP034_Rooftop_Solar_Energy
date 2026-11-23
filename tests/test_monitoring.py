import numpy as np
import pytest

from src.solar.monitoring import compute_psi


def test_psi_identical_distributions():
    values = np.arange(1, 101)
    assert compute_psi(values, values) == 0.0


def test_psi_detects_distribution_shift():
    expected = np.arange(1, 101)
    actual = np.arange(101, 201)

    assert compute_psi(expected, actual) > 0


@pytest.mark.parametrize(
    "expected,actual",
    [
        ([], [1, 2, 3]),
        ([1, 2, 3], []),
    ],
)
def test_psi_rejects_empty_inputs(expected, actual):
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_psi(expected, actual)


@pytest.mark.parametrize("invalid_bins", [0, 1, -5, 2.5, True])
def test_psi_rejects_invalid_bins(invalid_bins):
    with pytest.raises(ValueError, match="num_bins"):
        compute_psi([1, 2, 3], [1, 2, 3], num_bins=invalid_bins)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_psi_rejects_nonfinite_values(invalid_value):
    with pytest.raises(ValueError, match="finite"):
        compute_psi([1, invalid_value, 3], [1, 2, 3])


def test_psi_rejects_multidimensional_input():
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_psi([[1, 2], [3, 4]], [1, 2, 3])