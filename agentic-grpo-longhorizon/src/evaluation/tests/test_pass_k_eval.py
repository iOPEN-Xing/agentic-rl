import pytest

from src.evaluation.pass_k_eval import (
    estimate_pass_at_k,
    estimate_pass_power_k,
)


def test_pass_power_k_estimates_all_k_successes():
    assert estimate_pass_power_k(8, 4, 1) == pytest.approx(0.5)
    assert estimate_pass_power_k(8, 4, 4) == pytest.approx(1.0 / 70.0)
    assert estimate_pass_power_k(8, 3, 4) == 0.0
    assert estimate_pass_power_k(8, 8, 4) == 1.0


def test_pass_at_k_remains_at_least_one_success_metric():
    assert estimate_pass_at_k(8, 4, 4) == pytest.approx(69.0 / 70.0)
    assert estimate_pass_at_k(8, 0, 4) == 0.0
    assert estimate_pass_at_k(8, 8, 4) == 1.0


@pytest.mark.parametrize(
    ("n", "c", "k"),
    [
        (0, 0, 1),
        (4, -1, 1),
        (4, 5, 1),
        (4, 2, 0),
        (4, 2, 5),
    ],
)
def test_pass_estimators_validate_counts(n, c, k):
    with pytest.raises(ValueError):
        estimate_pass_at_k(n, c, k)
    with pytest.raises(ValueError):
        estimate_pass_power_k(n, c, k)
