import pytest

from forgelm.train import lr_at


def test_lr_schedule():
    assert lr_at(0, 100, 1.0, warmup=10) == pytest.approx(0.1)   # warmup starts low
    assert lr_at(9, 100, 1.0, warmup=10) == pytest.approx(1.0)   # reaches peak
    assert lr_at(100, 100, 1.0, warmup=10) == pytest.approx(0.1)  # decays to 10% of peak
    assert lr_at(55, 100, 1.0, warmup=10) < 1.0
