import numpy as np
import pytest

from exploratory_data_analysis._utils import flatten_or_raise


class TestFlattenOrRaise:
    @pytest.mark.parametrize(
        "shape",
        [(3,), (3, 1), (1, 3), (3, 1, 1), (3, 1, 1, 1)],
    )
    def test_flatten(self, shape: tuple[int, ...]):
        mat = np.ones(shape)
        res = flatten_or_raise(mat)

        assert res.ndim == 1
        assert res.size == mat.size

    @pytest.mark.parametrize(
        "shape",
        [(3, 2), (3, 1, 1, 2), (3, 1, 1, 1, 2)],
    )
    def test_raises(self, shape: tuple[int, ...]):
        mat = np.ones(shape)
        with pytest.raises(ValueError):
            _ = flatten_or_raise(mat)
