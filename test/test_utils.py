import numpy as np
import pickle
import tempfile
import os

import pytest

from jaxqualin.utils import (
    interweave,
    max_consecutive_trues,
    sorti,
    sign0,
    get_retrofac,
    get_m,
    load_pickle_file,
    all_close_to,
    _M_SENTINEL,
)


class TestInterweave:
    def test_basic(self):
        a = np.array([1.0, 3.0, 5.0])
        b = np.array([2.0, 4.0, 6.0])
        result = np.array(interweave(a, b))
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_single_element(self):
        a = np.array([1.0])
        b = np.array([2.0])
        result = np.array(interweave(a, b))
        expected = np.array([1.0, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_length(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([5.0, 6.0, 7.0, 8.0])
        result = interweave(a, b)
        assert len(result) == 8


class TestMaxConsecutiveTrues:
    def test_all_true(self):
        arr = [True, True, True, True]
        start, end = max_consecutive_trues(arr)
        assert start == 0
        assert end == 3

    def test_all_false(self):
        arr = [False, False, False]
        start, end = max_consecutive_trues(arr)
        assert start == 0
        assert end == 0

    def test_mixed(self):
        arr = [False, True, True, True, False]
        start, end = max_consecutive_trues(arr)
        assert end - start >= 3

    def test_with_tolerance(self):
        arr = [True, True, False, True, True]
        start, end = max_consecutive_trues(arr, tol=0.8)
        assert end > start

    def test_empty(self):
        arr = []
        start, end = max_consecutive_trues(arr)
        assert start == 0
        assert end == 0


class TestSorti:
    def test_sort_by_first_element(self):
        li = [[1, 'a'], [3, 'b'], [2, 'c']]
        result = sorti(li, 0)
        assert result[0][0] == 3
        assert result[-1][0] == 1

    def test_sort_by_second_element(self):
        li = [[1, 30], [2, 10], [3, 20]]
        result = sorti(li, 1)
        assert result[0][1] == 30
        assert result[-1][1] == 10


class TestSign0:
    def test_zero(self):
        assert sign0(0) == 1

    def test_positive(self):
        assert float(sign0(5)) == 1.0

    def test_negative(self):
        assert float(sign0(-3)) == -1.0


class TestGetRetrofac:
    def test_retro_true(self):
        assert get_retrofac(True) == -1

    def test_retro_false(self):
        assert get_retrofac(False) == 1


class TestGetM:
    def test_sentinel_value(self):
        assert get_m(_M_SENTINEL) == 0

    def test_regular_value(self):
        assert get_m(2) == 2

    def test_zero(self):
        assert get_m(0) == 0

    def test_negative(self):
        assert get_m(-2) == -2


class TestLoadPickleFile:
    def test_roundtrip(self):
        data = {"key": [1, 2, 3], "value": "test"}
        with tempfile.NamedTemporaryFile(suffix=".pickle", delete=False) as f:
            pickle.dump(data, f)
            tmppath = f.name
        try:
            loaded = load_pickle_file(tmppath)
            assert loaded == data
        finally:
            os.unlink(tmppath)


class TestAllCloseTo:
    def test_all_close(self):
        arr = np.array([1.0, 1.0, 1.0])
        assert all_close_to(arr, 1.0)

    def test_not_close(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert not all_close_to(arr, 1.0)

    def test_close_with_tolerance(self):
        arr = np.array([1.0, 1.0 + 1e-10, 1.0 - 1e-10])
        assert all_close_to(arr, 1.0)
