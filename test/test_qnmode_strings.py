import pytest

from jaxqualin.qnmode import (
    str_to_lmnx,
    lmnx_to_string,
    lmnxs_to_string,
    long_str_to_lmnxs,
    long_str_to_strs,
    str_list_sort,
    long_str_sort,
    sort_lmnx,
    first_n_overtones_string,
    lmnx_sum_lm,
    qnm_string_m_reverse,
    qnm_string_l_reverse,
)


class TestStrToLmnx:
    def test_simple_mode(self):
        result = str_to_lmnx("2.2.0")
        assert result == [[2, 2, 0]]

    def test_retrograde_mode(self):
        result = str_to_lmnx("-2.2.0")
        assert result == [[-2, 2, 0]]

    def test_negative_m(self):
        result = str_to_lmnx("2.-2.0")
        assert result == [[2, -2, 0]]

    def test_overtone(self):
        result = str_to_lmnx("2.2.1")
        assert result == [[2, 2, 1]]

    def test_quadratic_mode(self):
        result = str_to_lmnx("2.2.0x3.3.0")
        assert result == [[2, 2, 0], [3, 3, 0]]

    def test_constant(self):
        result = str_to_lmnx("constant")
        assert result == "constant"


class TestLmnxToString:
    def test_simple_mode(self):
        assert lmnx_to_string([[2, 2, 0]]) == "2.2.0"

    def test_retrograde(self):
        assert lmnx_to_string([[-2, 2, 0]]) == "-2.2.0"

    def test_negative_m(self):
        assert lmnx_to_string([[2, -2, 0]]) == "2.-2.0"

    def test_quadratic(self):
        assert lmnx_to_string([[2, 2, 0], [3, 3, 0]]) == "2.2.0x3.3.0"

    def test_roundtrip(self):
        for s in ["2.2.0", "-2.2.0", "2.-2.0", "2.2.1", "2.2.0x3.3.0"]:
            assert lmnx_to_string(str_to_lmnx(s)) == s


class TestLmnxsToString:
    def test_multiple_modes(self):
        lmnxs = [[[2, 2, 0]], [[3, 3, 0]]]
        result = lmnxs_to_string(lmnxs)
        assert result == ["2.2.0", "3.3.0"]

    def test_quadratic_in_list(self):
        lmnxs = [[[2, 2, 0], [3, 3, 0]]]
        result = lmnxs_to_string(lmnxs)
        assert result == ["2.2.0x3.3.0"]


class TestLongStrToLmnxs:
    def test_two_modes(self):
        result = long_str_to_lmnxs("2.2.0_2.2.1")
        assert result == [[[2, 2, 0]], [[2, 2, 1]]]

    def test_single_mode(self):
        result = long_str_to_lmnxs("2.2.0")
        assert result == [[[2, 2, 0]]]


class TestLongStrToStrs:
    def test_split(self):
        result = long_str_to_strs("2.2.0_2.2.1_3.3.0")
        assert result == ["2.2.0", "2.2.1", "3.3.0"]

    def test_single(self):
        result = long_str_to_strs("2.2.0")
        assert result == ["2.2.0"]


class TestStrListSort:
    def test_sort(self):
        result = str_list_sort(["3.3.0", "2.2.0", "2.2.1"])
        assert result == ["2.2.0", "2.2.1", "3.3.0"]


class TestLongStrSort:
    def test_sort(self):
        result = long_str_sort("3.3.0_2.2.0_2.2.1")
        assert result == "2.2.0_2.2.1_3.3.0"


class TestSortLmnx:
    def test_sort(self):
        lmnx = [[3, 3, 0], [2, 2, 0]]
        result = sort_lmnx(lmnx)
        assert result == [[2, 2, 0], [3, 3, 0]]

    def test_already_sorted(self):
        lmnx = [[2, 2, 0], [3, 3, 0]]
        result = sort_lmnx(lmnx)
        assert result == [[2, 2, 0], [3, 3, 0]]


class TestFirstNOvertonesString:
    def test_n_zero(self):
        result = first_n_overtones_string(2, 2, 0)
        assert result == "2.2.0"

    def test_n_two(self):
        result = first_n_overtones_string(2, 2, 2)
        assert result == "2.2.0_2.2.1_2.2.2"

    def test_different_lm(self):
        result = first_n_overtones_string(3, 3, 1)
        assert result == "3.3.0_3.3.1"


class TestLmnxSumLm:
    def test_single_mode(self):
        l_sum, m_sum = lmnx_sum_lm([[2, 2, 0]])
        assert l_sum == 2
        assert m_sum == 2

    def test_quadratic_mode(self):
        l_sum, m_sum = lmnx_sum_lm([[2, 2, 0], [3, 3, 0]])
        assert l_sum == 5
        assert m_sum == 5

    def test_constant(self):
        l_sum, m_sum = lmnx_sum_lm("constant")
        assert l_sum == 0
        assert m_sum == 0


class TestQnmStringMReverse:
    def test_positive_m(self):
        result = qnm_string_m_reverse("2.2.0")
        assert result == "2.-2.0"

    def test_negative_m(self):
        result = qnm_string_m_reverse("2.-2.0")
        assert result == "2.2.0"

    def test_constant(self):
        result = qnm_string_m_reverse("constant")
        assert result == "constant"

    def test_m_zero_becomes_sentinel(self):
        result = qnm_string_m_reverse("2.0.0")
        assert result == "2.-99.0"

    def test_sentinel_becomes_zero(self):
        result = qnm_string_m_reverse("2.-99.0")
        assert result == "2.0.0"


class TestQnmStringLReverse:
    def test_positive_l(self):
        result = qnm_string_l_reverse("2.2.0")
        assert result == "-2.2.0"

    def test_negative_l(self):
        result = qnm_string_l_reverse("-2.2.0")
        assert result == "2.2.0"

    def test_constant(self):
        result = qnm_string_l_reverse("constant")
        assert result == "constant"

    def test_quadratic(self):
        result = qnm_string_l_reverse("2.2.0x3.3.0")
        assert result == "-2.2.0x-3.3.0"
