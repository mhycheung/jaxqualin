"""Tests for selection utilities: flattest_region_quadrature, start_of_flat_region, etc."""
import numpy as np
import pytest
from types import SimpleNamespace

from jaxqualin.selection import (
    flattest_region_quadrature,
    start_of_flat_region,
    closest_free_mode_distance,
)


# ---------------------------------------------------------------------------
# flattest_region_quadrature
# ---------------------------------------------------------------------------

class TestFlatRegionQuadrature:

    def test_flat_in_middle(self):
        """Arrays that are flat in the middle and noisy at edges."""
        n = 200
        arr1 = np.ones(n) * 5.0
        arr2 = np.ones(n) * 1.0

        # Add noise at edges
        arr1[:30] += np.random.RandomState(42).randn(30) * 2
        arr1[170:] += np.random.RandomState(43).randn(30) * 2
        arr2[:30] += np.random.RandomState(44).randn(30) * 2
        arr2[170:] += np.random.RandomState(45).randn(30) * 2

        window = 40
        idx, fluc, start_idx = flattest_region_quadrature(window, arr1, arr2)

        # The flattest region should overlap with the middle section [30, 170)
        assert 20 <= idx <= 140
        assert fluc < 0.5

    def test_all_flat(self):
        """When both arrays are constant, the entire range should be selected."""
        n = 100
        arr1 = np.ones(n) * 3.0
        arr2 = np.ones(n) * 1.0

        window = 20
        idx, fluc, start_idx = flattest_region_quadrature(window, arr1, arr2)

        assert fluc < 1e-10
        assert start_idx == 0

    def test_linear_ramp_no_large_flat(self):
        """Linearly increasing arrays have growing spread in any window.

        The fluctuation should grow with window size, so the function
        should not find a large flat region (start_flat_indx stays -1).
        """
        n = 200
        arr1 = np.linspace(0, 10, n)
        arr2 = np.linspace(0, 5, n)

        window = 80
        idx, fluc, start_idx = flattest_region_quadrature(
            window, arr1, arr2, fluc_tol=0.05)

        # With a large window and tight tolerance, a linear ramp should not
        # have a flat region
        assert start_idx < 0

    def test_length_mismatch_raises(self):
        arr1 = np.ones(10)
        arr2 = np.ones(20)
        with pytest.raises(Exception, match="length"):
            flattest_region_quadrature(5, arr1, arr2)


# ---------------------------------------------------------------------------
# start_of_flat_region
# ---------------------------------------------------------------------------

class TestStartOfFlatRegion:

    def test_returns_nan_for_no_flat_region(self):
        """Linear ramp with tight tolerance should return nan."""
        n = 200
        arr1 = np.linspace(0, 10, n)
        arr2 = np.linspace(0, 5, n)

        result = start_of_flat_region(80, arr1, arr2, fluc_tol=0.05)
        assert np.isnan(result)

    def test_returns_valid_index_for_flat(self):
        """Constant arrays should return index 0."""
        n = 100
        arr1 = np.ones(n) * 3.0
        arr2 = np.ones(n) * 1.0
        result = start_of_flat_region(20, arr1, arr2)
        assert result == 0


# ---------------------------------------------------------------------------
# closest_free_mode_distance
# ---------------------------------------------------------------------------

class TestClosestFreeModeDistance:

    def test_exact_match_zero_distance(self):
        """If the free mode frequency matches the target, distance = 0."""
        target_omegar = 0.5
        target_omegai = -0.1

        result_full = SimpleNamespace(
            omega_dict={
                "real": {"omega_r_free_0": np.array([target_omegar])},
                "imag": {"omega_i_free_0": np.array([target_omegai])},
            }
        )
        target_mode = SimpleNamespace(
            omegar=target_omegar, omegai=target_omegai)

        dist = closest_free_mode_distance(result_full, target_mode)
        assert np.allclose(dist, 0.0, atol=1e-14)

    def test_nonzero_distance(self):
        """Distance should be positive for non-matching frequencies."""
        result_full = SimpleNamespace(
            omega_dict={
                "real": {"omega_r_free_0": np.array([0.5])},
                "imag": {"omega_i_free_0": np.array([-0.1])},
            }
        )
        target_mode = SimpleNamespace(omegar=0.6, omegai=-0.2)

        dist = closest_free_mode_distance(result_full, target_mode)
        expected = np.sqrt((0.5 - 0.6)**2 + (-0.1 - (-0.2))**2)
        assert np.allclose(dist, expected, rtol=1e-10)

    def test_picks_closest_of_multiple(self):
        """With two free modes, should pick the closest one."""
        result_full = SimpleNamespace(
            omega_dict={
                "real": {
                    "omega_r_free_0": np.array([0.3]),
                    "omega_r_free_1": np.array([0.5]),
                },
                "imag": {
                    "omega_i_free_0": np.array([-0.2]),
                    "omega_i_free_1": np.array([-0.1]),
                },
            }
        )
        target_mode = SimpleNamespace(omegar=0.5, omegai=-0.1)

        dist = closest_free_mode_distance(result_full, target_mode)
        assert np.allclose(dist, 0.0, atol=1e-14)
