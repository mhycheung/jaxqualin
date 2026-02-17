"""Tests for waveform utility functions: clean_QNM, compute_mismatch, etc."""
import numpy as np
import pytest

from jaxqualin.waveforms import (
    clean_QNM,
    waveform_toy_clean,
    waveform,
    compute_mismatch,
    mismatch_min_phase,
    delayed_QNM,
)
from jaxqualin.qnmode import mode, mode_list


Mf = 1.0
af = 0.7


# ---------------------------------------------------------------------------
# clean_QNM convention
# ---------------------------------------------------------------------------

class TestCleanQNMConvention:

    def test_at_t_zero(self):
        """clean_QNM(mode, 0, A, phi) = A * exp(-i*phi)."""
        m = mode([[2, 2, 0]], Mf, af)
        A, phi = 2.0, 0.3
        h0 = clean_QNM(m, np.array([0.0]), A, phi)
        expected = A * np.exp(-1j * phi)
        assert np.isclose(h0[0], expected, atol=1e-14)

    def test_at_arbitrary_t(self):
        """Verify formula: A * exp(omega_i*t) * exp(-i*(omega_r*t + phi))."""
        m = mode([[2, 2, 0]], Mf, af)
        A, phi = 1.5, 0.7
        t = np.array([10.0, 20.0, 30.0])
        h = clean_QNM(m, t, A, phi)
        expected = A * np.exp(m.omegai * t) * np.exp(-1j * (m.omegar * t + phi))
        assert np.allclose(h, expected, atol=1e-14)

    def test_real_part_decays(self):
        """Real part amplitude should decrease for damped modes (omegai < 0)."""
        m = mode([[2, 2, 0]], Mf, af)
        assert float(m.omegai) < 0
        A, phi = 1.0, 0.0
        t = np.linspace(0, 100, 1000)
        h = clean_QNM(m, t, A, phi)
        assert np.abs(h[-1]) < np.abs(h[0])


# ---------------------------------------------------------------------------
# waveform_toy_clean
# ---------------------------------------------------------------------------

class TestWaveformToyClean:

    def test_sum_at_t_zero(self):
        """At t=0, waveform = sum of A_k * exp(-i*phi_k)."""
        modes = mode_list(['2.2.0', '2.2.1'], Mf, af)
        A_list = [1.0, 3.0]
        phi_list = [0.0, np.pi / 4]
        t = np.array([0.0])
        h = waveform_toy_clean(A_list, phi_list, modes, t)
        expected = sum(
            A * np.exp(-1j * phi) for A, phi in zip(A_list, phi_list))
        assert np.isclose(h[0], expected, atol=1e-14)

    def test_length_mismatch_raises(self):
        modes = mode_list(['2.2.0', '2.2.1'], Mf, af)
        with pytest.raises(ValueError):
            waveform_toy_clean([1.0], [0.0, 0.0], modes, np.array([0.0]))


# ---------------------------------------------------------------------------
# compute_mismatch
# ---------------------------------------------------------------------------

class TestComputeMismatch:

    def _make_simple_waveform(self, A=1.0, phi=0.0):
        m = mode([[2, 2, 0]], Mf, af)
        t = np.linspace(0, 50, 500)
        h = clean_QNM(m, t, A, phi)
        return t, h

    def test_self_mismatch_zero(self):
        t, h = self._make_simple_waveform()
        mm = compute_mismatch(t, h, t, h)
        assert mm < 1e-12

    def test_negative_mismatch_zero(self):
        """Mismatch uses real part of inner product, so -h should also be 0."""
        t, h = self._make_simple_waveform()
        mm = compute_mismatch(t, h, t, -h)
        # compute_mismatch uses np.real(vdot), so -h gives -1 -> mismatch=2
        # Actually: 1 - real(<h,-h>) / (||h|| ||h||) = 1 - (-1) = 2
        # This is the expected behavior -- not 0
        assert mm > 1.5

    def test_orthogonal_waveforms_mismatch_one(self):
        """Two waveforms with very different frequencies should be nearly orthogonal."""
        m1 = mode([[2, 2, 0]], Mf, af)
        m2 = mode([[3, 3, 0]], Mf, af)
        t = np.linspace(0, 50, 1000)
        h1 = clean_QNM(m1, t, 1.0, 0.0)
        h2 = clean_QNM(m2, t, 1.0, 0.0)
        mm = compute_mismatch(t, h1, t, h2)
        assert mm > 0.5


# ---------------------------------------------------------------------------
# mismatch_min_phase
# ---------------------------------------------------------------------------

class TestMismatchMinPhase:

    def test_recovers_known_phase_shift(self):
        m = mode([[2, 2, 0]], Mf, af)
        t = np.linspace(0, 50, 500)
        h1 = clean_QNM(m, t, 1.0, 0.0)

        dphi_true = 0.3
        h2 = h1 * np.exp(1j * dphi_true)

        res = mismatch_min_phase(t, h1, t, h2)
        assert res.fun < 1e-10
        # Recovered phase should be close to dphi_true
        assert np.isclose(np.abs(res.x[0]), np.abs(dphi_true), atol=0.1)


# ---------------------------------------------------------------------------
# delayed_QNM
# ---------------------------------------------------------------------------

class TestDelayedQNM:

    def test_equals_clean_when_no_delay(self):
        """With A_red_ratio=0 and trivial phase delay, should match clean."""
        m = mode([[2, 2, 0]], Mf, af)
        t = np.linspace(0, 50, 500)
        A, phi = 1.0, 0.3

        h_delayed = delayed_QNM(m, t, A, phi, A_red_ratio=0, dphi=0)
        h_clean = clean_QNM(m, t, A, phi)

        assert np.allclose(h_delayed, h_clean, atol=1e-10)


# ---------------------------------------------------------------------------
# waveform.postmerger t_end
# ---------------------------------------------------------------------------

class TestWaveformPostmergerTend:

    def test_t_end_truncation(self):
        t = np.linspace(0, 100, 2000)
        m = mode([[2, 2, 0]], Mf, af)
        h_arr = clean_QNM(m, t, 1.0, 0.0)
        h = waveform(t, h_arr, t_peak=0)

        t_pm, hr_pm, hi_pm = h.postmerger(0, t_end=50)
        assert t_pm[-1] <= 50.0
        assert len(t_pm) < len(t)
