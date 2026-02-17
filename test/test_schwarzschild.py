"""Tests for the Schwarzschild (real-waveform) fitting mode.

Schwarzschild black holes have zero spin, so the gravitational waveform is
purely real: h(t) = sum_j A_j * exp(omega_i_j * t) * cos(omega_r_j * t + phi_j).

These tests cover:
  1. Low-level model functions with part="real"
  2. QNMFit (VARPRO) roundtrips with Schwarzschild=True
  3. QNMFitVarMa roundtrips with Schwarzschild=True
  4. QNMFitVaryingStartingTime full-pipeline tests
  5. Edge cases (popt lengths, nan results, result shapes)
"""
import numpy as np
import jax.numpy as jnp
import pytest

from jaxqualin.waveforms import waveform, clean_QNM
from jaxqualin.qnmode import mode_list, mode_free, long_str_to_qnms_free
from jaxqualin.fit import (
    qnm_fit_func,
    qnm_fit_func_wrapper,
    qnm_fit_func_wrapper_complex,
    QNMFit,
    QNMFitVarMa,
    QNMFitVaryingStartingTime,
)
from jaxqualin.utils import interweave


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schwarzschild_modes(mode_strings):
    """Create mode objects at a=0 (Schwarzschild)."""
    Mf = 1.0
    af = 0.0
    return mode_list(mode_strings, Mf, af), Mf, af


def _make_real_waveform_from_modes(modes, A_list, phi_list, t_arr):
    """Build a synthetic purely-real waveform from Schwarzschild QNMs.

    h(t) = sum_j A_j * exp(omega_i_j * t) * cos(omega_r_j * t + phi_j)
    This is the real part of clean_QNM.
    """
    h_real = np.zeros(len(t_arr), dtype=np.float64)
    for m, A, phi in zip(modes, A_list, phi_list):
        h_complex = clean_QNM(m, t_arr, A, phi)
        h_real += np.real(h_complex)
    h_complex_arr = h_real + 0j * h_real
    return waveform(t_arr, h_complex_arr, t_peak=0)


# ---------------------------------------------------------------------------
# 1. Low-level model function tests
# ---------------------------------------------------------------------------

class TestSchwarzschildModelFunction:
    """Verify low-level fit functions produce the correct real waveform."""

    def test_part_real_matches_explicit_formula(self):
        """part='real' of qnm_fit_func matches A*exp(omega_i*t)*cos(omega_r*t+phi)."""
        modes, Mf, af = _make_schwarzschild_modes(['2.2.0'])
        A, phi = 2.5, 0.7
        t = jnp.linspace(0, 80, 500)

        h_fit_real = qnm_fit_func(
            t, modes, [[A, phi]], [], part="real")

        omegar = modes[0].omegar
        omegai = modes[0].omegai
        h_explicit = A * jnp.exp(omegai * t) * jnp.cos(omegar * t + phi)

        assert np.allclose(np.array(h_fit_real), np.array(h_explicit), atol=1e-12)

    def test_part_real_equals_real_of_complex(self):
        """part='real' should equal np.real(part=None) for fixed modes."""
        modes, Mf, af = _make_schwarzschild_modes(['2.2.0', '3.2.0'])
        params = [[1.5, 0.3], [0.8, 1.2]]
        t = jnp.linspace(0, 80, 500)

        h_complex = qnm_fit_func(t, modes, params, [], part=None)
        h_real = qnm_fit_func(t, modes, params, [], part="real")

        assert np.allclose(
            np.real(np.array(h_complex)), np.array(h_real), atol=1e-12)

    def test_part_real_equals_real_of_complex_free_modes(self):
        """part='real' should equal np.real(part=None) for free modes."""
        modes, Mf, af = _make_schwarzschild_modes(['2.2.0'])
        omegar = float(modes[0].omegar)
        omegai = float(modes[0].omegai)
        free_params = [[2.0, 0.5, omegar, omegai]]
        t = jnp.linspace(0, 80, 500)

        h_complex = qnm_fit_func(t, [], [], free_params, part=None)
        h_real = qnm_fit_func(t, [], [], free_params, part="real")

        assert np.allclose(
            np.real(np.array(h_complex)), np.array(h_real), atol=1e-12)

    def test_wrapper_complex_schwarzschild_zeros_imag(self):
        """qnm_fit_func_wrapper_complex with Schwarzschild=True zeros imaginary slots."""
        modes, Mf, af = _make_schwarzschild_modes(['2.2.0'])
        A, phi = 1.0, 0.0
        popt = jnp.array([A, phi])
        t_interleave = interweave(jnp.linspace(0, 50, 100), jnp.linspace(0, 50, 100))

        h_riffle = qnm_fit_func_wrapper_complex(
            t_interleave, modes, 0, popt, Schwarzschild=True)
        h_arr = np.array(h_riffle)

        # Odd indices are imaginary slots; they should all be zero
        assert np.allclose(h_arr[1::2], 0.0, atol=1e-15)

    def test_wrapper_complex_schwarzschild_real_part_nonzero(self):
        """qnm_fit_func_wrapper_complex with Schwarzschild=True has non-trivial real slots."""
        modes, Mf, af = _make_schwarzschild_modes(['2.2.0'])
        A, phi = 1.0, 0.0
        popt = jnp.array([A, phi])
        t_interleave = interweave(jnp.linspace(0, 50, 100), jnp.linspace(0, 50, 100))

        h_riffle = qnm_fit_func_wrapper_complex(
            t_interleave, modes, 0, popt, Schwarzschild=True)
        h_arr = np.array(h_riffle)

        # Even indices are real slots; they should be non-trivial
        assert np.max(np.abs(h_arr[0::2])) > 0.1


# ---------------------------------------------------------------------------
# 2. QNMFit (VARPRO) roundtrip tests with Schwarzschild=True
# ---------------------------------------------------------------------------

class TestSchwarzschildQNMFitRoundtrip:
    """Roundtrip tests: create a purely real waveform, fit it, recover parameters."""

    def test_single_fixed_mode_amplitude(self):
        modes, Mf, af = _make_schwarzschild_modes(['2.2.0'])
        A, phi = 2.5, 0.3
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes, [A], [phi], t)

        fitter = QNMFit(h, t0=0.0, N_free=0, qnm_fixed_list=modes,
                        Schwarzschild=True)
        fitter.do_fit()
        popt = np.array(fitter.popt)
        assert np.isclose(popt[0], A, atol=1e-5), f"Expected A={A}, got {popt[0]}"

    def test_single_fixed_mode_phase(self):
        modes, Mf, af = _make_schwarzschild_modes(['2.2.0'])
        A, phi = 2.5, 0.3
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes, [A], [phi], t)

        fitter = QNMFit(h, t0=0.0, N_free=0, qnm_fixed_list=modes,
                        Schwarzschild=True)
        fitter.do_fit()
        popt = np.array(fitter.popt)
        assert np.isclose(popt[1], phi, atol=1e-5), f"Expected phi={phi}, got {popt[1]}"

    def test_single_fixed_mode_mismatch(self):
        modes, Mf, af = _make_schwarzschild_modes(['2.2.0'])
        A, phi = 2.5, 0.3
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes, [A], [phi], t)

        fitter = QNMFit(h, t0=0.0, N_free=0, qnm_fixed_list=modes,
                        Schwarzschild=True)
        fitter.do_fit()
        assert fitter.mismatch < 1e-8, f"Mismatch too large: {fitter.mismatch}"

    def test_two_fixed_modes_amplitudes(self):
        modes, Mf, af = _make_schwarzschild_modes(['2.2.0', '3.2.0'])
        A_list = [1.0, 0.5]
        phi_list = [0.0, np.pi / 3]
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes, A_list, phi_list, t)

        fitter = QNMFit(h, t0=0.0, N_free=0, qnm_fixed_list=modes,
                        Schwarzschild=True)
        fitter.do_fit()
        popt = np.array(fitter.popt)
        assert np.isclose(popt[0], A_list[0], atol=1e-5), \
            f"A0: expected {A_list[0]}, got {popt[0]}"
        assert np.isclose(popt[2], A_list[1], atol=1e-5), \
            f"A1: expected {A_list[1]}, got {popt[2]}"

    def test_two_fixed_modes_phases(self):
        modes, Mf, af = _make_schwarzschild_modes(['2.2.0', '3.2.0'])
        A_list = [1.0, 0.5]
        phi_list = [0.0, np.pi / 3]
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes, A_list, phi_list, t)

        fitter = QNMFit(h, t0=0.0, N_free=0, qnm_fixed_list=modes,
                        Schwarzschild=True)
        fitter.do_fit()
        popt = np.array(fitter.popt)
        assert np.isclose(popt[1], phi_list[0], atol=1e-5), \
            f"phi0: expected {phi_list[0]}, got {popt[1]}"
        assert np.isclose(popt[3], phi_list[1], atol=1e-5), \
            f"phi1: expected {phi_list[1]}, got {popt[3]}"

    def test_two_fixed_modes_mismatch(self):
        modes, Mf, af = _make_schwarzschild_modes(['2.2.0', '3.2.0'])
        A_list = [1.0, 0.5]
        phi_list = [0.0, np.pi / 3]
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes, A_list, phi_list, t)

        fitter = QNMFit(h, t0=0.0, N_free=0, qnm_fixed_list=modes,
                        Schwarzschild=True)
        fitter.do_fit()
        assert fitter.mismatch < 1e-8, f"Mismatch too large: {fitter.mismatch}"

    def test_free_mode_recovers_frequency(self):
        modes, Mf, af = _make_schwarzschild_modes(['2.2.0'])
        target_omegar = float(modes[0].omegar)
        target_omegai = float(modes[0].omegai)
        A, phi = 1.5, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes, [A], [phi], t)

        fitter = QNMFit(
            h, t0=0.0, N_free=1, qnm_fixed_list=[],
            Schwarzschild=True,
            guess_free=[1, 1, target_omegar * 0.9, target_omegai * 0.9])
        fitter.do_fit()
        popt = np.array(fitter.popt)
        recovered_omegar = popt[2]
        recovered_omegai = popt[3]
        assert np.isclose(recovered_omegar, target_omegar, rtol=1e-3), \
            f"omega_r: expected {target_omegar}, got {recovered_omegar}"
        assert np.isclose(recovered_omegai, target_omegai, rtol=1e-3), \
            f"omega_i: expected {target_omegai}, got {recovered_omegai}"

    def test_reconstruction_is_real(self):
        """The reconstructed waveform should be real-valued for Schwarzschild."""
        modes, Mf, af = _make_schwarzschild_modes(['2.2.0'])
        A, phi = 2.0, 0.4
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes, [A], [phi], t)

        fitter = QNMFit(h, t0=0.0, N_free=0, qnm_fixed_list=modes,
                        Schwarzschild=True)
        fitter.do_fit()
        recon = np.array(fitter.reconstruct_h)
        assert np.allclose(np.imag(recon), 0.0, atol=1e-15), \
            "Schwarzschild reconstruction should be purely real"


# ---------------------------------------------------------------------------
# 3. QNMFitVarMa roundtrip tests with Schwarzschild=True
# ---------------------------------------------------------------------------

class TestSchwarzschildQNMFitVarMaRoundtrip:
    """Roundtrip tests for the variable-M,a fitter in Schwarzschild mode."""

    def test_single_fixed_mode_recovers_A_phi_M(self):
        """Fit a single fixed-freq mode, recover A, phi, and M."""
        Mf_true = 1.0
        af_true = 0.0
        modes = mode_list(['2.2.0'], Mf_true, af_true)
        A, phi = 2.0, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes, [A], [phi], t)

        qnm_free = long_str_to_qnms_free('2.2.0')
        fitter = QNMFitVarMa(
            h, t0=0.0, qnm_free_list=qnm_free,
            Schwarzschild=True,
            guess_free=[A * 0.8, phi * 0.8],
            guess_M_a=[Mf_true * 0.9])
        fitter.do_fit()
        popt = np.array(fitter.popt)

        recovered_A = popt[0]
        recovered_phi = popt[1]
        recovered_M = popt[2]

        assert np.isclose(recovered_A, A, rtol=0.05), \
            f"A: expected {A}, got {recovered_A}"
        assert np.isclose(recovered_M, Mf_true, rtol=0.05), \
            f"M: expected {Mf_true}, got {recovered_M}"

    def test_popt_length_schwarzschild(self):
        """popt should have length 2*N_fix + 2*N_free + 1 (no 'a' parameter)."""
        Mf_true = 1.0
        af_true = 0.0
        modes_fixed = mode_list(['2.2.0'], Mf_true, af_true)
        A, phi = 1.0, 0.0
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes_fixed, [A], [phi], t)

        qnm_free = long_str_to_qnms_free('3.2.0')
        fitter = QNMFitVarMa(
            h, t0=0.0, qnm_free_list=qnm_free,
            qnm_fixed_list=modes_fixed,
            Schwarzschild=True,
            guess_M_a=[Mf_true * 0.9])
        fitter.do_fit()
        popt = np.array(fitter.popt)

        N_fix = len(modes_fixed)
        N_free = len(qnm_free)
        expected_len = 2 * N_fix + 2 * N_free + 1
        assert len(popt) == expected_len, \
            f"popt length: expected {expected_len}, got {len(popt)}"

    def test_mismatch_near_zero(self):
        Mf_true = 1.0
        af_true = 0.0
        modes = mode_list(['2.2.0'], Mf_true, af_true)
        A, phi = 2.0, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes, [A], [phi], t)

        qnm_free = long_str_to_qnms_free('2.2.0')
        fitter = QNMFitVarMa(
            h, t0=0.0, qnm_free_list=qnm_free,
            Schwarzschild=True,
            guess_free=[A * 0.8, phi * 0.8],
            guess_M_a=[Mf_true * 0.9])
        fitter.do_fit()
        assert fitter.mismatch < 1e-4, f"Mismatch too large: {fitter.mismatch}"


# ---------------------------------------------------------------------------
# 4. QNMFitVaryingStartingTime full pipeline tests
# ---------------------------------------------------------------------------

class TestSchwarzschildVaryingStartingTime:
    """Full pipeline tests for QNMFitVaryingStartingTime with Schwarzschild."""

    def test_var_Ma_false_amplitude_constant_across_t0(self):
        """VARPRO path: amplitude should be constant across starting times."""
        modes, Mf, af = _make_schwarzschild_modes(['2.2.0'])
        A0, phi0 = 2.0, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes, [A0], [phi0], t)

        t0_arr = np.linspace(0, 10, 6)
        fitter = QNMFitVaryingStartingTime(
            h, t0_arr, N_free=0,
            qnm_fixed_list=modes, load_pickle=False,
            Schwarzschild=True,
            run_string_prefix='schw_vst_test',
            save_results=False)
        fitter.do_fits()
        result = fitter.result_full
        A_arr = np.array(result.A_dict['A_2.2.0'])

        assert np.allclose(A_arr, A0, rtol=1e-4), \
            f"Amplitudes not constant: {A_arr}"

    def test_var_Ma_false_phase_constant_across_t0(self):
        """VARPRO path: phase should be constant across starting times."""
        modes, Mf, af = _make_schwarzschild_modes(['2.2.0'])
        A0, phi0 = 2.0, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes, [A0], [phi0], t)

        t0_arr = np.linspace(0, 10, 6)
        fitter = QNMFitVaryingStartingTime(
            h, t0_arr, N_free=0,
            qnm_fixed_list=modes, load_pickle=False,
            Schwarzschild=True,
            run_string_prefix='schw_vst_phase_test',
            save_results=False)
        fitter.do_fits()
        result = fitter.result_full
        phi_arr = np.array(result.phi_dict['phi_2.2.0'])
        diff = np.angle(np.exp(1j * (phi_arr - phi0)))
        assert np.allclose(diff, 0, atol=1e-4), \
            f"Phases not constant: {phi_arr}"

    def test_var_Ma_true_recovers_M(self):
        """VarMa path: should recover M correctly, no 'a' in Ma_dict."""
        Mf_true = 1.0
        af_true = 0.0
        modes_fixed = mode_list(['2.2.0'], Mf_true, af_true)
        qnm_free = long_str_to_qnms_free('2.2.0')

        A, phi = 2.0, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes_fixed, [A], [phi], t)

        t0_arr = np.array([0.0, 5.0])
        fitter = QNMFitVaryingStartingTime(
            h, t0_arr, N_free=0,
            qnm_fixed_list=[],
            qnm_free_list=qnm_free,
            var_M_a=True,
            Schwarzschild=True,
            load_pickle=False,
            run_string_prefix='schw_varMa_test',
            save_results=False)
        fitter.do_fits()
        result = fitter.result_full

        assert "M" in result.Ma_dict, "Ma_dict should contain 'M'"
        assert "a" not in result.Ma_dict, "Ma_dict should NOT contain 'a' for Schwarzschild"

        M_arr = np.array(result.Ma_dict["M"])
        assert np.allclose(M_arr, Mf_true, rtol=0.05), \
            f"M not recovered: expected {Mf_true}, got {M_arr}"

    def test_var_Ma_true_mismatch_small(self):
        """VarMa path: mismatch should be small for Schwarzschild fit."""
        Mf_true = 1.0
        af_true = 0.0
        modes_fixed = mode_list(['2.2.0'], Mf_true, af_true)
        qnm_free = long_str_to_qnms_free('2.2.0')

        A, phi = 2.0, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes_fixed, [A], [phi], t)

        t0_arr = np.array([0.0])
        fitter = QNMFitVaryingStartingTime(
            h, t0_arr, N_free=0,
            qnm_fixed_list=[],
            qnm_free_list=qnm_free,
            var_M_a=True,
            Schwarzschild=True,
            load_pickle=False,
            run_string_prefix='schw_varMa_mm_test',
            save_results=False)
        fitter.do_fits()
        result = fitter.result_full
        mismatch = result.mismatch_arr[0]
        assert mismatch < 1e-3, f"Mismatch too large: {mismatch}"


# ---------------------------------------------------------------------------
# 5. Edge case tests
# ---------------------------------------------------------------------------

class TestSchwarzschildEdgeCases:
    """Edge case tests for Schwarzschild mode."""

    def test_varma_result_popt_shape_schwarzschild(self):
        """QNMFitVaryingStartingTimeResultVarMa popt shape for Schwarzschild."""
        Mf_true = 1.0
        af_true = 0.0
        modes_fixed = mode_list(['2.2.0'], Mf_true, af_true)
        qnm_free = long_str_to_qnms_free('2.2.0')

        A, phi = 1.0, 0.0
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes_fixed, [A], [phi], t)

        t0_arr = np.array([0.0, 5.0, 10.0])
        fitter = QNMFitVaryingStartingTime(
            h, t0_arr, N_free=0,
            qnm_fixed_list=[],
            qnm_free_list=qnm_free,
            var_M_a=True,
            Schwarzschild=True,
            load_pickle=False,
            run_string_prefix='schw_shape_test',
            save_results=False)
        fitter.do_fits()
        result = fitter.result_full

        N_fix = 0
        N_free = len(qnm_free)
        expected_popt_rows = 2 * N_fix + 2 * N_free + 1
        assert result.popt_full.shape == (expected_popt_rows, len(t0_arr)), \
            f"popt_full shape: expected ({expected_popt_rows}, {len(t0_arr)}), " \
            f"got {result.popt_full.shape}"

    def test_make_nan_result_popt_length_schwarzschild(self):
        """make_nan_result should produce correct-length popt for Schwarzschild."""
        Mf_true = 1.0
        af_true = 0.0
        qnm_free = long_str_to_qnms_free('2.2.0')

        A, phi = 1.0, 0.0
        modes_fixed = mode_list(['2.2.0'], Mf_true, af_true)
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes_fixed, [A], [phi], t)

        t0_arr = np.array([0.0])
        fitter = QNMFitVaryingStartingTime(
            h, t0_arr, N_free=0,
            qnm_fixed_list=[],
            qnm_free_list=qnm_free,
            var_M_a=True,
            Schwarzschild=True,
            load_pickle=False,
            run_string_prefix='schw_nan_test',
            save_results=False)

        nan_result = fitter.make_nan_result()
        N_fix = 0
        N_free = len(qnm_free)
        expected_len = 2 * N_fix + 2 * N_free + 1
        assert len(nan_result.popt) == expected_len, \
            f"nan popt length: expected {expected_len}, got {len(nan_result.popt)}"
        assert np.all(np.isnan(nan_result.popt)), \
            "All popt values should be NaN"

    def test_varpro_result_shape_schwarzschild(self):
        """QNMFitVaryingStartingTimeResult popt shape for Schwarzschild (non-varMa)."""
        modes, Mf, af = _make_schwarzschild_modes(['2.2.0'])
        A, phi = 1.0, 0.0
        t = np.linspace(0, 100, 2000)
        h = _make_real_waveform_from_modes(modes, [A], [phi], t)

        t0_arr = np.array([0.0, 5.0, 10.0])
        fitter = QNMFitVaryingStartingTime(
            h, t0_arr, N_free=0,
            qnm_fixed_list=modes,
            Schwarzschild=True,
            load_pickle=False,
            run_string_prefix='schw_varpro_shape_test',
            save_results=False)
        fitter.do_fits()
        result = fitter.result_full

        N_fix = len(modes)
        N_free = 0
        expected_popt_rows = 2 * N_fix + 4 * N_free
        assert result.popt_full.shape == (expected_popt_rows, len(t0_arr)), \
            f"popt_full shape: expected ({expected_popt_rows}, {len(t0_arr)}), " \
            f"got {result.popt_full.shape}"
