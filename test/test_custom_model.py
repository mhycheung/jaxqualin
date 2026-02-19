"""Integration tests for QNMFitModel with custom QNMModel and backward compat."""
import numpy as np
import pytest

from jaxqualin.qnmode import (
    QNMModel, KerrModel, model_mode_free, model_mode,
    mode_free, mode, custom_mode, custom_mode_list,
    mode_list, long_str_to_qnms_free,
)
from jaxqualin.waveforms import waveform, clean_QNM
from jaxqualin.fit import QNMFit, QNMFitModel, QNMFitVarMa, QNMFitVaryingStartingTime


def _make_clean_waveform(modes, A_list, phi_list, t_arr):
    h_arr = np.zeros(len(t_arr), dtype=np.complex128)
    for m, A, phi in zip(modes, A_list, phi_list):
        h_arr += clean_QNM(m, t_arr, A, phi)
    return waveform(t_arr, h_arr, t_peak=0)


# ---------------------------------------------------------------------------
# Backward compatibility — existing Kerr API unchanged
# ---------------------------------------------------------------------------

class TestBackwardCompat:

    def test_standard_kerr_call(self):
        """QNMFitVarMa with model=None should work exactly as before."""
        Mf, af = 1.0, 0.7
        modes = mode_list(['2.2.0'], Mf, af)
        A, phi = 2.0, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_clean_waveform(modes, [A], [phi], t)

        qnm_free = long_str_to_qnms_free('2.2.0')
        fitter = QNMFitVarMa(
            h, t0=0.0, qnm_free_list=qnm_free,
            real=False,
            guess_free=[A * 0.8, phi * 0.8],
            guess_M_a=[Mf * 0.9, af * 0.9])
        fitter.do_fit()
        assert fitter.mismatch < 1e-3

    def test_schwarzschild_call(self):
        """Schwarzschild with model=None should work."""
        Mf = 1.0
        modes = mode_list(['2.2.0'], Mf, 0.0)
        A, phi = 2.0, 0.3
        t = np.linspace(0, 100, 2000)
        h = _make_clean_waveform(modes, [A], [phi], t)

        qnm_free = long_str_to_qnms_free('2.2.0')
        fitter = QNMFitVarMa(
            h, t0=0.0, qnm_free_list=qnm_free,
            real=True,
            guess_free=[A * 0.8, phi * 0.8],
            guess_M_a=[Mf * 0.9])
        fitter.do_fit()
        popt = np.array(fitter.popt)
        expected_len = 2 * 1 + 1  # 2 per free mode + M
        assert len(popt) == expected_len

    def test_popt_length_kerr_unchanged(self):
        """popt = 2*N_fix + 2*N_free + 2 with model=None."""
        Mf, af = 1.0, 0.7
        modes_fixed = mode_list(['2.2.0'], Mf, af)
        modes_all = mode_list(['2.2.0', '2.2.1'], Mf, af)
        t = np.linspace(0, 100, 2000)
        h = _make_clean_waveform(modes_all, [1.0, 0.5], [0.0, 0.1], t)

        qnm_free = long_str_to_qnms_free('2.2.1')
        fitter = QNMFitVarMa(
            h, t0=0.0, qnm_free_list=qnm_free,
            qnm_fixed_list=modes_fixed,
            real=False,
            guess_M_a=[Mf * 0.9, af * 0.9])
        fitter.do_fit()
        expected = 2 * 1 + 2 * 1 + 2
        assert len(fitter.popt) == expected


# ---------------------------------------------------------------------------
# QNMFitVarMa with explicit KerrModel
# ---------------------------------------------------------------------------

class TestExplicitKerrModel:

    def test_explicit_kerr_matches_default(self):
        """QNMFitModel(model=KerrModel()) should give same result as default."""
        Mf, af = 1.0, 0.7
        modes = mode_list(['2.2.0'], Mf, af)
        A, phi = 2.0, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_clean_waveform(modes, [A], [phi], t)

        km = KerrModel()
        qnm_free_model = [model_mode_free([[2, 2, 0]], model=km)]

        fitter = QNMFitModel(
            h, t0=0.0, qnm_free_list=qnm_free_model,
            model=km,
            model_params_guess={"M": Mf * 0.9, "a": af * 0.9})
        fitter.do_fit()

        popt = np.array(fitter.popt)
        recovered_M = popt[2]
        recovered_a = popt[3]

        assert np.isclose(recovered_M, Mf, rtol=0.05)
        assert np.isclose(recovered_a, af, rtol=0.05)
        assert fitter.mismatch < 1e-3

    def test_popt_length_with_kerr_model(self):
        """popt = 2*N_fix + 2*N_free + n_params."""
        Mf, af = 1.0, 0.7
        modes = mode_list(['2.2.0'], Mf, af)
        A, phi = 1.0, 0.0
        t = np.linspace(0, 100, 2000)
        h = _make_clean_waveform(modes, [A], [phi], t)

        km = KerrModel()
        qnm_free_model = [model_mode_free([[2, 2, 0]], model=km)]

        fitter = QNMFitModel(
            h, t0=0.0, qnm_free_list=qnm_free_model,
            model=km,
            model_params_guess={"M": Mf * 0.9, "a": af * 0.9})
        fitter.do_fit()
        expected = 2 * 0 + 2 * 1 + 2  # N_fix=0, N_free=1, n_params=2
        assert len(fitter.popt) == expected


# ---------------------------------------------------------------------------
# Custom (non-Kerr) QNMModel
# ---------------------------------------------------------------------------

class SimpleShiftModel(QNMModel):
    """omega = alpha + beta*i (independent of lmnx for simplicity)."""
    param_names = ["alpha", "beta"]

    def compute_omega(self, lmnx, alpha, beta, **kwargs):
        return alpha + 1j * beta

    def param_bounds(self):
        return {"alpha": (0, 10), "beta": (-5, 0)}


class TestCustomModel:

    def test_single_free_mode_recovery(self):
        """Recover alpha and beta from a synthetic waveform."""
        alpha_true, beta_true = 0.5, -0.08
        omega_true = alpha_true + 1j * beta_true
        A_true, phi_true = 2.0, 0.3
        t = np.linspace(0, 100, 2000)
        h_arr = A_true * np.exp(-1j * (omega_true * t + phi_true))
        h = waveform(t, h_arr, t_peak=0)

        sm = SimpleShiftModel()
        modes = [model_mode_free([[2, 2, 0]], model=sm)]

        fitter = QNMFitModel(
            h, t0=0.0, qnm_free_list=modes,
            model=sm,
            model_params_guess={"alpha": 0.4, "beta": -0.1})
        fitter.do_fit()

        popt = np.array(fitter.popt)
        recovered_alpha = popt[2]
        recovered_beta = popt[3]

        assert np.isclose(recovered_alpha, alpha_true, atol=0.01)
        assert np.isclose(recovered_beta, beta_true, atol=0.01)
        assert fitter.mismatch < 1e-3

    def test_popt_length_custom_model(self):
        """popt should have 2*N_free + n_model_params elements."""
        alpha, beta = 0.5, -0.08
        omega = alpha + 1j * beta
        t = np.linspace(0, 100, 2000)
        h_arr = 1.0 * np.exp(-1j * omega * t)
        h = waveform(t, h_arr, t_peak=0)

        sm = SimpleShiftModel()
        modes = [model_mode_free([[2, 2, 0]], model=sm)]

        fitter = QNMFitModel(
            h, t0=0.0, qnm_free_list=modes,
            model=sm,
            model_params_guess={"alpha": 0.4, "beta": -0.1})
        fitter.do_fit()
        expected = 2 * 1 + 2  # N_free=1, n_params=2
        assert len(fitter.popt) == expected

    def test_bounds_respected(self):
        """Model param_bounds should restrict parameter ranges."""
        alpha_true, beta_true = 0.5, -0.08
        omega = alpha_true + 1j * beta_true
        t = np.linspace(0, 100, 2000)
        h_arr = 1.0 * np.exp(-1j * omega * t)
        h = waveform(t, h_arr, t_peak=0)

        sm = SimpleShiftModel()
        modes = [model_mode_free([[2, 2, 0]], model=sm)]

        fitter = QNMFitModel(
            h, t0=0.0, qnm_free_list=modes,
            model=sm,
            model_params_guess={"alpha": 0.4, "beta": -0.1})
        fitter.do_fit()
        popt = np.array(fitter.popt)
        alpha_fit = popt[2]
        beta_fit = popt[3]
        assert 0 <= alpha_fit <= 10
        assert -5 <= beta_fit <= 0


# ---------------------------------------------------------------------------
# Augmented Kerr model
# ---------------------------------------------------------------------------

class KerrPlusDelta(KerrModel):
    param_names = ["M", "a", "delta"]

    def compute_omega(self, lmnx, M, a, delta, **kwargs):
        omega_kerr = super().compute_omega(lmnx, M, a)
        return omega_kerr + delta

    def param_bounds(self):
        b = super().param_bounds()
        b["delta"] = (-1.0, 1.0)
        return b


class TestAugmentedKerrModel:

    def test_recovers_delta(self):
        Mf, af = 1.0, 0.7
        delta_true = 0.02
        km = KerrModel()
        omega_base = km.compute_omega([[2, 2, 0]], M=Mf, a=af)
        omega_shifted = omega_base + delta_true

        A_true, phi_true = 2.0, 0.3
        t = np.linspace(0, 100, 2000)
        h_arr = A_true * np.exp(-1j * (omega_shifted * t + phi_true))
        h = waveform(t, h_arr, t_peak=0)

        model = KerrPlusDelta()
        modes = [model_mode_free([[2, 2, 0]], model=model)]

        fitter = QNMFitModel(
            h, t0=0.0, qnm_free_list=modes,
            model=model,
            model_params_guess={"M": Mf * 0.95, "a": af * 0.95, "delta": 0.0})
        fitter.do_fit()

        popt = np.array(fitter.popt)
        recovered_delta = popt[4]
        assert np.isclose(recovered_delta, delta_true, atol=0.02)
        assert fitter.mismatch < 1e-3

    def test_popt_length_augmented(self):
        """popt = 2*N_free + 3 (M, a, delta)."""
        Mf, af = 1.0, 0.7
        km = KerrModel()
        omega = km.compute_omega([[2, 2, 0]], M=Mf, a=af)

        t = np.linspace(0, 100, 2000)
        h_arr = 1.0 * np.exp(-1j * omega * t)
        h = waveform(t, h_arr, t_peak=0)

        model = KerrPlusDelta()
        modes = [model_mode_free([[2, 2, 0]], model=model)]

        fitter = QNMFitModel(
            h, t0=0.0, qnm_free_list=modes,
            model=model,
            model_params_guess={"M": Mf * 0.9, "a": af * 0.9, "delta": 0.0})
        fitter.do_fit()
        expected = 2 * 1 + 3  # N_free=1, 3 model params
        assert len(fitter.popt) == expected


# ---------------------------------------------------------------------------
# QNMFitVaryingStartingTime with custom model
# ---------------------------------------------------------------------------

class TestVaryingStartingTimeCustomModel:

    def test_varying_time_with_kerr_model(self):
        """Full pipeline with explicit KerrModel through VaryingStartingTime."""
        Mf, af = 1.0, 0.7
        modes = mode_list(['2.2.0'], Mf, af)
        A, phi = 2.0, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_clean_waveform(modes, [A], [phi], t)

        km = KerrModel()
        qnm_free = [model_mode_free([[2, 2, 0]], model=km)]

        t0_arr = np.array([0.0, 5.0])
        fitter = QNMFitVaryingStartingTime(
            h, t0_arr, N_free=0,
            qnm_fixed_list=[],
            qnm_free_list=qnm_free,
            var_M_a=True,
            load_pickle=False,
            run_string_prefix='custom_kerr_vst',
            save_results=False,
            model=km,
            model_params_guess={"M": Mf * 0.9, "a": af * 0.9})
        fitter.do_fits()
        result = fitter.result_full

        assert "M" in result.model_params_dict
        assert "a" in result.model_params_dict
        # backward-compat alias still works
        assert result.Ma_dict is result.model_params_dict
        M_arr = np.array(result.model_params_dict["M"])
        a_arr = np.array(result.model_params_dict["a"])
        assert np.allclose(M_arr, Mf, rtol=0.05)
        assert np.allclose(a_arr, af, rtol=0.05)

    def test_varying_time_with_custom_model(self):
        """Full pipeline with SimpleShiftModel through VaryingStartingTime."""
        alpha_true, beta_true = 0.5, -0.08
        omega_true = alpha_true + 1j * beta_true
        A_true, phi_true = 2.0, 0.3
        t = np.linspace(0, 100, 2000)
        h_arr = A_true * np.exp(-1j * (omega_true * t + phi_true))
        h = waveform(t, h_arr, t_peak=0)

        sm = SimpleShiftModel()
        modes = [model_mode_free([[2, 2, 0]], model=sm)]

        t0_arr = np.array([0.0, 5.0])
        fitter = QNMFitVaryingStartingTime(
            h, t0_arr, N_free=0,
            qnm_fixed_list=[],
            qnm_free_list=modes,
            var_M_a=True,
            load_pickle=False,
            run_string_prefix='custom_shift_vst',
            save_results=False,
            model=sm,
            model_params_guess={"alpha": 0.4, "beta": -0.1})
        fitter.do_fits()
        result = fitter.result_full

        assert "alpha" in result.model_params_dict
        assert "beta" in result.model_params_dict
        alpha_arr = np.array(result.model_params_dict["alpha"])
        beta_arr = np.array(result.model_params_dict["beta"])
        assert np.allclose(alpha_arr, alpha_true, atol=0.05)
        assert np.allclose(beta_arr, beta_true, atol=0.05)


# ---------------------------------------------------------------------------
# Mixed fixed + free with model
# ---------------------------------------------------------------------------

class TestMixedFixedFreeModel:

    def test_fixed_custom_plus_free_model(self):
        """Fixed custom_mode + free model_mode_free modes together."""
        Mf, af = 1.0, 0.7
        mode_fixed = mode([[2, 2, 0]], Mf, af)
        km = KerrModel()
        omega_221 = km.compute_omega([[2, 2, 1]], M=Mf, a=af)

        A_list = [1.0, 0.5]
        phi_list = [0.0, 0.1]
        t = np.linspace(0, 100, 2000)
        h_arr = (
            A_list[0] * np.exp(-1j * ((mode_fixed.omegar + 1j * mode_fixed.omegai) * t + phi_list[0]))
            + A_list[1] * np.exp(-1j * (omega_221 * t + phi_list[1]))
        )
        h = waveform(t, h_arr, t_peak=0)

        qnm_free = [model_mode_free([[2, 2, 1]], model=km)]
        fitter = QNMFitModel(
            h, t0=0.0,
            qnm_free_list=qnm_free,
            qnm_fixed_list=[mode_fixed],
            model=km,
            model_params_guess={"M": Mf * 0.9, "a": af * 0.9})
        fitter.do_fit()
        assert fitter.mismatch < 1e-2
