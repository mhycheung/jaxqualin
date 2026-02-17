"""Unit tests for low-level fit model functions and basis construction."""
import numpy as np
import jax.numpy as jnp
import pytest

from jaxqualin.waveforms import clean_QNM
from jaxqualin.qnmode import mode_list, make_mirror_ratio_list
from jaxqualin.fit import (
    qnm_fit_func,
    qnm_fit_func_mirror_fixed,
    model_func_optimized,
    _compute_linear_params_and_popt,
)


@pytest.fixture
def modes_220_221():
    Mf, af = 1.0, 0.7
    return mode_list(['2.2.0', '2.2.1'], Mf, af), Mf, af


# ---------------------------------------------------------------------------
# qnm_fit_func vs clean_QNM
# ---------------------------------------------------------------------------

class TestQnmFitFuncVsCleanQNM:
    """The two functions must produce identical waveforms."""

    def test_single_mode_complex(self):
        Mf, af = 1.0, 0.7
        modes = mode_list(['2.2.0'], Mf, af)
        A, phi = 2.0, 0.3
        t = np.linspace(0, 50, 500)

        h_clean = clean_QNM(modes[0], t, A, phi)
        h_fit = qnm_fit_func(
            jnp.array(t), modes, [[A, phi]], [], part=None)

        assert np.allclose(np.array(h_clean), np.array(h_fit), atol=1e-12)

    def test_single_mode_real_part(self):
        Mf, af = 1.0, 0.7
        modes = mode_list(['2.2.0'], Mf, af)
        A, phi = 2.0, 0.3
        t = np.linspace(0, 50, 500)

        h_clean = clean_QNM(modes[0], t, A, phi)
        h_fit_real = qnm_fit_func(
            jnp.array(t), modes, [[A, phi]], [], part="real")

        assert np.allclose(
            np.real(np.array(h_clean)), np.array(h_fit_real), atol=1e-12)

    def test_single_mode_imag_part(self):
        Mf, af = 1.0, 0.7
        modes = mode_list(['2.2.0'], Mf, af)
        A, phi = 2.0, 0.3
        t = np.linspace(0, 50, 500)

        h_clean = clean_QNM(modes[0], t, A, phi)
        h_fit_imag = qnm_fit_func(
            jnp.array(t), modes, [[A, phi]], [], part="imag")

        assert np.allclose(
            np.imag(np.array(h_clean)), np.array(h_fit_imag), atol=1e-12)


# ---------------------------------------------------------------------------
# qnm_fit_func_mirror_fixed
# ---------------------------------------------------------------------------

class TestQnmFitFuncMirrorFixed:

    def test_mirror_includes_both_terms(self):
        Mf, af = 1.0, 0.7
        modes = mode_list(['2.2.0'], Mf, af)
        iota, psi = np.pi / 3, np.pi / 2
        mirror_ratio_list = make_mirror_ratio_list(modes, iota, psi)

        A, phi = 1.0, 0.0
        t = jnp.linspace(0, 50, 500)

        h_mirror = qnm_fit_func_mirror_fixed(
            t, modes, [[A, phi]], mirror_ratio_list, part=None)
        h_prograde_only = qnm_fit_func(
            t, modes, [[A, phi]], [], part=None)

        # Mirror waveform should differ from prograde-only
        assert not np.allclose(
            np.array(h_mirror), np.array(h_prograde_only), atol=1e-6)

        # But they should agree at t=0 only approximately (mirror contrib)
        diff = np.array(h_mirror) - np.array(h_prograde_only)
        assert np.max(np.abs(diff)) > 1e-3


# ---------------------------------------------------------------------------
# model_func_optimized basis shape
# ---------------------------------------------------------------------------

class TestModelFuncOptimizedShape:

    def test_no_mirror_shape(self, modes_220_221):
        modes, Mf, af = modes_220_221
        t = jnp.linspace(0, 50, 100)
        omegar = jnp.array([m.omegar for m in modes])
        omegai = jnp.array([m.omegai for m in modes])

        # 2 fixed, 1 free, no mirror -> 3 columns
        nonlinear = jnp.array([0.5, -0.1])
        basis = model_func_optimized(
            nonlinear, t, omegar, omegai, jnp.array([]),
            N_free=1, N_fix=2, include_mirror=False)
        assert basis.shape == (100, 3)

    def test_mirror_shape(self, modes_220_221):
        modes, Mf, af = modes_220_221
        t = jnp.linspace(0, 50, 100)
        omegar = jnp.array([m.omegar for m in modes])
        omegai = jnp.array([m.omegai for m in modes])
        iota, psi = np.pi / 3, np.pi / 2
        mirror_ratio_list = make_mirror_ratio_list(modes, iota, psi)
        mirror_arr = jnp.array(mirror_ratio_list)

        # 2 fixed, 0 free, with mirror -> 4 columns (2 prograde + 2 mirror)
        basis = model_func_optimized(
            jnp.array([]), t, omegar, omegai, mirror_arr,
            N_free=0, N_fix=2, include_mirror=True)
        assert basis.shape == (100, 4)


# ---------------------------------------------------------------------------
# model_func_optimized basis values
# ---------------------------------------------------------------------------

class TestModelFuncOptimizedValues:

    def test_fixed_mode_basis_is_exp(self):
        Mf, af = 1.0, 0.7
        modes = mode_list(['2.2.0'], Mf, af)
        t = jnp.linspace(0, 50, 100)
        omegar = jnp.array([modes[0].omegar])
        omegai = jnp.array([modes[0].omegai])

        basis = model_func_optimized(
            jnp.array([]), t, omegar, omegai, jnp.array([]),
            N_free=0, N_fix=1, include_mirror=False)

        omega = modes[0].omegar + 1j * modes[0].omegai
        expected = jnp.exp(-1j * omega * t)
        assert np.allclose(np.array(basis[:, 0]), np.array(expected), atol=1e-12)

    def test_mirror_basis_columns_interleaved(self):
        Mf, af = 1.0, 0.7
        modes = mode_list(['2.2.0'], Mf, af)
        iota, psi = np.pi / 3, np.pi / 2
        mirror_ratio_list = make_mirror_ratio_list(modes, iota, psi)
        mirror_arr = jnp.array(mirror_ratio_list)

        t = jnp.linspace(0, 50, 100)
        omegar = jnp.array([modes[0].omegar])
        omegai = jnp.array([modes[0].omegai])

        basis = model_func_optimized(
            jnp.array([]), t, omegar, omegai, mirror_arr,
            N_free=0, N_fix=1, include_mirror=True)

        # Column 0: prograde
        omega = modes[0].omegar + 1j * modes[0].omegai
        expected_pro = jnp.exp(-1j * omega * t)
        assert np.allclose(
            np.array(basis[:, 0]), np.array(expected_pro), atol=1e-12)

        # Column 1: mirror
        ratio_amp = mirror_ratio_list[0][0]
        ratio_phase = mirror_ratio_list[0][1]
        expected_mir = ratio_amp * jnp.exp(
            -1j * (-omega.real + 1j * omega.imag) * t - ratio_phase)
        assert np.allclose(
            np.array(basis[:, 1]), np.array(expected_mir), atol=1e-12)


# ---------------------------------------------------------------------------
# _compute_linear_params_and_popt convention
# ---------------------------------------------------------------------------

class TestComputeLinearParamsConvention:

    def test_A_exp_neg_i_phi_equals_c(self):
        """Verify that A * exp(-i*phi) = c for each extracted coefficient."""
        Mf, af = 1.0, 0.7
        modes = mode_list(['2.2.0', '2.2.1'], Mf, af)

        # Inject known complex coefficients
        c0 = 1.5 * np.exp(-1j * 0.3)
        c1 = 2.0 * np.exp(-1j * 1.2)

        t = jnp.linspace(0, 80, 500)
        omegar = jnp.array([m.omegar for m in modes])
        omegai = jnp.array([m.omegai for m in modes])

        # Build y from basis * coefficients
        basis = model_func_optimized(
            jnp.array([]), t, omegar, omegai, jnp.array([]),
            N_free=0, N_fix=2, include_mirror=False)
        y = basis @ jnp.array([c0, c1])
        sigma = jnp.ones(len(t))
        mask = jnp.ones(len(t))

        popt = _compute_linear_params_and_popt(
            jnp.array([]), t, y, sigma, mask,
            omegar, omegai, jnp.array([]),
            N_free=0, N_fix=2, include_mirror=False)

        popt_np = np.array(popt)
        A0, phi0 = popt_np[0], popt_np[1]
        A1, phi1 = popt_np[2], popt_np[3]

        recovered_c0 = A0 * np.exp(-1j * phi0)
        recovered_c1 = A1 * np.exp(-1j * phi1)

        assert np.isclose(recovered_c0, c0, atol=1e-10)
        assert np.isclose(recovered_c1, c1, atol=1e-10)

    def test_mirror_prograde_extraction(self):
        """With mirror, prograde coefficients at even indices are extracted."""
        Mf, af = 1.0, 0.7
        modes = mode_list(['2.2.0', '2.2.1'], Mf, af)
        iota, psi = np.pi / 3, np.pi / 2
        mirror_ratio_list = make_mirror_ratio_list(modes, iota, psi)
        mirror_arr = jnp.array(mirror_ratio_list)

        c_pro_0 = 1.0 + 0.0j
        c_mir_0 = 0.5 + 0.1j
        c_pro_1 = 0.0 - 2.0j
        c_mir_1 = 0.3 - 0.4j

        t = jnp.linspace(0, 80, 500)
        omegar = jnp.array([m.omegar for m in modes])
        omegai = jnp.array([m.omegai for m in modes])

        basis = model_func_optimized(
            jnp.array([]), t, omegar, omegai, mirror_arr,
            N_free=0, N_fix=2, include_mirror=True)
        y = basis @ jnp.array([c_pro_0, c_mir_0, c_pro_1, c_mir_1])
        sigma = jnp.ones(len(t))
        mask = jnp.ones(len(t))

        popt = _compute_linear_params_and_popt(
            jnp.array([]), t, y, sigma, mask,
            omegar, omegai, mirror_arr,
            N_free=0, N_fix=2, include_mirror=True)

        popt_np = np.array(popt)
        A0, phi0 = popt_np[0], popt_np[1]
        A1, phi1 = popt_np[2], popt_np[3]

        recovered_c0 = A0 * np.exp(-1j * phi0)
        recovered_c1 = A1 * np.exp(-1j * phi1)

        assert np.isclose(recovered_c0, c_pro_0, atol=1e-10)
        assert np.isclose(recovered_c1, c_pro_1, atol=1e-10)
