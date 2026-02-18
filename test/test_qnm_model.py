"""Tests for QNMModel, KerrModel, model_mode_free, and model_mode."""
import numpy as np
import pytest

from jaxqualin.qnmode import (
    QNMModel, KerrModel, model_mode_free, model_mode,
    mode_free, mode,
)


# ---------------------------------------------------------------------------
# QNMModel base class
# ---------------------------------------------------------------------------

class TestQNMModel:

    def test_base_raises(self):
        m = QNMModel()
        with pytest.raises(NotImplementedError):
            m.compute_omega([[2, 2, 0]])

    def test_default_bounds(self):
        m = QNMModel()
        assert m.param_bounds() == {}

    def test_n_params(self):
        m = QNMModel()
        assert m.n_params == 0


# ---------------------------------------------------------------------------
# KerrModel
# ---------------------------------------------------------------------------

class TestKerrModel:

    def test_param_names(self):
        km = KerrModel()
        assert km.param_names == ["M", "a"]
        assert km.n_params == 2

    def test_matches_mode_free(self):
        """KerrModel.compute_omega should match mode_free.fix_mode."""
        Mf, af = 1.0, 0.7
        km = KerrModel()
        omega_km = km.compute_omega([[2, 2, 0]], M=Mf, a=af)

        mf = mode_free([[2, 2, 0]])
        mf.fix_mode(Mf, af)

        assert np.isclose(np.real(omega_km), mf.omegar, rtol=1e-10)
        assert np.isclose(np.imag(omega_km), mf.omegai, rtol=1e-10)

    def test_matches_mode_free_different_spin(self):
        """Check for a different spin value."""
        Mf, af = 0.95, 0.3
        km = KerrModel()
        omega_km = km.compute_omega([[2, 2, 0]], M=Mf, a=af)

        mf = mode_free([[2, 2, 0]])
        mf.fix_mode(Mf, af)

        assert np.isclose(np.real(omega_km), mf.omegar, rtol=1e-10)
        assert np.isclose(np.imag(omega_km), mf.omegai, rtol=1e-10)

    def test_overtone(self):
        Mf, af = 1.0, 0.7
        km = KerrModel()
        omega_km = km.compute_omega([[2, 2, 1]], M=Mf, a=af)

        mf = mode_free([[2, 2, 1]])
        mf.fix_mode(Mf, af)

        assert np.isclose(np.real(omega_km), mf.omegar, rtol=1e-10)
        assert np.isclose(np.imag(omega_km), mf.omegai, rtol=1e-10)

    def test_retrograde(self):
        Mf, af = 1.0, 0.7
        km = KerrModel()
        omega_km = km.compute_omega([[-2, 2, 0]], M=Mf, a=af)

        mf = mode_free([[-2, 2, 0]])
        mf.fix_mode(Mf, af)

        assert np.isclose(np.real(omega_km), mf.omegar, rtol=1e-10)
        assert np.isclose(np.imag(omega_km), mf.omegai, rtol=1e-10)

    def test_quadratic_mode(self):
        Mf, af = 1.0, 0.7
        km = KerrModel()
        lmnx = [[2, 2, 0], [3, 3, 0]]
        omega_km = km.compute_omega(lmnx, M=Mf, a=af)

        mf = mode_free(lmnx)
        mf.fix_mode(Mf, af)

        assert np.isclose(np.real(omega_km), mf.omegar, rtol=1e-10)
        assert np.isclose(np.imag(omega_km), mf.omegai, rtol=1e-10)

    def test_spin_clamping(self):
        km = KerrModel()
        omega_high = km.compute_omega([[2, 2, 0]], M=1.0, a=1.5)
        omega_limit = km.compute_omega([[2, 2, 0]], M=1.0, a=0.99)
        assert np.isclose(omega_high, omega_limit)

    def test_constant_mode(self):
        km = KerrModel()
        omega = km.compute_omega("constant", M=1.0, a=0.7)
        assert np.isclose(omega, 0.0)

    def test_param_bounds(self):
        km = KerrModel()
        bounds = km.param_bounds()
        assert "M" in bounds
        assert "a" in bounds
        assert bounds["a"] == (-0.99, 0.99)

    def test_retro_def_orbit_false(self):
        Mf, af = 1.0, 0.7
        km = KerrModel(retro_def_orbit=False)
        omega_km = km.compute_omega([[-2, 2, 0]], M=Mf, a=af)

        mf = mode_free([[-2, 2, 0]])
        mf.fix_mode(Mf, af, retro_def_orbit=False)

        assert np.isclose(np.real(omega_km), mf.omegar, rtol=1e-10)
        assert np.isclose(np.imag(omega_km), mf.omegai, rtol=1e-10)


# ---------------------------------------------------------------------------
# Custom QNMModel subclass for testing
# ---------------------------------------------------------------------------

class SimpleShiftModel(QNMModel):
    """A trivial model for testing: omega = alpha + beta * i."""
    param_names = ["alpha", "beta"]

    def compute_omega(self, lmnx, alpha, beta, **kwargs):
        return alpha + 1j * beta

    def param_bounds(self):
        return {"alpha": (0, 10), "beta": (-5, 0)}


class TestSimpleShiftModel:

    def test_compute_omega(self):
        m = SimpleShiftModel()
        omega = m.compute_omega([[2, 2, 0]], alpha=0.5, beta=-0.08)
        assert np.isclose(omega, 0.5 - 0.08j)

    def test_param_bounds(self):
        m = SimpleShiftModel()
        bounds = m.param_bounds()
        assert bounds["alpha"] == (0, 10)
        assert bounds["beta"] == (-5, 0)

    def test_n_params(self):
        m = SimpleShiftModel()
        assert m.n_params == 2


# ---------------------------------------------------------------------------
# model_mode_free tests
# ---------------------------------------------------------------------------

class TestModelModeFree:

    def test_init_string_lmnx(self):
        km = KerrModel()
        mmf = model_mode_free("2.2.0", model=km)
        assert mmf.lmnx == [[2, 2, 0]]
        assert mmf.string() == "2.2.0"

    def test_init_list_lmnx(self):
        km = KerrModel()
        mmf = model_mode_free([[2, 2, 0]], model=km)
        assert mmf.string() == "2.2.0"

    def test_fix_mode_sets_omega(self):
        km = KerrModel()
        mmf = model_mode_free([[2, 2, 0]], model=km)
        mmf.fix_mode(M=1.0, a=0.7)
        assert hasattr(mmf, 'omegar')
        assert hasattr(mmf, 'omegai')
        assert hasattr(mmf, 'omega')

        mf = mode_free([[2, 2, 0]])
        mf.fix_mode(1.0, 0.7)
        assert np.isclose(mmf.omegar, mf.omegar, rtol=1e-10)
        assert np.isclose(mmf.omegai, mf.omegai, rtol=1e-10)

    def test_fix_mode_stores_params(self):
        km = KerrModel()
        mmf = model_mode_free([[2, 2, 0]], model=km)
        mmf.fix_mode(M=1.0, a=0.7)
        assert mmf.M == 1.0
        assert mmf.a == 0.7

    def test_label_override(self):
        km = KerrModel()
        mmf = model_mode_free([[2, 2, 0]], model=km, label="kerr_220")
        assert mmf.string() == "kerr_220"

    def test_tex_string(self):
        km = KerrModel()
        mmf = model_mode_free([[2, 2, 0]], model=km)
        tex = mmf.tex_string()
        assert "{,}" in tex

    def test_is_overtone(self):
        km = KerrModel()
        fund = model_mode_free([[2, 2, 0]], model=km)
        over = model_mode_free([[2, 2, 1]], model=km)
        assert fund.is_overtone() is False
        assert over.is_overtone() is True

    def test_sum_lm(self):
        km = KerrModel()
        mmf = model_mode_free([[2, 2, 0], [3, 3, 0]], model=km)
        assert mmf.sum_lm() == (5, 5)

    def test_constant_mode(self):
        km = KerrModel()
        mmf = model_mode_free("constant", model=km)
        assert mmf.string() == "constant"
        assert mmf.is_overtone() is False
        assert mmf.sum_lm() == (0, 0)


# ---------------------------------------------------------------------------
# model_mode tests
# ---------------------------------------------------------------------------

class TestModelMode:

    def test_init_fixes_frequency(self):
        km = KerrModel()
        mm = model_mode([[2, 2, 0]], model=km, M=1.0, a=0.7)
        assert hasattr(mm, 'omegar')
        assert hasattr(mm, 'omegai')

        mf = mode([[2, 2, 0]], 1.0, 0.7)
        assert np.isclose(mm.omegar, mf.omegar, rtol=1e-10)
        assert np.isclose(mm.omegai, mf.omegai, rtol=1e-10)

    def test_custom_model(self):
        sm = SimpleShiftModel()
        mm = model_mode([[2, 2, 0]], model=sm, alpha=0.5, beta=-0.08)
        assert np.isclose(mm.omegar, 0.5)
        assert np.isclose(mm.omegai, -0.08)
