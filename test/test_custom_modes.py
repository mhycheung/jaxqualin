"""Tests for custom_mode, custom_mode_list, and integration with QNMFit."""
import numpy as np
import pytest

from jaxqualin.qnmode import custom_mode, custom_mode_list
from jaxqualin.waveforms import waveform
from jaxqualin.fit import QNMFit, QNMFitVaryingStartingTime


# ---------------------------------------------------------------------------
# custom_mode unit tests
# ---------------------------------------------------------------------------

class TestCustomMode:

    def test_basic_attributes(self):
        cm = custom_mode(0.5 - 0.08j)
        assert np.isclose(cm.omegar, 0.5)
        assert np.isclose(cm.omegai, -0.08)
        assert cm.lmnx is None

    def test_label_string(self):
        cm = custom_mode(0.5 - 0.08j, label="my_mode")
        assert cm.string() == "my_mode"
        assert cm.tex_string() == "$my_mode$"

    def test_auto_label(self):
        cm = custom_mode(0.5 - 0.08j)
        cm._auto_index = 3
        assert cm.string() == "mode_3"

    def test_no_label_no_index(self):
        cm = custom_mode(0.5 - 0.08j)
        assert cm.string() == "custom"

    def test_lmn_label_tex(self):
        cm = custom_mode(0.5 - 0.08j, label="2.2.0")
        tex = cm.tex_string()
        assert "{,}" in tex

    def test_quadratic_label_tex(self):
        cm = custom_mode(0.5 - 0.08j, label="2.2.0x3.3.0")
        tex = cm.tex_string()
        assert r"\times" in tex

    def test_is_overtone(self):
        cm = custom_mode(0.5 - 0.08j)
        assert cm.is_overtone() is False

    def test_sum_lm(self):
        cm = custom_mode(0.5 - 0.08j)
        assert cm.sum_lm() == (0, 0)

    def test_fix_mode_is_noop(self):
        cm = custom_mode(0.5 - 0.08j)
        cm.fix_mode(1.0, 0.7)
        assert np.isclose(cm.omegar, 0.5)


# ---------------------------------------------------------------------------
# custom_mode_list unit tests
# ---------------------------------------------------------------------------

class TestCustomModeList:

    def test_auto_labels(self):
        omegas = [0.5 - 0.08j, 0.3 - 0.05j]
        modes = custom_mode_list(omegas)
        assert modes[0].string() == "mode_0"
        assert modes[1].string() == "mode_1"

    def test_explicit_labels(self):
        omegas = [0.5 - 0.08j, 0.3 - 0.05j]
        modes = custom_mode_list(omegas, labels=["fund", "over"])
        assert modes[0].string() == "fund"
        assert modes[1].string() == "over"

    def test_mixed_labels(self):
        omegas = [0.5 - 0.08j, 0.3 - 0.05j]
        modes = custom_mode_list(omegas, labels=["fund", None])
        assert modes[0].string() == "fund"
        assert modes[1].string() == "mode_1"

    def test_len(self):
        modes = custom_mode_list([0.5j, 0.3j, 0.1j])
        assert len(modes) == 3


# ---------------------------------------------------------------------------
# Integration: custom_mode with QNMFit
# ---------------------------------------------------------------------------

def _make_custom_waveform(modes, A_list, phi_list, t_arr):
    """Build a synthetic waveform from custom_mode objects."""
    h_arr = np.zeros(len(t_arr), dtype=np.complex128)
    for m, A, phi in zip(modes, A_list, phi_list):
        omega_c = complex(m.omegar) + 1j * complex(m.omegai)
        h_arr += A * np.exp(-1j * (omega_c * t_arr + phi))
    return waveform(t_arr, h_arr, t_peak=0)


class TestCustomModeQNMFit:

    def test_single_custom_fixed_fit(self):
        omega = 0.5 - 0.08j
        modes = [custom_mode(omega, label="test_mode")]
        A_true, phi_true = 2.0, 0.3
        t = np.linspace(0, 100, 2000)
        h = _make_custom_waveform(modes, [A_true], [phi_true], t)

        fitter = QNMFit(h, t0=0.0, N_free=0, qnm_fixed_list=modes)
        fitter.do_fit()
        popt = np.array(fitter.popt)

        assert np.isclose(popt[0], A_true, atol=1e-5)
        assert np.isclose(popt[1], phi_true, atol=1e-5)
        assert fitter.mismatch < 1e-10

    def test_two_custom_fixed_modes(self):
        modes = custom_mode_list(
            [0.5 - 0.08j, 0.45 - 0.09j],
            labels=["fundamental", "overtone"])
        A_list = [1.0, 3.0]
        phi_list = [0.0, np.pi / 4]
        t = np.linspace(0, 100, 2000)
        h = _make_custom_waveform(modes, A_list, phi_list, t)

        fitter = QNMFit(h, t0=0.0, N_free=0, qnm_fixed_list=modes)
        fitter.do_fit()
        assert fitter.mismatch < 1e-10

    def test_custom_mode_label_in_result(self):
        modes = custom_mode_list(
            [0.5 - 0.08j], labels=["my_fund"])
        A, phi = 1.0, 0.0
        t = np.linspace(0, 100, 2000)
        h = _make_custom_waveform(modes, [A], [phi], t)

        fitter = QNMFitVaryingStartingTime(
            h, np.array([0.0]), N_free=0,
            qnm_fixed_list=modes, load_pickle=False,
            run_string_prefix='custom_label_test',
            save_results=False)
        fitter.do_fits()
        result = fitter.result_full
        assert "A_my_fund" in result.A_dict


# ---------------------------------------------------------------------------
# Mirror mode guard
# ---------------------------------------------------------------------------

class TestMirrorModeGuard:

    def test_mirror_raises_for_custom_mode(self):
        from jaxqualin.fit import qnm_fit_func_varMa_mirror
        modes = [custom_mode(0.5 - 0.08j)]
        with pytest.raises(ValueError, match="Mirror mode fitting"):
            qnm_fit_func_varMa_mirror(
                np.linspace(0, 10, 100),
                modes, [], [[1.0, 0.0]], [],
                0.5, 0.0, 1.0, 0.7)
