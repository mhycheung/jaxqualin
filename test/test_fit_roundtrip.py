"""Roundtrip tests for the VARPRO fitting pipeline.

These tests create synthetic waveforms with known parameters,
fit them, and verify the parameters are recovered correctly.
"""
import numpy as np
import pytest

from jaxqualin.waveforms import waveform, clean_QNM
from jaxqualin.qnmode import mode_list, mode_free, make_mirror_ratio_list, long_str_to_qnms_free
from jaxqualin.fit import QNMFit, QNMFitVarMa, QNMFitVaryingStartingTime
from jaxqualin.utils import all_close_to


def _make_clean_waveform(modes, A_list, phi_list, t_arr):
    """Build a synthetic waveform from a sum of clean QNMs."""
    h_arr = np.zeros(len(t_arr), dtype=np.complex128)
    for mode, A, phi in zip(modes, A_list, phi_list):
        h_arr += clean_QNM(mode, t_arr, A, phi)
    return waveform(t_arr, h_arr, t_peak=0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_mode_220():
    """A single (2,2,0) mode with known A and phi."""
    Mf, af = 1.0, 0.7
    modes = mode_list(['2.2.0'], Mf, af)
    A, phi = 2.5, 0.3
    t = np.linspace(0, 100, 2000)
    h = _make_clean_waveform(modes, [A], [phi], t)
    return h, modes, A, phi, Mf, af


@pytest.fixture
def two_mode_220_221():
    """Two fixed modes (2,2,0) and (2,2,1) with known amplitudes."""
    Mf, af = 1.0, 0.7
    modes = mode_list(['2.2.0', '2.2.1'], Mf, af)
    A_list = [1.0, 3.0]
    phi_list = [0.0, np.pi / 4]
    t = np.linspace(0, 100, 2000)
    h = _make_clean_waveform(modes, A_list, phi_list, t)
    return h, modes, A_list, phi_list, Mf, af


# ---------------------------------------------------------------------------
# Single fixed mode roundtrip
# ---------------------------------------------------------------------------

class TestSingleFixedModeRoundtrip:

    def test_amplitude_recovered(self, single_mode_220):
        h, modes, A, phi, Mf, af = single_mode_220
        fitter = QNMFit(h, t0=0.0, N_free=0, qnm_fixed_list=modes)
        fitter.do_fit()
        popt = np.array(fitter.popt)
        assert np.isclose(popt[0], A, atol=1e-6)

    def test_phase_recovered(self, single_mode_220):
        h, modes, A, phi, Mf, af = single_mode_220
        fitter = QNMFit(h, t0=0.0, N_free=0, qnm_fixed_list=modes)
        fitter.do_fit()
        popt = np.array(fitter.popt)
        assert np.isclose(popt[1], phi, atol=1e-6)

    def test_mismatch_near_zero(self, single_mode_220):
        h, modes, A, phi, Mf, af = single_mode_220
        fitter = QNMFit(h, t0=0.0, N_free=0, qnm_fixed_list=modes)
        fitter.do_fit()
        assert fitter.mismatch < 1e-10


# ---------------------------------------------------------------------------
# Two fixed modes roundtrip
# ---------------------------------------------------------------------------

class TestTwoFixedModesRoundtrip:

    def test_amplitudes_recovered(self, two_mode_220_221):
        h, modes, A_list, phi_list, Mf, af = two_mode_220_221
        fitter = QNMFit(h, t0=0.0, N_free=0, qnm_fixed_list=modes)
        fitter.do_fit()
        popt = np.array(fitter.popt)
        assert np.isclose(popt[0], A_list[0], atol=1e-6)
        assert np.isclose(popt[2], A_list[1], atol=1e-6)

    def test_phases_recovered(self, two_mode_220_221):
        h, modes, A_list, phi_list, Mf, af = two_mode_220_221
        fitter = QNMFit(h, t0=0.0, N_free=0, qnm_fixed_list=modes)
        fitter.do_fit()
        popt = np.array(fitter.popt)
        assert np.isclose(popt[1], phi_list[0], atol=1e-6)
        assert np.isclose(popt[3], phi_list[1], atol=1e-6)

    def test_mismatch_near_zero(self, two_mode_220_221):
        h, modes, A_list, phi_list, Mf, af = two_mode_220_221
        fitter = QNMFit(h, t0=0.0, N_free=0, qnm_fixed_list=modes)
        fitter.do_fit()
        assert fitter.mismatch < 1e-10


# ---------------------------------------------------------------------------
# Free mode roundtrip
# ---------------------------------------------------------------------------

class TestFreeModeRoundtrip:

    def test_free_mode_recovers_frequency(self):
        Mf, af = 1.0, 0.7
        modes = mode_list(['2.2.0'], Mf, af)
        target_omegar = modes[0].omegar
        target_omegai = modes[0].omegai

        A, phi = 1.5, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_clean_waveform(modes, [A], [phi], t)

        fitter = QNMFit(
            h, t0=0.0, N_free=1, qnm_fixed_list=[],
            guess_free=[1, 1, target_omegar * 0.9, target_omegai * 0.9])
        fitter.do_fit()
        popt = np.array(fitter.popt)
        recovered_omegar = popt[2]
        recovered_omegai = popt[3]
        assert np.isclose(recovered_omegar, target_omegar, rtol=1e-3)
        assert np.isclose(recovered_omegai, target_omegai, rtol=1e-3)


# ---------------------------------------------------------------------------
# Mirror fit roundtrip
# ---------------------------------------------------------------------------

class TestMirrorFitRoundtrip:

    @pytest.mark.parametrize("iota,psi", [
        (np.pi / 3, np.pi / 2),
        (np.pi / 6, 0.0),
        (np.pi / 2, np.pi / 4),
    ])
    def test_mirror_recovers_amplitudes(self, iota, psi):
        Mf, af = 1, 0.7
        modes_prograde = mode_list(['2.2.0', '2.2.1'], Mf, af)
        mirror_ratio_list = make_mirror_ratio_list(modes_prograde, iota, psi)

        A220, phi220 = 1.0, 0.0
        A221, phi221 = 2.0, np.pi / 3

        all_modes = mode_list(
            ['2.2.0', '2.-2.0', '2.2.1', '2.-2.1'], Mf, af)
        A_phi_dict = {}
        for i, (A, phi) in enumerate([(A220, phi220), (A221, phi221)]):
            Ac = A * np.exp(-1j * phi)
            Amc = np.conj(Ac) * mirror_ratio_list[i][0] * np.exp(
                1j * mirror_ratio_list[i][1])
            pro_str = modes_prograde[i].string()
            retro_str = all_modes[2 * i + 1].string()
            A_phi_dict[pro_str] = (A, phi)
            A_phi_dict[retro_str] = (np.abs(Amc), -np.angle(Amc))

        t_arr = np.linspace(0, 120, 1000)
        h_arr = np.zeros(t_arr.shape, dtype=np.complex128)
        for m in all_modes:
            Av, phiv = A_phi_dict[m.string()]
            h_arr += clean_QNM(m, t_arr, Av, phiv)
        h = waveform(t_arr, h_arr, t_peak=0)

        fitter = QNMFitVaryingStartingTime(
            h, np.array([0.0]), N_free=0,
            qnm_fixed_list=modes_prograde, load_pickle=False,
            run_string_prefix='mirror_roundtrip',
            include_mirror=True, iota=iota, psi=psi)
        fitter.do_fits()
        result = fitter.result_full

        assert np.isclose(
            np.array(result.A_dict['A_2.2.0'])[0], A220, atol=1e-4)
        assert np.isclose(
            np.array(result.A_dict['A_2.2.1'])[0], A221, atol=1e-4)


# ---------------------------------------------------------------------------
# Reconstruction accuracy
# ---------------------------------------------------------------------------

class TestReconstructionAccuracy:

    def test_reconstruction_matches_input(self, two_mode_220_221):
        h, modes, A_list, phi_list, Mf, af = two_mode_220_221
        fitter = QNMFitVaryingStartingTime(
            h, np.array([0.0]), N_free=0,
            qnm_fixed_list=modes, load_pickle=False,
            run_string_prefix='recon_test')
        fitter.do_fits()
        result = fitter.result_full

        time_pm, hr_pm, hi_pm = h.postmerger(0.0)
        h_data = np.array(hr_pm + 1j * hi_pm)
        h_recon = np.array(result.reconstruct_waveform(0, np.array(time_pm)))

        mismatch = 1 - abs(np.vdot(h_data, h_recon)) / (
            np.linalg.norm(h_data) * np.linalg.norm(h_recon))
        assert mismatch < 1e-10


# ---------------------------------------------------------------------------
# Varying starting time consistency
# ---------------------------------------------------------------------------

class TestVaryingStartingTimeConsistency:

    def test_amplitude_constant_across_t0(self):
        """For a clean QNM with absolute-time basis, the fitted amplitude
        A = |c| should be the same regardless of starting time, since the
        basis exp(-i*omega*t) uses absolute time."""
        Mf, af = 1.0, 0.7
        modes = mode_list(['2.2.0'], Mf, af)
        A0, phi0 = 2.0, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_clean_waveform(modes, [A0], [phi0], t)

        t0_arr = np.linspace(0, 10, 6)
        fitter = QNMFitVaryingStartingTime(
            h, t0_arr, N_free=0,
            qnm_fixed_list=modes, load_pickle=False,
            run_string_prefix='vst_test')
        fitter.do_fits()
        result = fitter.result_full
        A_arr = np.array(result.A_dict['A_2.2.0'])

        assert np.allclose(A_arr, A0, rtol=1e-5)

    def test_phase_constant_across_t0(self):
        """For a clean QNM with absolute-time basis, the fitted phase
        should be the same regardless of starting time."""
        Mf, af = 1.0, 0.7
        modes = mode_list(['2.2.0'], Mf, af)
        A0, phi0 = 2.0, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_clean_waveform(modes, [A0], [phi0], t)

        t0_arr = np.linspace(0, 10, 6)
        fitter = QNMFitVaryingStartingTime(
            h, t0_arr, N_free=0,
            qnm_fixed_list=modes, load_pickle=False,
            run_string_prefix='vst_test2')
        fitter.do_fits()
        result = fitter.result_full
        phi_arr = np.array(result.phi_dict['phi_2.2.0'])

        diff = np.angle(np.exp(1j * (phi_arr - phi0)))
        assert np.allclose(diff, 0, atol=1e-5)


# ---------------------------------------------------------------------------
# Kerr (non-Schwarzschild) QNMFitVarMa roundtrips
# ---------------------------------------------------------------------------

class TestKerrVarMaRoundtrip:
    """Roundtrip tests for QNMFitVarMa with non-zero spin (Kerr)."""

    def test_recovers_A_phi_M_a(self):
        """Fit a single free-frequency mode and recover A, phi, M, and a."""
        Mf_true, af_true = 1.0, 0.7
        modes = mode_list(['2.2.0'], Mf_true, af_true)
        A, phi = 2.0, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_clean_waveform(modes, [A], [phi], t)

        qnm_free = long_str_to_qnms_free('2.2.0')
        fitter = QNMFitVarMa(
            h, t0=0.0, qnm_free_list=qnm_free,
            real=False,
            guess_free=[A * 0.8, phi * 0.8],
            guess_M_a=[Mf_true * 0.9, af_true * 0.9])
        fitter.do_fit()
        popt = np.array(fitter.popt)

        recovered_A = popt[0]
        recovered_phi = popt[1]
        recovered_M = popt[2]
        recovered_a = popt[3]

        assert np.isclose(recovered_A, A, rtol=0.05), \
            f"A: expected {A}, got {recovered_A}"
        assert np.isclose(recovered_M, Mf_true, rtol=0.05), \
            f"M: expected {Mf_true}, got {recovered_M}"
        assert np.isclose(recovered_a, af_true, rtol=0.05), \
            f"a: expected {af_true}, got {recovered_a}"

    def test_popt_length_kerr(self):
        """popt should have length 2*N_fix + 2*N_free + 2 (M and a)."""
        Mf_true, af_true = 1.0, 0.7
        modes_fixed = mode_list(['2.2.0'], Mf_true, af_true)
        A, phi = 1.0, 0.0
        t = np.linspace(0, 100, 2000)
        h = _make_clean_waveform(modes_fixed, [A], [phi], t)

        qnm_free = long_str_to_qnms_free('2.2.1')
        fitter = QNMFitVarMa(
            h, t0=0.0, qnm_free_list=qnm_free,
            qnm_fixed_list=modes_fixed,
            real=False,
            guess_M_a=[Mf_true * 0.9, af_true * 0.9])
        fitter.do_fit()
        popt = np.array(fitter.popt)

        N_fix = len(modes_fixed)
        N_free = len(qnm_free)
        expected_len = 2 * N_fix + 2 * N_free + 2
        assert len(popt) == expected_len, \
            f"popt length: expected {expected_len}, got {len(popt)}"

    def test_mismatch_near_zero(self):
        """A clean QNM fitted with correct guess should give small mismatch."""
        Mf_true, af_true = 1.0, 0.7
        modes = mode_list(['2.2.0'], Mf_true, af_true)
        A, phi = 2.0, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_clean_waveform(modes, [A], [phi], t)

        qnm_free = long_str_to_qnms_free('2.2.0')
        fitter = QNMFitVarMa(
            h, t0=0.0, qnm_free_list=qnm_free,
            real=False,
            guess_free=[A * 0.8, phi * 0.8],
            guess_M_a=[Mf_true * 0.9, af_true * 0.9])
        fitter.do_fit()
        assert fitter.mismatch < 1e-4, f"Mismatch too large: {fitter.mismatch}"

    def test_two_modes_fixed_plus_free(self):
        """One fixed mode and one free-frequency mode in Kerr."""
        Mf_true, af_true = 1.0, 0.7
        modes_fixed = mode_list(['2.2.0'], Mf_true, af_true)
        modes_all = mode_list(['2.2.0', '2.2.1'], Mf_true, af_true)
        A_list = [1.0, 0.5]
        phi_list = [0.0, np.pi / 4]
        t = np.linspace(0, 100, 2000)
        h = _make_clean_waveform(modes_all, A_list, phi_list, t)

        qnm_free = long_str_to_qnms_free('2.2.1')
        fitter = QNMFitVarMa(
            h, t0=0.0, qnm_free_list=qnm_free,
            qnm_fixed_list=modes_fixed,
            real=False,
            guess_M_a=[Mf_true * 0.95, af_true * 0.95])
        fitter.do_fit()
        assert fitter.mismatch < 1e-3, f"Mismatch too large: {fitter.mismatch}"


# ---------------------------------------------------------------------------
# Kerr VarMa via QNMFitVaryingStartingTime
# ---------------------------------------------------------------------------

class TestKerrVarMaVaryingStartingTime:
    """Full pipeline tests for QNMFitVaryingStartingTime with var_M_a in Kerr."""

    def test_recovers_M_and_a(self):
        """VarMa path: should recover both M and a, and Ma_dict has both."""
        Mf_true, af_true = 1.0, 0.7
        modes_fixed = mode_list(['2.2.0'], Mf_true, af_true)
        qnm_free = long_str_to_qnms_free('2.2.0')

        A, phi = 2.0, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_clean_waveform(modes_fixed, [A], [phi], t)

        t0_arr = np.array([0.0, 5.0])
        fitter = QNMFitVaryingStartingTime(
            h, t0_arr, N_free=0,
            qnm_fixed_list=[],
            qnm_free_list=qnm_free,
            var_M_a=True,
            real=False,
            load_pickle=False,
            run_string_prefix='kerr_varMa_test',
            save_results=False)
        fitter.do_fits()
        result = fitter.result_full

        assert "M" in result.Ma_dict, "Ma_dict should contain 'M'"
        assert "a" in result.Ma_dict, "Ma_dict should contain 'a' for Kerr"

        M_arr = np.array(result.Ma_dict["M"])
        a_arr = np.array(result.Ma_dict["a"])
        assert np.allclose(M_arr, Mf_true, rtol=0.05), \
            f"M not recovered: expected {Mf_true}, got {M_arr}"
        assert np.allclose(a_arr, af_true, rtol=0.05), \
            f"a not recovered: expected {af_true}, got {a_arr}"

    def test_mismatch_small(self):
        """VarMa Kerr path: mismatch should be small."""
        Mf_true, af_true = 1.0, 0.7
        modes_fixed = mode_list(['2.2.0'], Mf_true, af_true)
        qnm_free = long_str_to_qnms_free('2.2.0')

        A, phi = 2.0, 0.5
        t = np.linspace(0, 100, 2000)
        h = _make_clean_waveform(modes_fixed, [A], [phi], t)

        t0_arr = np.array([0.0])
        fitter = QNMFitVaryingStartingTime(
            h, t0_arr, N_free=0,
            qnm_fixed_list=[],
            qnm_free_list=qnm_free,
            var_M_a=True,
            real=False,
            load_pickle=False,
            run_string_prefix='kerr_varMa_mm_test',
            save_results=False)
        fitter.do_fits()
        result = fitter.result_full
        mismatch = result.mismatch_arr[0]
        assert mismatch < 1e-3, f"Mismatch too large: {mismatch}"

    def test_popt_shape_kerr(self):
        """popt_full should have 2*N_fix + 2*N_free + 2 rows (M and a)."""
        Mf_true, af_true = 1.0, 0.7
        modes_fixed = mode_list(['2.2.0'], Mf_true, af_true)
        qnm_free = long_str_to_qnms_free('2.2.0')

        A, phi = 1.0, 0.0
        t = np.linspace(0, 100, 2000)
        h = _make_clean_waveform(modes_fixed, [A], [phi], t)

        t0_arr = np.array([0.0, 5.0, 10.0])
        fitter = QNMFitVaryingStartingTime(
            h, t0_arr, N_free=0,
            qnm_fixed_list=[],
            qnm_free_list=qnm_free,
            var_M_a=True,
            real=False,
            load_pickle=False,
            run_string_prefix='kerr_varMa_shape',
            save_results=False)
        fitter.do_fits()
        result = fitter.result_full

        N_fix = 0
        N_free = len(qnm_free)
        expected_popt_rows = 2 * N_fix + 2 * N_free + 2
        assert result.popt_full.shape == (expected_popt_rows, len(t0_arr)), \
            f"popt_full shape: expected ({expected_popt_rows}, {len(t0_arr)}), " \
            f"got {result.popt_full.shape}"
