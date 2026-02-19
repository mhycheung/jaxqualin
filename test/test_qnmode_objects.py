"""Tests for QNM mode objects: mode, mode_free, and associated utilities."""
import numpy as np
import pytest

from jaxqualin.qnmode import (
    mode, mode_free, mode_list,
    str_to_mode, str_to_lmnx,
    potential_modes, remove_duplicated_modes,
    lower_overtone_present, lmnx_to_string, qnms_to_string,
    S_mirror_fac, S_mirror_fac_complex,
    make_mirror_ratio_list,
)


Mf = 1.0
af = 0.7


# ---------------------------------------------------------------------------
# mode_free.fix_mode
# ---------------------------------------------------------------------------

class TestModeFreeFix:

    def test_fix_mode_sets_frequencies(self):
        mf = mode_free([[2, 2, 0]])
        mf.fix_mode(Mf, af)
        assert hasattr(mf, 'omegar')
        assert hasattr(mf, 'omegai')
        assert mf.omegar != 0
        assert mf.omegai != 0

    def test_fix_mode_matches_mode_class(self):
        """mode(lmnx, M, a) should give same freq as mode_free + fix_mode."""
        mf = mode_free([[2, 2, 0]])
        mf.fix_mode(Mf, af)
        m_fixed = mode([[2, 2, 0]], Mf, af)
        assert np.isclose(float(mf.omegar), float(m_fixed.omegar), rtol=1e-12)
        assert np.isclose(float(mf.omegai), float(m_fixed.omegai), rtol=1e-12)

    def test_spin_clamping_high(self):
        mf = mode_free([[2, 2, 0]])
        mf.fix_mode(Mf, 0.999)
        m_clamped = mode([[2, 2, 0]], Mf, 0.99)
        assert np.isclose(float(mf.omegar), float(m_clamped.omegar), rtol=1e-10)

    def test_spin_clamping_low(self):
        mf = mode_free([[2, 2, 0]])
        mf.fix_mode(Mf, -0.999)
        m_clamped = mode([[2, 2, 0]], Mf, -0.99)
        assert np.isclose(float(mf.omegar), float(m_clamped.omegar), rtol=1e-10)


# ---------------------------------------------------------------------------
# mode_free.string roundtrip
# ---------------------------------------------------------------------------

class TestStringRoundtrip:

    def test_mode_string_roundtrip(self):
        m_orig = mode([[2, 2, 0]], Mf, af)
        s = m_orig.string()
        m_recovered = str_to_mode(s, Mf, af)
        assert np.isclose(float(m_orig.omegar), float(m_recovered.omegar), rtol=1e-12)
        assert np.isclose(float(m_orig.omegai), float(m_recovered.omegai), rtol=1e-12)

    def test_quadratic_string_roundtrip(self):
        m_orig = mode([[2, 2, 0], [3, 3, 0]], Mf, af)
        s = m_orig.string()
        assert s == '2.2.0x3.3.0'
        m_recovered = str_to_mode(s, Mf, af)
        assert np.isclose(float(m_orig.omegar), float(m_recovered.omegar), rtol=1e-12)

    def test_constant_string(self):
        mf = mode_free("constant")
        assert mf.string() == "constant"


# ---------------------------------------------------------------------------
# Sign conventions
# ---------------------------------------------------------------------------

class TestSignConventions:

    def test_prograde_220_positive_spin(self):
        """Prograde (2,2,0) with positive spin: omegar > 0 and omegai < 0."""
        m = mode([[2, 2, 0]], Mf, af)
        assert float(m.omegar) > 0
        assert float(m.omegai) < 0

    def test_retrograde_neg2_2_0(self):
        """Retrograde mode (-2,2,0)."""
        m_retro = mode([[-2, 2, 0]], Mf, af)
        m_pro = mode([[2, 2, 0]], Mf, af)
        # Retrograde should have the opposite sign of omegar
        assert np.sign(float(m_retro.omegar)) != np.sign(float(m_pro.omegar))


# ---------------------------------------------------------------------------
# mode_list
# ---------------------------------------------------------------------------

class TestModeList:

    def test_returns_correct_count(self):
        modes = mode_list(['2.2.0', '2.2.1'], Mf, af)
        assert len(modes) == 2

    def test_mode_strings_match(self):
        modes = mode_list(['2.2.0', '2.2.1'], Mf, af)
        assert modes[0].string() == '2.2.0'
        assert modes[1].string() == '2.2.1'

    def test_empty_list(self):
        modes = mode_list([], Mf, af)
        assert len(modes) == 0


# ---------------------------------------------------------------------------
# potential_modes
# ---------------------------------------------------------------------------

class TestPotentialModes:

    def test_contains_overtones(self):
        relevant_lm_list = [[2, 2]]
        pm = potential_modes(2, 2, Mf, af, relevant_lm_list, return_lmnx=True)
        assert '2.2.0' in pm
        assert '2.2.1' in pm

    def test_contains_constant(self):
        relevant_lm_list = [[2, 2]]
        pm = potential_modes(2, 2, Mf, af, relevant_lm_list, return_lmnx=True)
        assert 'constant' in pm

    def test_no_constant_when_excluded(self):
        relevant_lm_list = [[2, 2]]
        pm = potential_modes(
            2, 2, Mf, af, relevant_lm_list,
            return_lmnx=True, include_constant=False)
        assert 'constant' not in pm


# ---------------------------------------------------------------------------
# remove_duplicated_modes
# ---------------------------------------------------------------------------

class TestRemoveDuplicatedModes:

    def test_removes_duplicates(self):
        modes = mode_list(['2.2.0', '2.2.0'], Mf, af)
        clean = remove_duplicated_modes(modes)
        assert len(clean) == 1

    def test_keeps_distinct_modes(self):
        modes = mode_list(['2.2.0', '2.2.1'], Mf, af)
        clean = remove_duplicated_modes(modes)
        assert len(clean) == 2


# ---------------------------------------------------------------------------
# lower_overtone_present
# ---------------------------------------------------------------------------

class TestLowerOvertonePresent:

    def test_overtone_with_lower_present(self):
        found = mode_list(['2.2.0', '2.2.1'], Mf, af)
        test = mode_list(['2.2.1'], Mf, af)[0]
        assert lower_overtone_present(test, found) is True

    def test_overtone_without_lower(self):
        found = mode_list(['2.2.1'], Mf, af)
        test = mode_list(['2.2.1'], Mf, af)[0]
        assert lower_overtone_present(test, found) is False

    def test_fundamental_always_true(self):
        found = mode_list(['3.3.0'], Mf, af)
        test = mode_list(['2.2.0'], Mf, af)[0]
        assert lower_overtone_present(test, found) is True

    def test_constant_always_true(self):
        test = mode("constant", Mf, af)
        found = mode_list(['2.2.0'], Mf, af)
        assert lower_overtone_present(test, found) is True


# ---------------------------------------------------------------------------
# S_mirror_fac and S_mirror_fac_complex
# ---------------------------------------------------------------------------

class TestMirrorFunctions:

    def test_real_complex_consistent(self):
        """Real-valued ratio should equal |complex ratio|."""
        l, m, n = 2, 2, 0
        iota, psi = np.pi / 3, np.pi / 2
        real_ratio = S_mirror_fac(iota, af, l, m, n, psi)
        complex_ratio = S_mirror_fac_complex(iota, af, l, m, n, psi)
        assert np.isclose(real_ratio, np.abs(complex_ratio), rtol=1e-10)

    def test_face_on_limit(self):
        """At iota=0 (face-on), for m=2: S(-m)/S(m) has a known structure."""
        l, m, n = 2, 2, 0
        iota = 0.0
        ratio = S_mirror_fac(iota, af, l, m, n)
        # At face-on, retrograde angular function should be very small
        # relative to prograde for m>0
        assert ratio < 0.1


# ---------------------------------------------------------------------------
# make_mirror_ratio_list
# ---------------------------------------------------------------------------

class TestMakeMirrorRatioList:

    def test_length_matches_modes(self):
        modes = mode_list(['2.2.0', '2.2.1'], Mf, af)
        iota, psi = np.pi / 3, np.pi / 2
        ratio_list = make_mirror_ratio_list(modes, iota, psi)
        assert len(ratio_list) == 2

    def test_ratio_is_pair(self):
        modes = mode_list(['2.2.0'], Mf, af)
        iota, psi = np.pi / 3, np.pi / 2
        ratio_list = make_mirror_ratio_list(modes, iota, psi)
        # Each ratio is [amplitude_ratio, phase_diff]
        assert len(ratio_list[0]) == 2

    def test_raises_for_quadratic(self):
        m = mode([[2, 2, 0], [3, 3, 0]], Mf, af)
        iota, psi = np.pi / 3, np.pi / 2
        with pytest.raises(NotImplementedError):
            make_mirror_ratio_list([m], iota, psi)
