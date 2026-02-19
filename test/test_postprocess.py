"""Tests for postprocessing mode classifiers."""
import numpy as np
import pandas as pd
import pytest

from jaxqualin.postprocess import (
    is_quadratic,
    is_overtone,
    is_fundamental,
    is_retro,
    harm_type,
    classify_modes,
)


# ---------------------------------------------------------------------------
# is_quadratic
# ---------------------------------------------------------------------------

class TestIsQuadratic:

    def test_quadratic_mode(self):
        row = {'mode_string': '2.2.0x2.2.0'}
        assert is_quadratic(row) is True

    def test_linear_mode(self):
        row = {'mode_string': '2.2.0'}
        assert is_quadratic(row) is False

    def test_constant_mode(self):
        row = {'mode_string': 'constant'}
        assert is_quadratic(row) is False


# ---------------------------------------------------------------------------
# is_overtone
# ---------------------------------------------------------------------------

class TestIsOvertone:

    def test_overtone(self):
        row = {'mode_string': '2.2.1'}
        assert is_overtone(row) is True

    def test_fundamental(self):
        row = {'mode_string': '2.2.0'}
        assert is_overtone(row) is False

    def test_quadratic_with_overtone(self):
        row = {'mode_string': '2.2.0x2.2.1'}
        assert is_overtone(row) is True

    def test_constant(self):
        row = {'mode_string': 'constant'}
        assert is_overtone(row) is False


# ---------------------------------------------------------------------------
# is_fundamental
# ---------------------------------------------------------------------------

class TestIsFundamental:

    def test_fundamental(self):
        row = {'mode_string': '2.2.0'}
        assert is_fundamental(row) is True

    def test_overtone(self):
        row = {'mode_string': '2.2.1'}
        assert is_fundamental(row) is False

    def test_quadratic(self):
        row = {'mode_string': '2.2.0x2.2.0'}
        assert is_fundamental(row) is False

    def test_constant(self):
        row = {'mode_string': 'constant'}
        assert is_fundamental(row) is False


# ---------------------------------------------------------------------------
# is_retro
# ---------------------------------------------------------------------------

class TestIsRetro:

    def test_retrograde(self):
        row = {'mode_string': '-2.2.0', 'retro': True}
        assert is_retro(row) is True

    def test_prograde(self):
        row = {'mode_string': '2.2.0', 'retro': False}
        assert is_retro(row) is False

    def test_constant(self):
        row = {'mode_string': 'constant', 'retro': False}
        assert is_retro(row) is False


# ---------------------------------------------------------------------------
# harm_type
# ---------------------------------------------------------------------------

class TestHarmType:

    def test_basic(self):
        """Mode (2,2,0) in the (l=2,m=2) harmonic -> basic."""
        row = {'l': 2, 'm': 2, 'mode_string': '2.2.0'}
        assert harm_type(row) == 'basic'

    def test_mixing(self):
        """Mode (3,2,0) in the (l=2,m=2) harmonic -> mixing (l differs)."""
        row = {'l': 2, 'm': 2, 'mode_string': '3.2.0'}
        assert harm_type(row) == 'mixing'

    def test_recoil(self):
        """Mode (3,3,0) in the (l=2,m=2) harmonic -> recoil (m differs)."""
        row = {'l': 2, 'm': 2, 'mode_string': '3.3.0'}
        assert harm_type(row) == 'recoil'

    def test_constant(self):
        row = {'l': 2, 'm': 2, 'mode_string': 'constant'}
        assert harm_type(row) == 'constant'


# ---------------------------------------------------------------------------
# classify_modes
# ---------------------------------------------------------------------------

class TestClassifyModes:

    def test_adds_all_columns(self):
        df = pd.DataFrame([
            {'l': 2, 'm': 2, 'mode_string': '2.2.0', 'retro': False,
             'q': 1.0, 'chi_1_z': 0.0, 'chi_2_z': 0.0, 'chi_rem': 0.7},
            {'l': 2, 'm': 2, 'mode_string': '2.2.1', 'retro': False,
             'q': 1.0, 'chi_1_z': 0.0, 'chi_2_z': 0.0, 'chi_rem': 0.7},
            {'l': 2, 'm': 2, 'mode_string': '2.2.0x2.2.0', 'retro': False,
             'q': 1.0, 'chi_1_z': 0.0, 'chi_2_z': 0.0, 'chi_rem': 0.7},
            {'l': 2, 'm': 2, 'mode_string': 'constant', 'retro': False,
             'q': 1.0, 'chi_1_z': 0.0, 'chi_2_z': 0.0, 'chi_rem': 0.7},
        ])
        result = classify_modes(df)

        expected_cols = ['is_quadratic', 'is_fundamental', 'is_overtone',
                         'is_retrograde', 'harm_type', 'natural_l', 'natural_m']
        for col in expected_cols:
            assert col in result.columns

    def test_classification_values(self):
        df = pd.DataFrame([
            {'l': 2, 'm': 2, 'mode_string': '2.2.0', 'retro': False,
             'q': 1.0, 'chi_1_z': 0.0, 'chi_2_z': 0.0, 'chi_rem': 0.7},
            {'l': 2, 'm': 2, 'mode_string': '2.2.1', 'retro': False,
             'q': 1.0, 'chi_1_z': 0.0, 'chi_2_z': 0.0, 'chi_rem': 0.7},
        ])
        result = classify_modes(df)

        # First row: fundamental
        assert result.iloc[0]['is_fundamental'] == True
        assert result.iloc[0]['is_overtone'] == False
        assert result.iloc[0]['is_quadratic'] == False

        # Second row: overtone
        assert result.iloc[1]['is_fundamental'] == False
        assert result.iloc[1]['is_overtone'] == True
        assert result.iloc[1]['is_quadratic'] == False
