from jaxqualin.waveforms import waveform

import numpy as np
import json

import pytest

def load_test_waveform_dict(test_waveform_dict):
    h_full_real = np.array(test_waveform_dict['h_full_real'])
    h_full_imag = np.array(test_waveform_dict['h_full_imag'])
    h_full = h_full_real + 1j*h_full_imag
    time_full = np.array(test_waveform_dict['time_full'])
    Mf = test_waveform_dict['Mf']
    af = test_waveform_dict['af']
    h = waveform(time_full, h_full)
    return h, Mf, af

@pytest.fixture
def test_waveform_tuple():
    with open('test/test_waveform.json', 'r') as f:
        test_waveform_dict = json.load(f)
    return load_test_waveform_dict(test_waveform_dict)

def load_results_dict(path):
    with open(path, 'r') as f:
        results_dict = json.load(f)
    for key in results_dict.keys():
        results_dict[key] = np.array(results_dict[key])
    return results_dict

@pytest.fixture
def test_results_free():
    return load_results_dict('test/test_results_free.json')

@pytest.fixture
def test_results_fixed():
    return load_results_dict('test/test_results_fixed.json')

@pytest.fixture
def omega220r_test():
    return 0.5534871992325887

@pytest.fixture
def omega220i_test():
    return -0.08542197708263985

@pytest.fixture
def omegan220r_test():
    return -0.326523709041974

@pytest.fixture
def omegan220i_test():
    return -0.09325537321737846


