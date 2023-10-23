from jaxqualin.waveforms import waveform, clean_QNM
from jaxqualin.qnmode import mode_list, make_mirror_ratio_list

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

@pytest.fixture
def S_mirror_fac_220_test():
    return 0.0860378659486808-0.0005612628166044703j

@pytest.fixture
def mirror_waveform():
    iota = np.pi/3
    psi = np.pi/2
    Mf = 1
    af = 0.7

    modes_prograde = mode_list(['2.2.0', '2.2.1', '3.2.0'], Mf, af)
    mirror_ratio_list = make_mirror_ratio_list(modes_prograde, iota, psi)

    A220, phi220 = 1., 0.
    A221, phi221 = 3., np.pi/2
    A320, phi320 = 1e-2, np.pi

    Ac220 = A220*np.exp(-1.j*phi220)
    Ac221 = A221*np.exp(-1.j*phi221)
    Ac320 = A320*np.exp(-1.j*phi320)

    Amc220 = np.conj(Ac220)*mirror_ratio_list[0][0]*np.exp(1.j*mirror_ratio_list[0][1])
    Amc221 = np.conj(Ac221)*mirror_ratio_list[1][0]*np.exp(1.j*mirror_ratio_list[1][1])
    Amc320 = np.conj(Ac320)*mirror_ratio_list[2][0]*np.exp(1.j*mirror_ratio_list[2][1])

    Am220, phim220 = np.abs(Amc220), -np.angle(Amc220)
    Am221, phim221 = np.abs(Amc221), -np.angle(Amc221)
    Am320, phim320 = np.abs(Amc320), -np.angle(Amc320)

    modes = mode_list(['2.2.0', '2.-2.0', '2.2.1', '2.-2.1', '3.2.0', '3.-2.0'], Mf, af)
    A_phi_dict = {'2.2.0': dict(A = A220, phi = phi220),
                '2.-2.0': dict(A = Am220, phi = phim220),
                '2.2.1': dict(A = A221, phi = phi221),
                '2.-2.1': dict(A = Am221, phi = phim221),
                '3.2.0': dict(A = A320, phi = phi320),
                '3.-2.0': dict(A = Am320, phi = phim320)}

    t_arr = np.linspace(0, 120, 1000)
    h_arr = np.zeros(t_arr.shape, dtype = np.complex128)

    for i, mode in enumerate(modes):
        h_arr += clean_QNM(mode, t_arr, 
                            A_phi_dict[mode.string()]['A'], 
                            A_phi_dict[mode.string()]['phi'])
    
    h = waveform(t_arr, h_arr, t_peak = 0)
    return (Mf, af, iota, psi, h)




