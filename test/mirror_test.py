from jaxqualin.fit import QNMFitVaryingStartingTime
from jaxqualin.qnmode import mode_list, S_mirror_fac_complex, A_pos_to_A_neg
from jaxqualin.utils import all_close_to

import numpy as np

def test_S_fac_mirror(S_mirror_fac_220_test):
    S_test = S_mirror_fac_complex(np.pi/3, 0.7, 2, 2, 0, psi = np.pi/2)
    assert np.isclose(S_test, S_mirror_fac_220_test)


def test_A_pos_to_A_neg(S_mirror_fac_220_test):
    A_pos = 1. + 0.5j
    A_neg = A_pos_to_A_neg(A_pos, np.pi/3, 0.7, 2, 2, 0, psi = np.pi/2)
    assert np.isclose(A_neg, S_mirror_fac_220_test*np.conj(A_pos))

def test_mirror_fit(mirror_waveform):
    Mf, af, iota, psi, h = mirror_waveform

    t0_arr = np.linspace(0, 5, num = 6) # array of starting times to fit for
                                        # t0 = 0 is the peak of the straisn
    qnm_fixed_list = mode_list(['2.2.0', '2.2.1', '3.2.0'],
                                    Mf, af) # list of QNMs with fixed frequencies in the fit model
    run_string_prefix = f"mirror_test_incl_mirror_clean" # prefix of pickle file for saving the results
    N_free = 0 # number of free modes to use

    # fitter object
    fitter = QNMFitVaryingStartingTime(
                                h, t0_arr, N_free = N_free,
                                qnm_fixed_list = qnm_fixed_list, load_pickle = False,
                                run_string_prefix = run_string_prefix,
                                include_mirror=True, iota = iota, psi = psi)
    
    fitter.do_fits()
    result = fitter.result_full

    assert all_close_to(result.A_dict['A_2.2.0'], 1.)
    assert all_close_to(result.A_dict['A_2.2.1'], 3.)
    assert all_close_to(result.A_dict['A_3.2.0'], -0.01)
    assert all_close_to(result.phi_dict['phi_2.2.0'], 0.)
    assert all_close_to(result.phi_dict['phi_2.2.1'], np.pi/2)
    assert all_close_to(result.phi_dict['phi_3.2.0'], 0.)
