from jaxqualin.qnmode import mode, mode_list
from jaxqualin.waveforms import waveform
from jaxqualin.fit import QNMFitVaryingStartingTime

import numpy as np

#TODO: make this less boilerplatey by reading the settings from the results json file
def test_fit_free(test_waveform_tuple, test_results_free):
    h, Mf, af = test_waveform_tuple
    t0_arr = np.linspace(0, 10, num = 11)

    qnm_fixed_list = []
    run_string_prefix = 'test'
    N_free = 2

    fitter = QNMFitVaryingStartingTime(
                                h, t0_arr, N_free = N_free,
                                qnm_fixed_list = qnm_fixed_list, load_pickle = False,
                                run_string_prefix = run_string_prefix)
    
    fitter.do_fits()
    results = fitter.result_full
    for key in test_results_free.keys():
        assert np.allclose(np.array(test_results_free[key]), results.results_dict[key])

def test_fit_fixed(test_waveform_tuple, test_results_fixed):
    h, Mf, af = test_waveform_tuple
    t0_arr = np.linspace(0, 10, num = 11)

    qnm_fixed_list = mode_list(['2.2.0', '2.2.1'], Mf, af)
    run_string_prefix = 'test'
    N_free = 0

    fitter = QNMFitVaryingStartingTime(
                                h, t0_arr, N_free = N_free,
                                qnm_fixed_list = qnm_fixed_list, load_pickle = False,
                                run_string_prefix = run_string_prefix)
    
    fitter.do_fits()
    results = fitter.result_full
    for key in test_results_fixed.keys():
        assert np.allclose(np.array(test_results_fixed[key]), results.results_dict[key])
    