from jaxqualin.qnmode import mode, mode_list
from jaxqualin.waveforms import waveform
from jaxqualin.fit import QNMFitVaryingStartingTime

from itertools import permutations
import re
import numpy as np

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
    free_key_pattern = re.compile(r"^(?P<base>.+_free_)(?P<idx>\d+)$")
    expected = test_results_free
    actual = results.results_dict

    expected_free_keys = {k for k in expected if free_key_pattern.match(k)}
    actual_free_keys = {k for k in actual if free_key_pattern.match(k)}

    # Non-free keys should still match directly.
    for key in expected.keys() - expected_free_keys:
        assert np.allclose(np.array(expected[key]), actual[key])

    expected_bases = sorted({free_key_pattern.match(k).group("base") for k in expected_free_keys})
    actual_bases = sorted({free_key_pattern.match(k).group("base") for k in actual_free_keys})
    assert expected_bases == actual_bases

    expected_indices = sorted({int(free_key_pattern.match(k).group("idx")) for k in expected_free_keys})
    actual_indices = sorted({int(free_key_pattern.match(k).group("idx")) for k in actual_free_keys})
    assert expected_indices == actual_indices

    def matches_for_permutation(perm):
        for base in expected_bases:
            for pos, expected_idx in enumerate(expected_indices):
                actual_idx = actual_indices[perm[pos]]
                expected_key = f"{base}{expected_idx}"
                actual_key = f"{base}{actual_idx}"
                if not np.allclose(np.array(expected[expected_key]), actual[actual_key]):
                    return False
        return True

    assert any(matches_for_permutation(perm) for perm in permutations(range(len(expected_indices))))

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
    