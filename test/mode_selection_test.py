from jaxqualin.selection import ModeSearchAllFreeVaryingN
from jaxqualin.qnmode import qnms_to_string
import numpy as np

def test_mode_search_all_free_varying_n(test_waveform_tuple):
    h, Mf, af = test_waveform_tuple
    h.set_lm(2, 2)
    modesearcher = ModeSearchAllFreeVaryingN(h, Mf, af, [(2,2), (2,1), (3,3), (4,4)],
                                             N_list = [3], 
                                            t0_arr = np.linspace(10, 30, 201))
    modesearcher.do_mode_searches()
    modes_found = qnms_to_string(modesearcher.found_modes_final)
    assert set(modes_found) == set(['constant', '2.2.0']) 

