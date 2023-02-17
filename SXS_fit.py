import QuasinormalMode
from QuasinormalMode import *
import Waveforms
from Waveforms import *
import Fit
from Fit import *
import utils
from utils import *
import plot
from plot import *
import ModeSelection
from ModeSelection import *
import sys

SXS_num = sys.argv[1]
mode_search_complete = ModeSearchAllFreeVaryingNSXSAllRelevant(SXS_num, N_list = [5, 6, 7, 8, 9, 10], 
                                                              load_pickle = False
                                                              )
mode_search_complete.do_all_searches()
Mf = mode_search_complete.relevant_lm_mode_searcher_varying_N[0].M
af = mode_search_complete.relevant_lm_mode_searcher_varying_N[0].a
predicted_qnm_list = []
plot_relevant_mode_search_full(mode_search_complete, predicted_qnm_list = predicted_qnm_list, postfix_string="run4")
