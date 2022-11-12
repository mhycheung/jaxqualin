import numpy as np
import pandas as pd
from Waveforms import *
from QuasinormalMode import *
from ModeSelection import *
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DF_SAVE_PATH = os.path.join(ROOT_PATH, "pickle/data_frame")

def append_A_and_phis(mode_searcher_vary_N, df, **kwargs):
    best_run_indx = mode_searcher_vary_N.best_run_indx
    best_run_indx = mode_searcher_vary_N.best_run_indx
    fluc_least_indx_list = mode_searcher_vary_N.flatness_checkers[best_run_indx].fluc_least_indx_list
    qnm_strings = qnms_to_string(mode_searcher_vary_N.found_modes_final)
    range_indx = mode_searcher_vary_N.flatness_checkers[best_run_indx].flatness_length
    for start_indx, qnm_string in zip(fluc_least_indx_list, qnm_strings):
        A_arr = np.abs(mode_searcher_vary_N.fixed_fitters[best_run_indx].result_full.A_dict["A_" + qnm_string])
        A_med = np.quantile(A_arr[start_indx:start_indx+range_indx], 0.5)
        A_hi = np.quantile(A_arr[start_indx:start_indx+range_indx], 0.95)
        A_low = np.quantile(A_arr[start_indx:start_indx+range_indx], 0.05)
        phi_arr = mode_searcher_vary_N.fixed_fitters[best_run_indx].result_full.phi_dict["phi_" + qnm_string]
        phi_med = np.quantile(phi_arr[start_indx:start_indx+range_indx], 0.5)
        phi_hi = np.quantile(phi_arr[start_indx:start_indx+range_indx], 0.95)
        phi_low = np.quantile(phi_arr[start_indx:start_indx+range_indx], 0.05)
        kwargs.update(A_med = A_med, A_hi = A_hi, A_low = A_low,
                               phi_med = phi_med, phi_hi = phi_hi, phi_low = phi_low, mode_string = qnm_string)
        df_row = pd.Series(kwargs)
        df_row_frame = df_row.to_frame().T
        df_row_frame["retro"] = df_row_frame["retro"].astype(bool)
        df = pd.concat([df, df_row_frame])
    return df
        
def append_A_and_phis_all_lm(mode_search_complete, df, **kwargs):
    SXSnum = mode_search_complete.SXSnum
    q_chi_dict = get_chi_q_SXS(SXSnum)
    relevant_lm_list = mode_search_complete.relevant_lm_list
    retro = mode_search_complete.retro
    kwargs.update(**q_chi_dict, SXS_num = SXSnum, retro = retro)
    for i, lm in enumerate(relevant_lm_list):
        l, m = lm
        mode_searcher_vary_N = mode_search_complete.relevant_lm_mode_searcher_varying_N[i].mode_searcher_vary_N
        df = append_A_and_phis(mode_searcher_vary_N, df, l = l, m = m, **kwargs)
    return df

def create_data_frame(SXS_num_list, N_list, df_save_prefix = "default"):
    df = pd.DataFrame(columns = ["SXS_num", "chi_1_z", "chi_2_z", "q", "l", "m", "retro", "mode_string",
                             "A_med", "A_hi", "A_low",
              "phi_med", "phi_hi", "phi_low"])
    df = df.astype({"retro": bool})
    for SXS_num in SXS_num_list:
        mode_search_complete = ModeSearchAllFreeVaryingNSXSAllRelevant(str(SXS_num), N_list = N_list, 
                                                              load_pickle = True
                                                              )
        mode_search_complete.do_all_searches()
        df = append_A_and_phis_all_lm(mode_search_complete, df)
    file_path = os.path.join(DF_SAVE_PATH, f"{df_save_prefix}.csv")
    df.to_csv(file_path)