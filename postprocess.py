import numpy as np
import pandas as pd
from Waveforms import *
from QuasinormalMode import *
from ModeSelection import *
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DF_SAVE_PATH = os.path.join(ROOT_PATH, "pickle/data_frame")


def append_A_and_phis(mode_searcher_vary_N, df, **kwargs):
    M_rem = mode_searcher_vary_N.M
    chi_rem = mode_searcher_vary_N.a
    best_run_indx = mode_searcher_vary_N.best_run_indx
    t_peak_lm = mode_searcher_vary_N.h.peaktime
    if hasattr(mode_searcher_vary_N.h, "t_peak"):
        t_peak_dom = mode_searcher_vary_N.h.t_peak
    else:
        t_peak_dom = t_peak_lm
    t_shift = t_peak_dom - t_peak_lm
    fluc_least_indx_list = mode_searcher_vary_N.flatness_checkers[
        best_run_indx].fluc_least_indx_list
    found_modes = mode_searcher_vary_N.found_modes_final
    qnm_strings = qnms_to_string(found_modes)
    range_indx = mode_searcher_vary_N.flatness_checkers[best_run_indx].flatness_length
    for i, (start_indx, qnm_string) in enumerate(zip(fluc_least_indx_list, qnm_strings)):
        A_arr_pos = np.exp(found_modes[i].omegai*t_shift) * \
            mode_searcher_vary_N.fixed_fitters[best_run_indx].result_full.A_dict["A_" + qnm_string]
        A_arr = np.abs(A_arr_pos)
        A_med_pos = np.quantile(
            A_arr_pos[start_indx:start_indx+range_indx], 0.5)
        A_med = np.abs(A_med_pos)
        A_hi = np.quantile(A_arr[start_indx:start_indx+range_indx], 0.95)
        A_low = np.quantile(A_arr[start_indx:start_indx+range_indx], 0.05)
        phi_arr = mode_searcher_vary_N.fixed_fitters[best_run_indx].result_full.phi_dict["phi_" +
                                                                                         qnm_string] + found_modes[i].omegar*t_shift
        if A_med_pos < 0:
            phi_arr -= np.pi
        phi_med = np.quantile(phi_arr[start_indx:start_indx+range_indx], 0.5)
        phi_hi = np.quantile(phi_arr[start_indx:start_indx+range_indx], 0.95)
        phi_low = np.quantile(phi_arr[start_indx:start_indx+range_indx], 0.05)
        kwargs.update(A_med=A_med, A_hi=A_hi, A_low=A_low,
                      phi_med=phi_med, phi_hi=phi_hi, phi_low=phi_low, mode_string=qnm_string,
                      M_rem=M_rem, chi_rem=chi_rem)
        df_row = pd.Series(kwargs)
        df_row_frame = df_row.to_frame().T
        if "retro" in df_row_frame:
            df_row_frame["retro"] = df_row_frame["retro"].astype(bool)
        df = pd.concat([df, df_row_frame])
    return df


def append_A_and_phis_all_lm(mode_search_complete, df, **kwargs):
    SXSnum = mode_search_complete.SXSnum
    q_chi_dict = get_chi_q_SXS(SXSnum)
    relevant_lm_list = mode_search_complete.relevant_lm_list
    retro = mode_search_complete.retro
    kwargs.update(**q_chi_dict, SXS_num=SXSnum, retro=retro)
    for i, lm in enumerate(relevant_lm_list):
        l, m = lm
        mode_searcher_vary_N = mode_search_complete.relevant_lm_mode_searcher_varying_N[
            i].mode_searcher_vary_N
        df = append_A_and_phis(mode_searcher_vary_N, df, l=l, m=m, **kwargs)
    return df


def create_data_frame(SXS_num_list, df_save_prefix="default", **kwargs):
    df = pd.DataFrame(columns=["SXS_num", "M_rem", "chi_rem",  "chi_1_z", "chi_2_z", "q", "l", "m", "retro", "mode_string",
                               "A_med", "A_hi", "A_low",
                               "phi_med", "phi_hi", "phi_low"])
    df = df.astype({"retro": bool})
    for SXS_num in SXS_num_list:
        mode_search_complete = ModeSearchAllFreeVaryingNSXSAllRelevant(str(SXS_num),
                                                                       load_pickle=True,
                                                                       **kwargs
                                                                       )
        mode_search_complete.do_all_searches()
        df = append_A_and_phis_all_lm(mode_search_complete, df)
    file_path = os.path.join(DF_SAVE_PATH, f"{df_save_prefix}.csv")
    df.to_csv(file_path)


def create_data_frame_eff(eff_num_list, batch_runname, l=0, m=0, df_save_prefix="eff_default", **kwargs):
    df = pd.DataFrame(columns=["eff_num", "M_rem", "chi_rem",  "l", "m", "mode_string",
                               "A_med", "A_hi", "A_low", "phi_med", "phi_hi", "phi_low"])

    for eff_num in eff_num_list:
        mode_searcher = read_json_eff_mode_search(
            eff_num, batch_runname, load_pickle=True, **kwargs)
        kwargs.update(eff_num=eff_num)
        df = append_A_and_phis(mode_searcher, df, l=l, m=m, **kwargs)

    file_path = os.path.join(DF_SAVE_PATH, f"{df_save_prefix}.csv")
    df.to_csv(file_path)

    return df

def get_result(run_string_prefix, t0_arr, qnm_fixed_list, N_free, nonconvergence_cut = False):
    
    N_fix = len(qnm_fixed_list)

    if N_fix > 0:
        _qnm_fixed_string_list = sorted(qnms_to_string(qnm_fixed_list))
        qnm_fixed_string_ordered = '_'.join(_qnm_fixed_string_list)
        run_string = f"{run_string_prefix}_N_{N_free}_fix_{qnm_fixed_string_ordered}_t0_{t0_arr[0]:.4f}_{t0_arr[-1]:.4f}_{len(t0_arr)}"
    else:
        qnm_fixed_string_ordered = ''
        run_string = f"{run_string_prefix}_N_{N_free}_t0_{t0_arr[0]:.4f}_{t0_arr[-1]:.4f}_{len(t0_arr)}"
    if nonconvergence_cut:
        run_string += "_nc"
    file_path = os.path.join(
        FIT_SAVE_PATH, f"{run_string}_result.pickle")

    with open(file_path, 'rb') as f:
        result = pickle.load(f)

    return result
