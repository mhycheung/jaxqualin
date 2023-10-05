import numpy as np
import pandas as pd

from .waveforms import *
from .qnmode import *
from .selection import *

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
    best_flatness_checker = mode_searcher_vary_N.flatness_checkers[best_run_indx]
    fluc_least_indx_list = best_flatness_checker.fluc_least_indx_list
    flatness_length = best_flatness_checker.flatness_length
    quantile_range = best_flatness_checker.quantile_range
    med_min = best_flatness_checker.med_min
    weight_1 = best_flatness_checker.weight_1
    weight_2 = best_flatness_checker.weight_2
    flatness_tol = best_flatness_checker.flatness_tol
    t0_arr = best_flatness_checker.t0_arr
    found_modes = mode_searcher_vary_N.found_modes_final
    qnm_strings = qnms_to_string(found_modes)
    range_indx = mode_searcher_vary_N.flatness_checkers[best_run_indx].flatness_length
    for i, (start_indx, qnm_string) in enumerate(
            zip(fluc_least_indx_list, qnm_strings)):
        A_arr_pos = np.exp(found_modes[i].omegai * t_shift) * \
            mode_searcher_vary_N.fixed_fitters[best_run_indx].result_full.A_dict["A_" + qnm_string]
        A_arr = np.abs(A_arr_pos)
        A_med_pos = np.quantile(
            A_arr_pos[start_indx:start_indx + range_indx], 0.5)
        A_med = np.abs(A_med_pos)
        A_hi = np.quantile(A_arr[start_indx:start_indx + range_indx], 0.95)
        A_low = np.quantile(A_arr[start_indx:start_indx + range_indx], 0.05)
        phi_arr = mode_searcher_vary_N.fixed_fitters[best_run_indx].result_full.phi_dict["phi_" +
                                                                                         qnm_string] + found_modes[i].omegar * t_shift
        flat_start_indx = start_of_flat_region(flatness_length, A_arr, phi_arr,
                                               quantile_range=quantile_range,
                                               med_min=med_min,
                                               weight_1=weight_1,
                                               weight_2=weight_2,
                                               fluc_tol=flatness_tol)

        if np.isnan(flat_start_indx):
            t_flat_start = np.nan
        else:
            t_flat_start = t0_arr[flat_start_indx]
        if A_med_pos < 0:
            phi_arr -= np.pi
        phi_med = np.quantile(phi_arr[start_indx:start_indx + range_indx], 0.5)
        phi_hi = np.quantile(phi_arr[start_indx:start_indx + range_indx], 0.95)
        phi_low = np.quantile(
            phi_arr[start_indx:start_indx + range_indx], 0.05)
        kwargs.update(
            A_med=A_med,
            A_hi=A_hi,
            A_low=A_low,
            phi_med=phi_med,
            phi_hi=phi_hi,
            phi_low=phi_low,
            mode_string=qnm_string,
            M_rem=M_rem,
            chi_rem=chi_rem,
            t_flat_start=t_flat_start)
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
    df = pd.DataFrame(columns=["SXS_num", "M_rem", "chi_rem",
                               "chi_1_z", "chi_2_z", "q", "l", "m",
                               "retro", "mode_string",
                               "A_med", "A_hi", "A_low",
                               "phi_med", "phi_hi", "phi_low", "t_flat_start"])
    df = df.astype({"retro": bool})
    failed_list = []
    for SXS_num in SXS_num_list:
        try:
            mode_search_complete = ModeSearchAllFreeVaryingNSXSAllRelevant(
                str(SXS_num), load_pickle=True, **kwargs)
            mode_search_complete.do_all_searches()
            df = append_A_and_phis_all_lm(mode_search_complete, df)
        except BaseException:
            failed_list.append(SXS_num)
    file_path = os.path.join(DF_SAVE_PATH, f"{df_save_prefix}.csv")
    df.to_csv(file_path)
    print("failed runs: ", failed_list)


def create_data_frame_eff(
        eff_num_list,
        batch_runname,
        l=0,
        m=0,
        df_save_prefix="eff_default",
        **kwargs):
    df = pd.DataFrame(columns=["eff_num", "M_rem", "chi_rem",
                               "l", "m", "mode_string",
                               "A_med", "A_hi", "A_low", "phi_med",
                               "phi_hi", "phi_low", "t_flat_start"])

    for eff_num in eff_num_list:
        mode_searcher = read_json_eff_mode_search(
            eff_num, batch_runname, load_pickle=True, **kwargs)
        kwargs.update(eff_num=eff_num)
        df = append_A_and_phis(mode_searcher, df, l=l, m=m, **kwargs)

    file_path = os.path.join(DF_SAVE_PATH, f"{df_save_prefix}.csv")
    df.to_csv(file_path)

    return df


def get_result(
        run_string_prefix,
        t0_arr,
        qnm_fixed_list,
        N_free,
        nonconvergence_cut=False):

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


def linfunc(p, x):
    m, c = p
    return m * x + c


def linfunc2(p, x):
    c = p
    return 2 * x + c


def is_quadratic(row):
    mode_string = row['mode_string']
    if mode_string == 'constant':
        return False
    lmnx = str_to_lmnx(mode_string)
    if len(lmnx) == 2:
        return True
    else:
        return False


def is_overtone(row):
    mode_string = row['mode_string']
    if mode_string == 'constant':
        return False
    lmnx = str_to_lmnx(mode_string)
    for lmn in lmnx:
        l, m, n = tuple(lmn)
        if n > 0:
            return True
    return False


def is_fundamental(row):
    mode_string = row['mode_string']
    if mode_string == 'constant':
        return False
    lmnx = str_to_lmnx(mode_string)
    if len(lmnx) == 1 and lmnx[0][2] == 0:
        return True
    else:
        return False


def harm_type(row):
    l_harm = row['l']
    m_harm = row['m']
    mode_string = row['mode_string']
    if mode_string == 'constant':
        return 'constant'
    lmnx = str_to_lmnx(mode_string)
    l_sum, m_sum = lmnx_sum_lm(lmnx)
    if l_sum == l_harm and abs(m_sum) == m_harm:
        return 'basic'
    elif l_sum != l_harm and abs(m_sum) == m_harm:
        return 'mixing'
    else:
        return 'recoil'


def is_retro(row):
    retro = row['retro']
    if retro:
        retrofac = -1
    else:
        retrofac = 1
    mode_string = row['mode_string']
    if mode_string == 'constant':
        return False
    lmnx = str_to_lmnx(mode_string)
    for lmn in lmnx:
        if lmn[0] < 0:
            return True
    return False


def natural_m(row):
    mode_string = row['mode_string']
    if mode_string == 'constant':
        return -1
    lmnx = str_to_lmnx(mode_string)
    _, m_sum = lmnx_sum_lm(lmnx)
    return abs(m_sum)


def natural_l(row):
    mode_string = row['mode_string']
    if mode_string == 'constant':
        return -1
    lmnx = str_to_lmnx(mode_string)
    l_sum, _ = lmnx_sum_lm(lmnx)
    return l_sum


def sym_mass_ratio(row):
    q = row['q']
    return q / (1 + q)**2


def chi_p(row):
    q = row['q']
    chi_1 = row['chi_1_z']
    chi_2 = row['chi_2_z']
    return (q * chi_1 + chi_2) / (1 + q)


def chi_m(row):
    q = row['q']
    chi_1 = row['chi_1_z']
    chi_2 = row['chi_2_z']
    return (q * chi_1 - chi_2) / (1 + q)


def chi_rem_retro(row):
    chi_rem = row['chi_rem']
    retro = row['retro']
    if retro:
        return -chi_rem
    else:
        return chi_rem


def classify_modes(df):
    col_quad = df.apply(is_quadratic, axis=1)
    col_fund = df.apply(is_fundamental, axis=1)
    col_over = df.apply(is_overtone, axis=1)
    col_retro = df.apply(is_retro, axis=1)
    col_type = df.apply(harm_type, axis=1)
    col_nat_l = df.apply(natural_l, axis=1)
    col_nat_m = df.apply(natural_m, axis=1)
    col_eta = df.apply(sym_mass_ratio, axis=1)
    col_chi_p = df.apply(chi_p, axis=1)
    col_chi_m = df.apply(chi_m, axis=1)
    col_chi_rem_retro = df.apply(chi_rem_retro, axis=1)
    df = df.assign(eta=col_eta.values,
                   chi_p=col_chi_p.values,
                   chi_m=col_chi_m.values,
                   is_quadratic=col_quad.values,
                   is_fundamental=col_fund.values,
                   is_overtone=col_over.values,
                   is_retrograde=col_retro.values,
                   harm_type=col_type.values,
                   natural_l=col_nat_l.values,
                   natural_m=col_nat_m.values,
                   chi_rem_retro=col_chi_rem_retro.values)
    return df


def screen_mode(
        df,
        l,
        m,
        mode_string_pro,
        mode_string_retro,
        greater=True,
        A_cut=1):
    df_mode = df.loc[((df["l"] == l) & (df["m"] == m) & (df["mode_string"] == mode_string_pro) & (df["retro"] == False)) |
                     ((df["l"] == l) & (df["m"] == m) & (df["mode_string"] == mode_string_retro) & (df["retro"]))]
    if greater:
        df_screen = df_mode[df_mode['A_med'] > A_cut]
    else:
        df_screen = df_mode[df_mode['A_med'] <= A_cut]

    return df_screen


def df_get_mode(df, l, m, mode_string_pro, include_retro=False):
    mode_string_retro = qnm_string_l_reverse(mode_string_pro)
    if include_retro:
        df_mode = df.loc[((df["l"] == l) & (df["m"] == m) & (df["mode_string"] == mode_string_pro) & (df["retro"] == False)) |
                         ((df["l"] == l) & (df["m"] == m) & (df["mode_string"] == mode_string_retro) & (df["retro"]))]
    else:
        df_mode = df.loc[(df["l"] == l) & (df["m"] == m) &
                         (df["mode_string"] == mode_string_pro)]
    return df_mode


def df_get_mode_3D(
        df,
        l,
        m,
        mode_string_pro,
        include_retro=True,
        eta=True,
        PN_quantities=True,
        SXS_screen=[],
        chi_low=0.,
        chi_hi=1.):
    df_mode = df_get_mode(df, l, m, mode_string_pro,
                          include_retro=include_retro)
    df_mode = df_mode[~df_mode['SXS_num'].isin(SXS_screen)]
    df_mode = df_mode[df_mode['chi_rem'] >= chi_low]
    df_mode = df_mode[df_mode['chi_rem'] <= chi_hi]
    if PN_quantities:
        return df_mode[['SXS_num', 'chi_p', 'chi_m', 'eta', 'A_med']]
    elif eta:
        return df_mode[['SXS_num', 'chi_1_z', 'chi_2_z', 'eta', 'A_med']]
    else:
        return df_mode[['SXS_num', 'chi_1_z', 'chi_2_z', 'q', 'A_med']]


def df_get_mode_3D_full(
        df,
        l,
        m,
        mode_string_pro,
        include_retro=True,
        SXS_screen=[],
        chi_low=0.,
        chi_hi=1.):
    df_mode = df_get_mode(df, l, m, mode_string_pro,
                          include_retro=include_retro)
    df_mode = df_mode[~df_mode['SXS_num'].isin(SXS_screen)]
    df_mode = df_mode[df_mode['chi_rem'] >= chi_low]
    df_mode = df_mode[df_mode['chi_rem'] <= chi_hi]

    return df_mode[['SXS_num',
                    'chi_p',
                    'chi_m',
                    'eta',
                    'chi_1_z',
                    'chi_2_z',
                    'q',
                    'A_med',
                    'A_low',
                    'A_hi',
                    'phi_med',
                    'phi_med_adj',
                    'phi_low',
                    'phi_hi',
                    'chi_rem',
                    't_flat_start']]


def NP_quantities(x):
    chi_1, chi_2, q = tuple(x)
    eta = q / (1 + q)**2
    delta = np.sqrt(1 - 4 * eta)
    chi_p = (q * chi_1 + chi_2) / (1 + q)
    chi_m = (q * chi_1 - chi_2) / (1 + q)
    return q, eta, delta, chi_p, chi_m


def get_df_for_mode(df, l, m, mode_string_pro):
    mode_string_retro = qnm_string_l_reverse(mode_string_pro)
    df_mode = df.loc[((df["l"] == l) & (df["m"] == m) & (df["mode_string"] == mode_string_pro) & (df["retro"] == False)) |
                     ((df["l"] == l) & (df["m"] == m) & (df["mode_string"] == mode_string_retro) & (df["retro"]))]
    return df_mode


def get_df_for_coexisting_modes(df, mode_tuple_list):
    for i, mode_tuple in enumerate(mode_tuple_list):
        l, m, mode_string_pro = mode_tuple
        df_mode = get_df_for_mode(df, l, m, mode_string_pro)
        df_mode = df_mode.rename(
            columns={
                c: c +
                f'_{i+1}' for c in df.columns if c != 'SXS_num'})
        if i == 0:
            df_coexist = df_mode
        if i == 1:
            df_coexist = df_coexist.merge(df_mode, on="SXS_num", how="inner")
    return df_coexist


def mode_string_change_notation(mode_string):
    if mode_string == 'constant':
        return mode_string
    lmns = mode_string.split('x')
    strings = []
    for lmn in lmns:
        lmn_tuple = list(map(int, lmn.split('.')))
        l, m, n = lmn_tuple
        if l < 0:
            l = -l
            if m < 0:
                l = -l
                m = m
            else:
                m = -m
        elif m < 0:
            l = -l
            m = -m
        n = lmn_tuple[2]
        string = f"{l}.{m}.{n}"
        strings.append(string)
    return 'x'.join(strings)


def mode_string_retro_to_pro(mode_string):
    if mode_string == 'constant':
        return mode_string
    lmns = mode_string.split('x')
    strings = []
    for lmn in lmns:
        lmn_tuple = list(map(int, lmn.split('.')))
        l, m, n = lmn_tuple
        l = -l
        string = f"{l}.{m}.{n}"
        strings.append(string)
    return 'x'.join(strings)


def change_retro_to_pro(row):
    retro = row['retro']
    if retro:
        mode_string = mode_string_retro_to_pro(row['mode_string'])
    else:
        mode_string = row['mode_string']
    return mode_string


def df_change_retro_to_pro(df):
    col_mode_string = df.apply(change_retro_to_pro, axis=1)
    df = df.assign(mode_string=col_mode_string)
    return df
