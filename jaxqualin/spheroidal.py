import pandas as pd
import numpy as np
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_PATH, "data/")

def get_df_ssm():

    header = ['m', 'l', 'l_p', 'n_p', 'p1r',  'p2r',  'p3r',  'p4r',  'p1i', 
            'p2i',  'p3i',  'p4i',  'sr',  'si',  'mr',  'mi']

    with open(DATA_PATH + "swsh_fits.dat", 'r') as f:
        df_ssm = pd.read_csv(f, header = None, sep = "\s+")

    df_ssm.columns = header

    return df_ssm

df_ssm = get_df_ssm()

def ssmc(l, m, l_p, n_p, a):
    row = df_ssm[(df_ssm['m'] == m) & (df_ssm['l'] == l) & (df_ssm['l_p'] == l_p) & (df_ssm['n_p'] == n_p)]
    p1r = row['p1r'].values[0]
    p2r = row['p2r'].values[0]
    p3r = row['p3r'].values[0]
    p4r = row['p4r'].values[0]
    p1i = row['p1i'].values[0]
    p2i = row['p2i'].values[0]
    p3i = row['p3i'].values[0]
    p4i = row['p4i'].values[0]
    if l == l_p:
        delta_llp = 1
    else:
        delta_llp = 0
    mu_re = delta_llp + p1r * a**p2r + p3r * a**p4r
    mu_im = p1i * a**p2i + p3i * a**p4i
    return mu_re + 1.j*mu_im

def ssmc_ratio(l, m, l_p, n_p, a):
    mu_mix = ssmc(l, m, l_p, n_p, a)
    mu_org = ssmc(l_p, m, l_p, n_p, a)
    return mu_mix / mu_org

def mode_in_df(df, SXS_num, l, m, mode_string):
    row = df[(df['SXS_num'] == SXS_num) & (df['l'] == l) & (df['m'] == m) & (df['mode_string'] == mode_string)]
    return len(row) > 0

def mode_in_df_num(df_num, l, m, mode_string):
    row = df_num[(df_num['l'] == l) & (df_num['m'] == m) & (df_num['mode_string'] == mode_string)]
    return len(row) > 0

def mixing_check_SXS_nums(df, l, m, l_p, n_p):
    mix_check_num = []
    SXS_num_arr = df.SXS_num.unique()
    for SXS_num in SXS_num_arr:
        df_num = df[df['SXS_num'] == SXS_num]
        retro = df_num['retro'].values[0]
        if retro:
            m_mode = -m
        else:
            m_mode = m
        mix_exist = mode_in_df_num(df_num, l, m, f'{l_p}.{m_mode}.{n_p}')
        org_exist = mode_in_df_num(df_num, l_p, m, f'{l_p}.{m_mode}.{n_p}')
        if mix_exist and org_exist:
            mix_check_num.append(SXS_num)
    return mix_check_num

# primed mode mixing into un_primed harmonic
def give_mixing_ratio(df_num, l, m, l_p, n_p):
    retro = df_num['retro'].values[0]
    if retro:
        m_mode = -m
    else:
        m_mode = m
    row_mix = df_num[(df_num['l'] == l) & (df_num['m'] == m) & (df_num['mode_string'] == f'{l_p}.{m_mode}.{n_p}')]
    row_org = df_num[(df_num['l'] == l_p) & (df_num['m'] == m) & (df_num['mode_string'] == f'{l_p}.{m_mode}.{n_p}')]
    A_mix = row_mix['A_med'].values[0]
    A_org = row_org['A_med'].values[0]
    phi_mix = row_mix['phi_med'].values[0]%(2*np.pi)
    phi_org = row_org['phi_med'].values[0]%(2*np.pi)
    return A_mix / A_org, phi_mix - phi_org