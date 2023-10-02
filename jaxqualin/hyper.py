from scipy.interpolate import LinearNDInterpolator
import numpy as np

def make_hyper_interpolator(data_dict, PN = True):

    A = data_dict['A']
    dA = data_dict['dA']
    phi = data_dict['phi']
    dphi = data_dict['dphi']
    
    if PN:
        eta = data_dict['eta']
        chi_p = data_dict['chi_p']
        chi_m = data_dict['chi_m']
        points_eta_chi_p_chi_m = np.array([eta, chi_p, chi_m]).T
        interpolator_A = LinearNDInterpolator(points_eta_chi_p_chi_m, A)
        interpolator_dA = LinearNDInterpolator(points_eta_chi_p_chi_m, dA)
        interpolator_phi = LinearNDInterpolator(points_eta_chi_p_chi_m, phi)
        interpolator_dphi = LinearNDInterpolator(points_eta_chi_p_chi_m, dphi)
    else:

        q = data_dict['q']
        chi_1 = data_dict['chi_1']
        chi_2 = data_dict['chi_2']
        points_q_chi_1_chi_2 = np.array([q, chi_1, chi_2]).T
        interpolator_A = LinearNDInterpolator(points_q_chi_1_chi_2, A)
        interpolator_dA = LinearNDInterpolator(points_q_chi_1_chi_2, dA)
        interpolator_phi = LinearNDInterpolator(points_q_chi_1_chi_2, phi)
        interpolator_dphi = LinearNDInterpolator(points_q_chi_1_chi_2, dphi)

    return {'A': interpolator_A,
            'dA': interpolator_dA,
            'phi': interpolator_phi,
            'dphi': interpolator_dphi}