import sxs
import jax.numpy as jnp
import numpy as np
from bisect import bisect_left, bisect_right
from utils import *
from scipy.interpolate import griddata


class waveform:

    def __init__(self, fulltime, fullh, t_peak=None, t0=0, l=None, m=None):
        self.fulltime = fulltime
        self.fullh = fullh
        self.peakindx = self.argabsmax()
        if t_peak is None:
            self.peaktime = self.fulltime[self.peakindx]
        else:
            self.peaktime = t_peak
        self.time, self.hr, self.hi = self.postmerger(t0)
        self.h = self.hr + 1.j * self.hi
        self.l = l
        self.m = m

    def update_peaktime(self, t_peak):
        self.t_peak = t_peak

    def argabsmax(self, remove_num=500):
        return jnp.nanargmax(jnp.abs(self.fullh[remove_num:])) + remove_num

    def postmerger(self, t0):
        tstart = self.peaktime + t0
        startindx = bisect_left(self.fulltime, tstart)
        return self.fulltime[startindx:] - self.peaktime, jnp.real(
            self.fullh[startindx:]), jnp.imag(self.fullh[startindx:])


def get_waveform_SXS(SXSnum, l, m, res=0, N_ext=2):
    catalog = sxs.catalog.Catalog.load()
    waveformloadname = catalog.select(
        f"SXS:BBH:{SXSnum}/Lev./rhOverM")[-1 + res]
    metaloadname = catalog.select(
        f"SXS:BBH:{SXSnum}/Lev./metadata.json")[-1 + res]
    hs = sxs.load(waveformloadname, extrapolation_order=N_ext)
    metadata = sxs.load(metaloadname)
    Level = metaloadname[metaloadname.find("Lev") + 3]
    indx = hs.index(l, m)
    h = waveform(hs[:, indx].time, hs[:, indx].real +
                 1.j * hs[:, indx].imag, l=l, m=m)
    Mf = metadata['remnant_mass']
    a_arr = metadata['remnant_dimensionless_spin']
    af = np.linalg.norm(a_arr)
    if a_arr[2] >= 0:
        retro = False
    else:
        retro = True
    return h, Mf, af, Level, retro


def get_SXS_waveform_dict(SXSnum, res=0, N_ext=2):
    catalog = sxs.catalog.Catalog.load()
    waveformloadname = catalog.select(
        f"SXS:BBH:{SXSnum}/Lev./rhOverM")[-1 + res]
    metaloadname = catalog.select(
        f"SXS:BBH:{SXSnum}/Lev./metadata.json")[-1 + res]
    h = sxs.load(waveformloadname, extrapolation_order=N_ext)
    metadata = sxs.load(metaloadname)
    Level = metaloadname[metaloadname.find("Lev") + 3]
    modes = h.LM
    hdict = {}
    for mode in modes:
        l = mode[0]
        m = mode[1]
        indx = h.index(l, m)
        hdict[f"{l},{m}"] = [h[:, indx].time, h[:, indx].real, h[:, indx].imag]
    m1 = metadata['reference_mass1']
    m2 = metadata['reference_mass2']
    Mf = metadata['remnant_mass']
    a_arr = metadata['remnant_dimensionless_spin']
    af = np.linalg.norm(a_arr)
    if a_arr[2] >= 0:
        retro = False
    else:
        retro = True
    return Mf, af, Level, hdict, retro


def waveformabsmax(time, hr, hi, startcut=500):
    startindx = bisect_left(time, startcut)
    maxindx = np.argmax(hr[startindx:]**2 + hi[startindx:]**2) + startindx
    if isinstance(hr, np.ndarray):
        return maxindx, time[maxindx], np.sqrt(hr[maxindx]**2 + hi[maxindx]**2)
    return maxindx, time[maxindx], np.array(
        np.sqrt(hr[maxindx]**2 + hi[maxindx]**2))[0]


def getdommodes(hdict, tol=1 / 40, prec=False, includem0=False):
    keyabsmaxlist = []

    for key in hdict:
        time, hr, hi = hdict[key]
        _, _, habsmax = waveformabsmax(time, hr, hi)
        keyabsmaxlist.append([key, habsmax])

    keyabsmaxsorted = sorti(keyabsmaxlist, 1)
    maxamp = keyabsmaxsorted[0][1]
    dommodes = []
    for keyabsmax in keyabsmaxsorted:
        keystring = keyabsmax[0]
        lm = [int(string) for string in keystring.split(",")]
        if lm[1] == 0 and not includem0:
            continue
        if (lm[1] < 0 and not prec) or keyabsmax[1] < maxamp * tol:
            continue
        dommodes.append(lm)
    return dommodes


def get_relevant_lm_waveforms_SXS(
        SXSnum,
        tol=1 / 40,
        prec=False,
        includem0=False,
        res=0,
        N_ext=2):
    Mf, af, Level, hdict, retro = get_SXS_waveform_dict(SXSnum, res=res, N_ext=N_ext)
    relevant_lm_list = getdommodes(
        hdict, tol=1 / 40, prec=False, includem0=False)
    waveform_dict = {}
    for lm in relevant_lm_list:
        l = lm[0]
        m = lm[1]
        h_time, h_r, h_i = tuple(hdict[f"{l},{m}"])
        h = waveform(h_time, h_r + 1.j * h_i, l=l, m=m)
        waveform_dict[f"{l}.{m}"] = h
    return waveform_dict, retro


def lm_string_list_to_tuple(lm_string_list):
    tuple_list = []
    for lm_string in lm_string_list:
        tuple_list.append(tuple(map(int, lm_string.split('.'))))
    return tuple_list


def relevant_modes_dict_to_lm_tuple(relevant_modes_dict):
    lm_string_list = list(relevant_modes_dict.keys())
    return lm_string_list_to_tuple(lm_string_list)

def get_chi_q_SXS(SXSnum, res = 0):
    catalog = sxs.catalog.Catalog.load()
    metaloadname = catalog.select(
        f"SXS:BBH:{SXSnum}/Lev./metadata.json")[-1 + res]
    metadata = sxs.load(metaloadname)
    q = metadata['reference_mass_ratio']
    chi_1_z = metadata['reference_dimensionless_spin1'][2]
    chi_2_z = metadata['reference_dimensionless_spin2'][2]
    return {'q' : q, "chi_1_z" : chi_1_z, "chi_2_z" : chi_2_z}

def waveform_toy_clean(A_list, phi_list, qnm_list, t_arr, l = 2, m = 2):
    if not len(A_list) == len(phi_list) == len(qnm_list):
        raise ValueError
    N = len(A_list)
    fullh = np.zeros(len(t_arr), dtype = np.complex128)
    for i in range(N):
        omegar = qnm_list[i].omegar
        omegai = qnm_list[i].omegai
        fullh += A_list[i] * np.exp(-1.j * ((omegar + 1.j * omegai) * t_arr + phi_list[i]))
    return fullh

def get_waveform_toy_clean(A_list, phi_list, qnm_list, t_arr, l = 2, m = 2):
    fullh = waveform_toy_clean(A_list, phi_list, qnm_list, t_arr, l = l, m = m)
    h = waveform(t_arr, fullh, t_peak=0, t0=0, l=l, m=m)
    return h

def get_waveform_toy_bump(A_list, phi_list, qnm_list, t_arr, A_bump, sig_bump, t0_bump, l = 2, m = 2):
    bump = A_bump*np.exp(-(t_arr - t0_bump)**2/(2*sig_bump**2))
    fullh = waveform_toy_clean(A_list, phi_list, qnm_list, t_arr, l = l, m = m)
    fullh += bump
    h = waveform(t_arr, fullh, t_peak=0, t0=0, l=l, m=m)
    return h

def get_waveform_toy_stretch(A_list, phi_list, qnm_list, t_arr, A_stretch, sig_stretch, l = 2, m = 2):
    fullh_clean = waveform_toy_clean(A_list, phi_list, qnm_list, t_arr, l = l, m = m)
    t_stretch = t_arr*(A_stretch*np.exp(-t_arr/sig_stretch)+1)
    fullh = griddata(t_stretch, fullh_clean, t_arr)
    h = waveform(t_arr, fullh, t_peak=0, t0=0, l=l, m=m)
    return h

def waveform_toy_EOB_model(A_list, phi_list, qnm_list, t_arr, t_match, c_list, d_list, l = 2, m = 2):
    if not len(A_list) == len(phi_list) == len(qnm_list):
        raise ValueError
    c1, c3 = tuple(c_list)
    d1, d2, d3 = tuple(d_list)
    N = len(A_list)
    fullh = np.zeros(len(t_arr), dtype = np.complex128)
    for i in range(N):
        omegar = qnm_list[i].omegar
        omegai = qnm_list[i].omegai
        A_prime = -c1*(np.tanh(c3*(t_arr - t_match))-1)/2 + A_list[i]
        phi_prime = phi_list[i] - d1*np.log((1+d2*np.exp(-d3*(t_arr-t_match)))/(1+d2)) + d1*np.log(1/(1+d2))
        fullh += A_prime * np.exp(-1.j*phi_prime) * np.exp(-1.j * (omegar + 1.j * omegai) * t_arr)
    return fullh

def get_waveform_toy_EOB_model(A_list, phi_list, qnm_list, t_arr, t_match, c_list, d_list, l = 2, m = 2):
    fullh = waveform_toy_EOB_model(A_list, phi_list, qnm_list, t_arr, t_match, c_list, d_list, l = l, m = m)
    h = waveform(t_arr, fullh, t_peak=0, t0=0, l=l, m=m)
    return h

def get_waveform_toy_EOB_model_no_fund(A_list, phi_list, qnm_list, t_arr, t_match, c_list, d_list, l = 2, m = 2):
    fullh = waveform_toy_EOB_model(A_list, phi_list, qnm_list, t_arr, t_match, c_list, d_list, l = l, m = m)
    cleanh = waveform_toy_clean(A_list, phi_list, qnm_list, t_arr, l = l, m = m)
    h = waveform(t_arr, fullh-cleanh, t_peak=0, t0=0, l=l, m=m)
    return h

def waveform_toy_no_exp(A_list, phi_list, qnm_list, t_arr, t_match, c_list, d_list, l = 2, m = 2):
    if not len(A_list) == len(phi_list) == len(qnm_list):
        raise ValueError
    c1, c3 = tuple(c_list)
    d1, d2, d3 = tuple(d_list)
    N = len(A_list)
    fullh = np.zeros(len(t_arr), dtype = np.complex128)
    for i in range(N):
        omegar = qnm_list[i].omegar
        omegai = qnm_list[i].omegai
        A_prime = -c1/(t_arr - t_match - c3)**2 + A_list[i]
        phi_prime = phi_list[i] - d1*np.log((1+d2*np.exp(-d3*(t_arr-t_match)))/(1+d2)) + d1*np.log(1/(1+d2))
        fullh += A_prime * np.exp(-1.j*phi_prime) * np.exp(-1.j * (omegar + 1.j * omegai) * t_arr)
    return fullh

def get_waveform_toy_no_exp(A_list, phi_list, qnm_list, t_arr, t_match, c_list, d_list, l = 2, m = 2):
    fullh = waveform_toy_no_exp(A_list, phi_list, qnm_list, t_arr, t_match, c_list, d_list, l = l, m = m)
    h = waveform(t_arr, fullh, t_peak=0, t0=0, l=l, m=m)
    return h