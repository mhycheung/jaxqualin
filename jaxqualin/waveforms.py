from __future__ import annotations

import jaxlib
from typing import Dict, List, Tuple, Union, Optional
import os
import json
import h5py
import sxs
import jax.numpy as jnp
import numpy as np
from bisect import bisect_left, bisect_right

from .qnmode import long_str_to_qnms
from .utils import sorti

import jax

from scipy.interpolate import griddata
from scipy.stats import loguniform, uniform

from numpy.random import default_rng
from scipy.optimize import minimize
rng = default_rng(seed=1234)


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
ArrayImpl = jax.Array

DEFAULT_REMOVE_NUM = 500
DEFAULT_T_END = 120


class waveform:
    """
    A class representing a waveform containing a ringdown phase to be fitted.

    Attributes:
        fulltime: The full time array of the waveform. 
        fullh: The full complex waveform. 
        peaktime: The time of peak strain `jnp.abs(h)` of the waveform. 
        peakindx: The array index of the time of peak strain of the
            waveform.
        t_peak: The time of peak strain of the waveform. This can be defined
            by the user and overrides `peaktime`.
        time: The time array of the waveform after the peak, starting at
            `t_peak` and time shifted such that `t_peak = 0`.
        hr: The real part of the waveform after the peak.
        hi: The imaginary part of the waveform after the peak. 
        h: The complex waveform after the peak.
        l: The spherical harmonic mode number l of the waveform. 
        m: The spherical harmonic mode number m of the waveform.

    Methods:
        update_peaktime: Sets `t_peak` to override the peak time. 
        argabsmax: Returns the array index of the time of peak strain of the 
            waveform.
        postmerger: Returns the time, real part, and imaginary part of the
            waveform after the peak. 
        set_lm: Sets the spherical harmonic mode numbers l and m of the 
            waveform.

    """

    fulltime: np.ndarray
    fullh: np.ndarray
    peaktime: float
    peakindx: int
    t_peak: int
    time: np.ndarray
    hr: jnp.ndarray
    hi: jnp.ndarray
    h: jnp.ndarray
    l: int
    m: int

    def __init__(
            self,
            fulltime: np.ndarray,
            fullh: np.ndarray,
            t_peak: Optional[float] = None,
            t_start: float = 0.,
            t_end: float = np.inf,
            l: int = None,
            m: int = None,
            remove_num: int = DEFAULT_REMOVE_NUM) -> None:
        """
        Initialize a waveform.

        Parameters:
            fulltime: The full time array of the waveform.
            fullh: The full complex waveform.
            t_peak: The time of peak strain of the waveform. This can be
                defined by the user and overrides `peaktime`.
            t_start: The time after the peak to start the waveform.
            t_end: The time after the peak to end the waveform.
            l: The spherical harmonic mode number l of the waveform.
            m: The spherical harmonic mode number m of the waveform.
            remove_num: The number of points to remove from the beginning of
                the waveform to avoid numerical artifacts.
        """
        self.fulltime = fulltime
        self.fullh = fullh
        self.peakindx = self.argabsmax(remove_num=remove_num)
        if t_peak is None:
            self.peaktime = self.fulltime[self.peakindx]
        else:
            self.peaktime = t_peak
        self.time, self.hr, self.hi = self.postmerger(t_start, t_end)
        self.h = self.hr + 1.j * self.hi
        self.l = l
        self.m = m

    def update_peaktime(self, t_peak: float) -> None:
        """
        Override the peak time of the waveform.

        Parameters:
            t_peak: The user-defined peak time of the waveform.
        """
        self.t_peak = t_peak

    def argabsmax(self, remove_num: int = DEFAULT_REMOVE_NUM) -> int:
        """
        Returns the array index of the time of peak strain of the waveform.

        Parameters:
            remove_num: The number of points to remove from the beginning of the
                waveform to avoid numerical artifacts.

        Returns:
            The array index of the time of peak strain of the waveform.

        """
        return jnp.nanargmax(jnp.abs(self.fullh[remove_num:])) + remove_num

    def postmerger(self,
                   t_start: float,
                   t_end: float = np.inf) -> Tuple[np.ndarray,
                                                   jnp.ndarray,
                                                   jnp.ndarray]:
        """
        Returns the time, real part, and imaginary part of the waveform after
        the peak.

        Parameters:
            t_start: The time after the peak to start the waveform. t_end: The
                time after the peak to end the waveform.

        Returns:
            The time, real part, and imaginary part of the waveform after the
            peak.
        """
        tstart = self.peaktime + t_start
        tend = self.peaktime + t_end
        startindx = bisect_left(self.fulltime, tstart)
        endindx = bisect_left(self.fulltime, tend)
        # Use numpy for slicing to avoid JAX tracing overhead for each unique slice shape
        h_slice = np.asarray(self.fullh[startindx:endindx])
        return self.fulltime[startindx:endindx] - self.peaktime, np.real(h_slice), np.imag(h_slice)

    def set_lm(self, l: int, m: int) -> None:
        """
        Sets the spherical harmonic mode numbers l and m of the waveform.

        Parameters:
            l: The spherical harmonic mode number l of the waveform.
            m: The spherical harmonic mode number m of the waveform.

        """
        self.l = l
        self.m = m

def get_SXS_lev(SXS_num: str, res: int = 0, download: Optional[bool] = None) -> Tuple:
    sim = sxs.load(f"SXS:BBH:{SXS_num}", download = download)
    lev_numbers = sim.lev_numbers
    lev = lev_numbers[-1 + res]
    return sim, lev

def get_waveform_SXS(SXS_num: str, l: int, m: int, res: int = 0, N_ext: int = 2, t_end: float = DEFAULT_T_END, download: Optional[bool] = None) -> Tuple[waveform, float, float, int]:
    sim, lev = get_SXS_lev(SXS_num, res, download)
    hs = sxs.load(f"SXS:BBH:{SXS_num}/Lev{lev}", extrapolation = f"N{N_ext}").h
    metadata = sim.metadata
    indx = hs.index(l, m)
    h = waveform(hs[:, indx].time, hs[:, indx].real +
                 1.j * hs[:, indx].imag, l=l, m=m, t_end=t_end)
    Mf = metadata['remnant_mass']
    a_arr = metadata['remnant_dimensionless_spin']
    af = np.linalg.norm(a_arr) * np.sign(a_arr[2])
    return h, Mf, af, lev


def get_M_a_SXS(SXS_num: str, res: int = 0, download: Optional[bool] = None) -> Tuple[float, float]:
    sim, lev = get_SXS_lev(SXS_num, res, download)
    metadata = sim.metadata
    Mf = metadata['remnant_mass']
    a_arr = metadata['remnant_dimensionless_spin']
    af = np.linalg.norm(a_arr)
    return Mf, af


def get_SXS_waveform_dict(SXS_num: str, res: int = 0, N_ext: int = 2, download: Optional[bool] = None) -> Tuple[float, float, int, Dict]:
    sim, lev = get_SXS_lev(SXS_num, res, download)
    h = sxs.load(f"SXS:BBH:{SXS_num}/Lev{lev}", extrapolation = f"N{N_ext}").h
    metadata = sim.metadata
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
    af = np.linalg.norm(a_arr) * np.sign(a_arr[2])
    return Mf, af, lev, hdict


def waveformabsmax(time, hr, hi, remove_num=500):
    startindx = bisect_left(time, remove_num)
    maxindx = np.argmax(hr[startindx:]**2 + hi[startindx:]**2) + startindx
    if isinstance(hr, np.ndarray):
        return maxindx, time[maxindx], np.sqrt(hr[maxindx]**2 + hi[maxindx]**2)
    return maxindx, time[maxindx], np.array(
        np.sqrt(hr[maxindx]**2 + hi[maxindx]**2))[0]


def getdommodes(hdict, tol=1 / 50, tol_force=1 / 1000, prec=False, includem0=True,
                force_include_lm=[[2, 2], [3, 3], [4, 4], [5, 5], [6, 6]]):
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
        if lm in force_include_lm:
            tol_actual = tol_force
        else:
            tol_actual = tol
        if lm[1] == 0 and not includem0:
            continue
        if (lm[1] < 0 and not prec) or keyabsmax[1] < maxamp * tol_actual:
            continue
        dommodes.append(lm)

    return dommodes


def get_relevant_lm_waveforms_SXS(
        SXS_num,
        tol=1 / 50,
        tol_force=1 / 1000,
        force_early_sim=False,
        prec=False,
        includem0=True,
        t_end=DEFAULT_T_END,
        res=0,
        N_ext=2,
        CCE=False):
    if CCE:
        raise NotImplementedError
    else:
        Mf, af, Level, hdict = get_SXS_waveform_dict(
            SXS_num, res=res, N_ext=N_ext)
    if (int(SXS_num) < 305) and (not force_early_sim) and (not CCE):
        tol_force = tol
    relevant_lm_list = getdommodes(
        hdict, tol=tol, prec=prec, includem0=includem0, tol_force=tol_force)
    waveform_dict = {}
    for lm in relevant_lm_list:
        l = lm[0]
        m = lm[1]
        h_time, h_r, h_i = tuple(hdict[f"{l},{m}"])
        h = waveform(h_time, h_r + 1.j * h_i, l=l, m=m, t_end=t_end)
        waveform_dict[f"{l}.{m}"] = h
    return waveform_dict


def lm_string_list_to_tuple(lm_string_list):
    tuple_list = []
    for lm_string in lm_string_list:
        tuple_list.append(tuple(map(int, lm_string.split('.'))))
    return tuple_list


def relevant_modes_dict_to_lm_tuple(relevant_modes_dict):
    lm_string_list = list(relevant_modes_dict.keys())
    return lm_string_list_to_tuple(lm_string_list)


def get_chi_q_SXS(SXS_num, res=0, download = None):
    sim, lev = get_SXS_lev(SXS_num, res, download)
    metadata = sim.metadata
    q = metadata['reference_mass_ratio']
    chi_1_z = metadata['reference_dimensionless_spin1'][2]
    chi_2_z = metadata['reference_dimensionless_spin2'][2]
    return {'q': q, "chi_1_z": chi_1_z, "chi_2_z": chi_2_z}


def waveform_toy_clean(A_list, phi_list, qnm_list, t_arr, l=2, m=2):
    if not len(A_list) == len(phi_list) == len(qnm_list):
        raise ValueError
    N = len(A_list)
    fullh = np.zeros(len(t_arr), dtype=np.complex128)
    for i in range(N):
        omegar = qnm_list[i].omegar
        omegai = qnm_list[i].omegai
        fullh += A_list[i] * \
            np.exp(-1.j * ((omegar + 1.j * omegai) * t_arr + phi_list[i]))
    return fullh


def get_waveform_toy_clean(A_list, phi_list, qnm_list, t_arr, l=2, m=2):
    fullh = waveform_toy_clean(A_list, phi_list, qnm_list, t_arr, l=l, m=m)
    h = waveform(t_arr, fullh, t_peak=0, t_start=0, l=l, m=m)
    return h


def get_waveform_toy_bump(
        A_list,
        phi_list,
        qnm_list,
        t_arr,
        A_bump,
        sig_bump,
        t0_bump,
        l=2,
        m=2):
    bump = A_bump * np.exp(-(t_arr - t0_bump)**2 / (2 * sig_bump**2))
    fullh = waveform_toy_clean(A_list, phi_list, qnm_list, t_arr, l=l, m=m)
    fullh += bump
    h = waveform(t_arr, fullh, t_peak=0, t_start=0, l=l, m=m)
    return h


def get_waveform_toy_stretch(
        A_list,
        phi_list,
        qnm_list,
        t_arr,
        A_stretch,
        sig_stretch,
        l=2,
        m=2):
    fullh_clean = waveform_toy_clean(
        A_list, phi_list, qnm_list, t_arr, l=l, m=m)
    t_stretch = t_arr * (A_stretch * np.exp(-t_arr / sig_stretch) + 1)
    fullh = griddata(t_stretch, fullh_clean, t_arr)
    h = waveform(t_arr, fullh, t_peak=0, t_start=0, l=l, m=m)
    return h


def waveform_toy_EOB_model(
        A_list,
        phi_list,
        qnm_list,
        t_arr,
        t_match,
        c_list,
        d_list,
        l=2,
        m=2):
    if not len(A_list) == len(phi_list) == len(qnm_list):
        raise ValueError
    c1, c3 = tuple(c_list)
    d1, d2, d3 = tuple(d_list)
    N = len(A_list)
    fullh = np.zeros(len(t_arr), dtype=np.complex128)
    for i in range(N):
        omegar = qnm_list[i].omegar
        omegai = qnm_list[i].omegai
        A_prime = -c1 * (np.tanh(c3 * (t_arr - t_match)) - 1) / 2 + A_list[i]
        phi_prime = phi_list[i] - d1 * np.log(
            (1 + d2 * np.exp(-d3 * (t_arr - t_match))) / (1 + d2)) + d1 * np.log(1 / (1 + d2))
        fullh += A_prime * np.exp(-1.j * phi_prime) * \
            np.exp(-1.j * (omegar + 1.j * omegai) * t_arr)
    return fullh


def get_waveform_toy_EOB_model(
        A_list,
        phi_list,
        qnm_list,
        t_arr,
        t_match,
        c_list,
        d_list,
        l=2,
        m=2):
    fullh = waveform_toy_EOB_model(
        A_list,
        phi_list,
        qnm_list,
        t_arr,
        t_match,
        c_list,
        d_list,
        l=l,
        m=m)
    h = waveform(t_arr, fullh, t_peak=0, t_start=0, l=l, m=m)
    return h


def get_waveform_toy_EOB_model_no_fund(
        A_list,
        phi_list,
        qnm_list,
        t_arr,
        t_match,
        c_list,
        d_list,
        l=2,
        m=2):
    fullh = waveform_toy_EOB_model(
        A_list,
        phi_list,
        qnm_list,
        t_arr,
        t_match,
        c_list,
        d_list,
        l=l,
        m=m)
    cleanh = waveform_toy_clean(A_list, phi_list, qnm_list, t_arr, l=l, m=m)
    h = waveform(t_arr, fullh - cleanh, t_peak=0, t_start=0, l=l, m=m)
    return h


def waveform_toy_no_exp(
        A_list,
        phi_list,
        qnm_list,
        t_arr,
        t_match,
        c_list,
        d_list,
        l=2,
        m=2):
    if not len(A_list) == len(phi_list) == len(qnm_list):
        raise ValueError
    c1, c3 = tuple(c_list)
    d1, d2, d3 = tuple(d_list)
    N = len(A_list)
    fullh = np.zeros(len(t_arr), dtype=np.complex128)
    for i in range(N):
        omegar = qnm_list[i].omegar
        omegai = qnm_list[i].omegai
        A_prime = -c1 / (t_arr - t_match - c3)**2 + A_list[i]
        phi_prime = phi_list[i] - d1 * np.log(
            (1 + d2 * np.exp(-d3 * (t_arr - t_match))) / (1 + d2)) + d1 * np.log(1 / (1 + d2))
        fullh += A_prime * np.exp(-1.j * phi_prime) * \
            np.exp(-1.j * (omegar + 1.j * omegai) * t_arr)
    return fullh


def get_waveform_toy_no_exp(
        A_list,
        phi_list,
        qnm_list,
        t_arr,
        t_match,
        c_list,
        d_list,
        l=2,
        m=2):
    fullh = waveform_toy_no_exp(
        A_list,
        phi_list,
        qnm_list,
        t_arr,
        t_match,
        c_list,
        d_list,
        l=l,
        m=m)
    h = waveform(t_arr, fullh, t_peak=0, t_start=0, l=l, m=m)
    return h


def get_SXS_waveform_summed(SXS_num: str, iota: float, psi: float, 
                            l_max: int =4, res: int =0, N_ext: int=2) -> Tuple[waveform, float, float]:
    """
    Obtain the waveform of a SXS simulation summed over all modes up to l_max.

    Parameters:
        SXS_num: The SXS simulation number.
        iota: The inclination angle of the binary.
        psi: The phase to use.
        l_max: The maximum l mode to include in the waveform.
        res: The level of the simulation, relative to the highest resolution level 
            (`res = 0` means highest resolution). Must be zero or negative.
        N_ext: The extrapolation order of the simulation.

    Returns:
        h: The waveform.
        Mf: The final mass of the binary.
        af: The final spin of the binary.

    """

    Mf, af, Level, hdict = get_SXS_waveform_dict(SXS_num, res=res, N_ext=N_ext)

    hdict_complex = {}
    for key in hdict:
        lm_string = key.split(',')
        l = int(lm_string[0])
        m = int(lm_string[1])
        if l <= l_max:
            hdict_complex[(l, m)] = np.array(
                hdict[key][1] + 1.j * hdict[key][2])

    h_sum = _sum_modes_native(hdict_complex, iota, psi)

    t = hdict["2,2"][0]

    h = waveform(t, h_sum)

    return h, Mf, af


def _sum_modes_native(
        mode_array_dict: Dict[Tuple[int, int], np.ndarray],
        iota: float,
        psi: float) -> np.ndarray:
    """
    Sum spin-weighted spherical harmonic modes into a complex strain.
    """
    if len(mode_array_dict) == 0:
        raise ValueError("mode_array_dict must contain at least one (l, m) mode.")

    try:
        import quaternionic
        import spherical
    except ImportError as exc:
        raise ImportError(
            "Summing modes requires `spherical` and `quaternionic`. "
            "Install them in the active environment "
            "(e.g. `pixi add --pypi spherical quaternionic` "
            "or `pip install spherical quaternionic`)."
        ) from exc

    first_mode = next(iter(mode_array_dict.values()))
    mode_sum = np.zeros_like(first_mode, dtype=np.complex128)
    ell_max = max(l for l, _ in mode_array_dict)
    wigner = spherical.Wigner(ell_max)
    R = quaternionic.array.from_spherical_coordinates(iota, psi)
    y_all = wigner.sYlm(-2, R)

    for (l, m), h_lm in mode_array_dict.items():
        y_lm = y_all[wigner.Yindex(l, m)]
        mode_sum = mode_sum + np.asarray(h_lm, dtype=np.complex128) * y_lm

    return mode_sum


def delayed_QNM(
        mode,
        t,
        A,
        phi,
        A_red_ratio=1,
        A_delay=5,
        A_sig=1,
        dphi=-np.pi / 2,
        phi_delay=5,
        phi_sig=1):
    omega_i = mode.omegai
    omega_r = mode.omegar
    A_cons = A * np.exp(omega_i * t)
    A_red = A_red_ratio * A * (np.exp(omega_i * t)) * \
        (1 - np.tanh((t - A_delay) / A_sig)) / 2
    A_prime = A_cons - A_red
    A_osci = A_prime * np.exp(-1.j * (omega_r * t + phi))
    phase_delay = dphi * (1 - np.tanh((t - phi_delay) / phi_sig)) / 2
    return A_osci * np.exp(1.j * phase_delay)


def clean_QNM(mode, t, A, phi):
    omega_i = mode.omegai
    omega_r = mode.omegar
    return A * np.exp(omega_i * t) * np.exp(-1.j * (omega_r * t + phi))


def make_eff_ringdown_waveform(
        inj_dict,
        l,
        m,
        Mf,
        af,
        relevant_lm_list,
        noise_arr,
        time=np.linspace(
            0,
            150,
            num=1501),
    delay=True,
        t_end=DEFAULT_T_END):
    fund_string = list(inj_dict.keys())[0]
    mode_fund = long_str_to_qnms(fund_string, Mf, af)[0]
    A_fund, phi_fund = inj_dict[fund_string]

    if delay:
        h_fund = delayed_QNM(mode_fund, time, A_fund, phi_fund,
                             A_delay=0, A_sig=10,
                             phi_delay=0, dphi=-np.pi, phi_sig=5)
    else:
        h_fund = clean_QNM(mode_fund, time, A_fund, phi_fund)
    h0 = waveform(np.asarray(time), h_fund, remove_num=0, t_peak=0)

    h_effective = h0.h

    for key in list(inj_dict.keys())[1:]:
        omega = long_str_to_qnms(key, Mf, af)[0]
        A, phi = inj_dict[key]
        if delay:
            h_mode = delayed_QNM(omega, h0.time, A, phi,
                                 A_red_ratio=1, A_delay=5, A_sig=2,
                                 phi_delay=0, dphi=-np.pi, phi_sig=2)
        else:
            h_mode = clean_QNM(omega, h0.time, A, phi)
        h_effective += h_mode

    h_effective += np.asarray(noise_arr)

    h_eff = waveform(h0.time, h_effective, l=l, m=m,
                     remove_num=0, t_peak=0, t_end=t_end)

    return h_eff


def make_eff_ringdown_waveform_from_param(
        inject_params, delay=True, t_end=DEFAULT_T_END, noise=True):

    inj_dict = inject_params['inj_dict']
    l = inject_params['l']
    m = inject_params['m']
    Mf = inject_params['Mf']
    af = inject_params['af']
    relevant_lm_list = inject_params['relevant_lm_list']
    noise_arr = np.asarray(inject_params['noise_arr'])
    if not noise:
        noise_arr = 0
    time = np.asarray(inject_params['time'])
    h_eff = make_eff_ringdown_waveform(
        inj_dict,
        l,
        m,
        Mf,
        af,
        relevant_lm_list,
        noise_arr,
        time=time,
        delay=delay,
        t_end=t_end)
    return h_eff


def make_random_inject_params(
        inject_params_base,
        randomize_params,
        inj_dict_randomize,
        amp_order):

    inject_params = inject_params_base.copy()
    inj_dict = inject_params['inj_dict'].copy()

    for key, val in randomize_params.items():
        if key == 'af':
            inject_params[key] = uniform.rvs(*val)
        elif key == 'noise_random':
            if val:
                inject_params['noise_arr'] = rng.normal(
                    0, inject_params['noise_sig'], len(
                        inject_params['time']))
        else:
            inject_params[key] = loguniform.rvs(*val)

    amps_list = []
    for order in amp_order:
        num = len(order)
        amps = np.sort(uniform.rvs(
            *inj_dict_randomize[order[0]][0], size=num))[::-1]
        amps_list.append(amps)

    for key, val in inj_dict_randomize.items():
        for amps, order in zip(amps_list, amp_order):
            if key in order:
                indx = order.index(key)
                inj_dict[key] = (amps[indx], uniform.rvs(*val[1]))
                break
        else:
            inj_dict[key] = (loguniform.rvs(*val[0]), uniform.rvs(*val[1]))

    inject_params['inj_dict'] = inj_dict.copy()

    return inject_params


def compute_mismatch(t1: np.ndarray, h1: np.ndarray, t2: np.ndarray, h2: np.ndarray, tnum: int = 2000) -> float:
    t_low = t1[0]
    t_hi = min(t1[-1], t2[-1])
    t_grid = np.linspace(t_low, t_hi, num=max(tnum, len(t1)))
    h1_interp = griddata(t1, h1, t_grid)
    h2_interp = griddata(t2, h2, t_grid)
    mismatch = 1 - (np.real(np.vdot(h1_interp, h2_interp) / (
                    np.linalg.norm(h1_interp) * np.linalg.norm(h2_interp))))
    return mismatch


def mismatch_dphi_dt(dphi_dt, t1, h1, t2, h2, tnum=2000):
    dphi = dphi_dt[0]
    dt = dphi_dt[1]
    t2_shift = t2 + dt
    h2_shift = h2 * np.exp(1.j * dphi)
    return compute_mismatch(t1, h1, t2_shift, h2_shift, tnum=tnum)


def mismatch_min_phase(t1, h1, t2, h2, guess=[0, 0], tnum=2000):
    res = minimize(mismatch_dphi_dt, x0=guess, args=(t1, h1, t2, h2),
                   method='Nelder-Mead', tol=1e-13)
    return res


def estimate_resolution_mismatch(
    SXS_num, l, m, t0s=np.linspace(
        0, 50, num=51), remove_end=100, download = None):
    h_hi, _, _, Level, _ = get_waveform_SXS(SXS_num, l, m, res=0, download = download)
    h_low, _, _, Level, _ = get_waveform_SXS(SXS_num, l, m, res=-1, download = download)
    t_low_more, h_low_more_r, h_low_more_i = h_low.postmerger(-10)
    h_low_more = h_low_more_r + 1.j * h_low_more_i

    mismatches = []
    res = None
    guess = [0, 0]
    for t0 in t0s:
        t_hi_adj, h_hi_adj_r, h_hi_adj_i = h_hi.postmerger(t0)
        t_low_adj, h_low_adj_r, h_low_adj_i = h_low.postmerger(t0 - 5)
        h_hi_adj = h_hi_adj_r + 1.j * h_hi_adj_i
        h_low_adj = h_low_adj_r + 1.j * h_low_adj_i
        if t0 == 0:
            res = mismatch_min_phase(t_hi_adj[:-remove_end - 1],
                                     h_hi_adj[:-remove_end - 1],
                                     t_low_adj,
                                     h_low_adj,
                                     guess=guess)
            mismatches.append(res.fun)
            dphi = res.x[0]
            dt = res.x[1]
        else:
            mismatch = compute_mismatch(t_hi_adj[:-remove_end - 1],
                                        h_hi_adj[:-remove_end - 1],
                                        t_low_adj + dt,
                                        h_low_adj * np.exp(1.j * dphi))
            mismatches.append(mismatch)

    return mismatches
