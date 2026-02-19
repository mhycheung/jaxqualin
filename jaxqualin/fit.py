import numpy as np
import jax.numpy as jnp
from jax.numpy.linalg import pinv
import optimistix as optx
import scipy
from scipy.optimize import curve_fit, least_squares as scipy_least_squares

from .utils import interweave
from .qnmode import S_mirror_fac, qnms_to_string, make_mirror_ratio_list, mode, mode_free

from tqdm.auto import tqdm
import os
import pickle
from copy import copy
import random
from bisect import bisect_left, bisect_right
import warnings

from .waveforms import waveform

from dataclasses import dataclass, field
from typing import List, Tuple, Union, Optional, Dict, Any
from functools import partial

import logging

from jax import config, jit
config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)

FIT_SAVE_PATH = os.path.join(os.getcwd(), ".jaxqualin_cache/fits")

DEFAULT_SEED = 1234
DEFAULT_MAX_NFEV = 200000
DEFAULT_FIT_TOL = 1e-13
MARIMO_TQDM_NCOLS = 150


def _running_in_marimo() -> bool:
    """Return True when executing inside an active marimo runtime context."""
    try:
        from marimo._runtime.context import ContextNotInitializedError, get_context
    except Exception:
        return False
    try:
        get_context()
        return True
    except ContextNotInitializedError:
        return False
    except Exception:
        return False


def _tqdm_kwargs() -> Dict[str, Any]:
    """Use tqdm defaults normally; set ncols only in marimo notebooks."""
    if _running_in_marimo():
        return {"ncols": MARIMO_TQDM_NCOLS}
    return {}


@dataclass
class FitConfig:
    """Configuration for QNM fitting."""
    max_nfev: int = DEFAULT_MAX_NFEV
    sigma: float = 1.
    weight_by_amplitude: bool = False
    real: bool = False
    include_mirror: bool = False
    iota: float = None
    psi: float = None


@dataclass
class InitialGuessConfig:
    """Configuration for initial guess generation."""
    guess_num: int = 100
    A_log_low: float = -1
    A_log_hi: float = 1
    phi_low: float = 0
    phi_hi: float = 6.283185307179586  # 2*pi
    omega_r_low: float = -2
    omega_r_hi: float = 2
    omega_i_low: float = 0
    omega_i_hi: float = -1
    seed: int = DEFAULT_SEED
    A_val: float = None
    A_guess_relative: float = None


# Module-level flag to track if full warmup has been done
_FULL_WARMUP_DONE = False


def qnm_fit_func_mirror_fixed(
        t,
        qnm_fixed_list,
        fix_mode_params_list,
        mirror_ratio_list,
        part=None):
    Q = 0
    for qnm_fixed, fix_mode_params, mirror_ratio in zip(
            qnm_fixed_list, fix_mode_params_list, mirror_ratio_list):
        A, phi = tuple(fix_mode_params)
        omegar = qnm_fixed.omegar
        omegai = qnm_fixed.omegai
        if part is None:
            Q += A * jnp.exp(-1.j * ((omegar + 1.j * omegai) * t + phi))
            Q += mirror_ratio[0] * A * \
                jnp.exp(-1.j * ((-omegar + 1.j * omegai) * t - phi - mirror_ratio[1]))
        elif part == "real":
            Q += A * jnp.exp(omegai * t) * jnp.cos(omegar * t + phi)
            Q += mirror_ratio[0] * A * \
                jnp.exp(omegai * t) * jnp.cos(-omegar * t - phi - mirror_ratio[1])
        elif part == "imag":
            Q += -A * jnp.exp(omegai * t) * jnp.sin(omegar * t + phi)
            Q += - mirror_ratio[0] * A * \
                jnp.exp(omegai * t) * jnp.sin(-omegar * t - phi - mirror_ratio[1])
    return Q


def qnm_fit_func(
        t,
        qnm_fixed_list,
        fix_mode_params_list,
        free_mode_params_list,
        part=None):
    Q = 0
    for qnm_fixed, fix_mode_params in zip(
            qnm_fixed_list, fix_mode_params_list):
        A, phi = tuple(fix_mode_params)
        omegar = qnm_fixed.omegar
        omegai = qnm_fixed.omegai
        if part is None:
            Q += A * jnp.exp(-1.j * ((omegar + 1.j * omegai) * t + phi))
        elif part == "real":
            Q += A * jnp.exp(omegai * t) * jnp.cos(omegar * t + phi)
        elif part == "imag":
            Q += -A * jnp.exp(omegai * t) * jnp.sin(omegar * t + phi)
    for free_mode_params in free_mode_params_list:
        A, phi, omegar, omegai = tuple(free_mode_params)
        if part is None:
            Q += A * jnp.exp(-1.j * ((omegar + 1.j * omegai) * t + phi))
        elif part == "real":
            Q += A * jnp.exp(omegai * t) * jnp.cos(omegar * t + phi)
        elif part == "imag":
            Q += -A * jnp.exp(omegai * t) * jnp.sin(omegar * t + phi)
    return Q


def qnm_fit_func_varMa(
        t,
        qnm_fixed_list,
        qnm_free_list,
        fix_mode_params_list,
        free_mode_params_list,
        M,
        a,
        retro_def_orbit=True,
        part=None):
    Q = 0
    for qnm_fixed, fix_mode_params in zip(
            qnm_fixed_list, fix_mode_params_list):
        A, phi = tuple(fix_mode_params)
        omegar = qnm_fixed.omegar
        omegai = qnm_fixed.omegai
        if part is None:
            Q += A * np.exp(-1.j * ((omegar + 1.j * omegai) * t + phi))
        elif part == "real":
            Q += A * np.exp(omegai * t) * np.cos(omegar * t + phi)
        elif part == "imag":
            Q += -A * np.exp(omegai * t) * np.sin(omegar * t + phi)
    for free_mode_params, qnm_free in zip(
            free_mode_params_list, qnm_free_list):
        A, phi = tuple(free_mode_params)
        qnm_free.fix_mode(M, a, retro_def_orbit=retro_def_orbit)
        omegar = qnm_free.omegar
        omegai = qnm_free.omegai
        if part is None:
            Q += A * np.exp(-1.j * ((omegar + 1.j * omegai) * t + phi))
        elif part == "real":
            Q += A * np.exp(omegai * t) * np.cos(omegar * t + phi)
        elif part == "imag":
            Q += -A * np.exp(omegai * t) * np.sin(omegar * t + phi)
    return Q


def qnm_fit_func_varMa_mirror(
        t,
        qnm_fixed_list,
        qnm_free_list,
        fix_mode_params_list,
        free_mode_params_list,
        iota,
        psi,
        M,
        a,
        retro_def_orbit=True,
        part=None):
    Q = 0
    N_fix = len(qnm_fixed_list)
    for qnm_fixed, fix_mode_params in zip(
            qnm_fixed_list, fix_mode_params_list):
        A, phi = tuple(fix_mode_params)
        omegar = qnm_fixed.omegar
        omegai = qnm_fixed.omegai
        lmnx = qnm_fixed.lmnx
        if lmnx is None:
            raise ValueError(
                "Mirror mode fitting requires modes with lmnx quantum numbers. "
                "custom_mode objects cannot be used with include_mirror=True.")
        mirror_ratio = 1
        for lmn in lmnx:
            l, m, n = tuple(lmn)
            S_fac = S_mirror_fac(iota, a, l, m, n, psi=psi)
            mirror_ratio *= S_fac
        if part is None:
            Q += A * np.exp(-1.j * ((omegar + 1.j * omegai) * t + phi))
            Q += mirror_ratio * A * \
                np.exp(-1.j * ((-omegar + 1.j * omegai) * t - phi))
        elif part == "real":
            Q += A * np.exp(omegai * t) * np.cos(omegar * t + phi)
            Q += mirror_ratio * A * \
                np.exp(omegai * t) * np.cos(-omegar * t - phi)
        elif part == "imag":
            Q += -A * np.exp(omegai * t) * np.sin(omegar * t + phi)
            Q += - mirror_ratio * A * \
                np.exp(omegai * t) * np.sin(-omegar * t - phi)
    for free_mode_params, qnm_free in zip(
            free_mode_params_list, qnm_free_list):
        A, phi = tuple(free_mode_params)
        qnm_free.fix_mode(M, a, retro_def_orbit=retro_def_orbit)
        omegar = qnm_free.omegar
        omegai = qnm_free.omegai
        lmnx = qnm_free.lmnx
        if lmnx is None:
            raise ValueError(
                "Mirror mode fitting requires modes with lmnx quantum numbers. "
                "custom_mode objects cannot be used with include_mirror=True.")
        mirror_ratio = 1
        for lmn in lmnx:
            l, m, n = tuple(lmn)
            S_fac = S_mirror_fac(iota, a, l, m, n, psi=psi)
            mirror_ratio *= S_fac
        if part is None:
            Q += A * np.exp(-1.j * ((omegar + 1.j * omegai) * t + phi))
            Q += mirror_ratio * A * \
                np.exp(-1.j * ((-omegar + 1.j * omegai) * t - phi))
        elif part == "real":
            Q += A * np.exp(omegai * t) * np.cos(omegar * t + phi)
            Q += mirror_ratio * A * \
                np.exp(omegai * t) * np.cos(-omegar * t - phi)
        elif part == "imag":
            Q += -A * np.exp(omegai * t) * np.sin(omegar * t + phi)
            Q += - mirror_ratio * A * \
                np.exp(omegai * t) * np.sin(-omegar * t - phi)
    return Q

# https://stackoverflow.com/questions/34136737/using-scipy-curve-fit-for-a-variable-number-of-parameters


def qnm_fit_func_wrapper(t, qnm_fixed_list, N_free, *args, part=None):
    N_fix = len(qnm_fixed_list)
    fix_mode_params_list = []
    for i in range(N_fix):
        A = args[0][2 * i]
        phi = args[0][2 * i + 1]
        fix_mode_params_list.append([A, phi])
    free_mode_params_list = []
    for j in range(N_free):
        try:
            A = args[0][2 * N_fix + 4 * j]
            phi = args[0][2 * N_fix + 4 * j + 1]
            omegar = args[0][2 * N_fix + 4 * j + 2]
            omegai = args[0][2 * N_fix + 4 * j + 3]
            free_mode_params_list.append([A, phi, omegar, omegai])
        except BaseException:
            raise ValueError
    return qnm_fit_func(t, qnm_fixed_list, fix_mode_params_list,
                        free_mode_params_list, part=part)


def qnm_fit_func_mirror_wrapper(
        t,
        qnm_fixed_list,
        mirror_ratio_list,
        *args,
        part=None):
    N_fix = len(qnm_fixed_list)
    fix_mode_params_list = []
    for i in range(N_fix):
        A = args[0][2 * i]
        phi = args[0][2 * i + 1]
        fix_mode_params_list.append([A, phi])
    return qnm_fit_func_mirror_fixed(t, qnm_fixed_list, fix_mode_params_list,
                                     mirror_ratio_list, part=part)


def qnm_fit_func_wrapper_varMa(
        t,
        qnm_fixed_list,
        qnm_free_list,
        retro_def_orbit,
        *args,
        real=False,
        part=None):
    N_fix = len(qnm_fixed_list)
    N_free = len(qnm_free_list)
    fix_mode_params_list = []
    for i in range(N_fix):
        A = args[0][2 * i]
        phi = args[0][2 * i + 1]
        fix_mode_params_list.append([A, phi])
    free_mode_params_list = []
    for j in range(N_free):
        A = args[0][2 * N_fix + 2 * j]
        phi = args[0][2 * N_fix + 2 * j + 1]
        free_mode_params_list.append([A, phi])
    M = args[0][2 * (N_fix + N_free)]
    if real:
        return qnm_fit_func_varMa(
            t,
            qnm_fixed_list,
            qnm_free_list,
            fix_mode_params_list,
            free_mode_params_list,
            M,
            0.,
            retro_def_orbit=retro_def_orbit,
            part=part)
    else:
        a = args[0][2 * (N_fix + N_free) + 1]
        return qnm_fit_func_varMa(
            t,
            qnm_fixed_list,
            qnm_free_list,
            fix_mode_params_list,
            free_mode_params_list,
            M,
            a,
            retro_def_orbit=retro_def_orbit,
            part=part)


def qnm_fit_func_wrapper_varMa_mirror(
        t,
        qnm_fixed_list,
        qnm_free_list,
        iota,
        psi,
        retro_def_orbit,
        *args,
        real=False,
        part=None):
    N_fix = len(qnm_fixed_list)
    N_free = len(qnm_free_list)
    fix_mode_params_list = []
    for i in range(N_fix):
        A = args[0][2 * i]
        phi = args[0][2 * i + 1]
        fix_mode_params_list.append([A, phi])
    free_mode_params_list = []
    for j in range(N_free):
        A = args[0][2 * N_fix + 2 * j]
        phi = args[0][2 * N_fix + 2 * j + 1]
        free_mode_params_list.append([A, phi])
    M = args[0][2 * (N_fix + N_free)]
    if real:
        return qnm_fit_func_varMa_mirror(
            t,
            qnm_fixed_list,
            qnm_free_list,
            fix_mode_params_list,
            free_mode_params_list,
            iota,
            psi,
            M,
            0.,
            retro_def_orbit=retro_def_orbit,
            part=part)
    else:
        a = args[0][2 * (N_fix + N_free) + 1]
        return qnm_fit_func_varMa_mirror(
            t,
            qnm_fixed_list,
            qnm_free_list,
            fix_mode_params_list,
            free_mode_params_list,
            iota,
            psi,
            M,
            a,
            retro_def_orbit=retro_def_orbit,
            part=part)


# https://stackoverflow.com/questions/50203879/curve-fitting-of-complex-data


def qnm_fit_func_wrapper_complex(
        t,
        qnm_fixed_list,
        N_free,
        *args,
        real=False):
    N = len(t)
    t_real = t[0::2]
    t_imag = t[1::2]
    h_real = qnm_fit_func_wrapper(
        t_real, qnm_fixed_list, N_free, *args, part="real")
    if real:
        h_imag = jnp.zeros(int(N / 2))
    else:
        h_imag = qnm_fit_func_wrapper(
            t_imag, qnm_fixed_list, N_free, *args, part="imag")
    h_riffle = interweave(h_real, h_imag)
    return h_riffle


def qnm_fit_func_wrapper_complex_mirror(
        t,
        qnm_fixed_list,
        mirror_ratio_list,
        N_free,
        *args,
        real=False):
    N = len(t)
    t_real = t[0::2]
    t_imag = t[1::2]
    h_real = qnm_fit_func_mirror_wrapper(
        t_real, qnm_fixed_list, mirror_ratio_list, *args, part="real")
    if real:
        h_imag = jnp.zeros(int(N / 2))
    else:
        h_imag = qnm_fit_func_mirror_wrapper(
            t_imag, qnm_fixed_list, mirror_ratio_list, *args, part="imag")
    h_riffle = interweave(h_real, h_imag)
    return h_riffle


def qnm_fit_func_wrapper_complex_varMa(
        t,
        qnm_fixed_list,
        qnm_free_list,
        retro_def_orbit,
        *args):
    N = len(t)
    t_real = t[0::2]
    t_imag = t[1::2]
    h_real = qnm_fit_func_wrapper_varMa(
        t_real,
        qnm_fixed_list,
        qnm_free_list,
        retro_def_orbit,
        *args,
        part="real")
    h_imag = qnm_fit_func_wrapper_varMa(
        t_imag,
        qnm_fixed_list,
        qnm_free_list,
        retro_def_orbit,
        *args,
        part="imag")
    h_riffle = interweave(h_real, h_imag)
    return h_riffle


def qnm_fit_func_wrapper_complex_varMa_mirror(
        t,
        qnm_fixed_list,
        qnm_free_list,
        iota,
        psi,
        retro_def_orbit,
        *args):
    N = len(t)
    t_real = t[0::2]
    t_imag = t[1::2]
    h_real = qnm_fit_func_wrapper_varMa_mirror(
        t_real,
        qnm_fixed_list,
        qnm_free_list,
        iota,
        psi,
        retro_def_orbit,
        *args,
        part="real")
    h_imag = qnm_fit_func_wrapper_varMa_mirror(
        t_imag,
        qnm_fixed_list,
        qnm_free_list,
        iota,
        psi,
        retro_def_orbit,
        *args,
        part="imag")
    h_riffle = interweave(h_real, h_imag)
    return h_riffle


# ---------------------------------------------------------------------------
# Generalised wrappers for arbitrary QNMModel parameters
# ---------------------------------------------------------------------------


def qnm_fit_func_var_model(
        t,
        qnm_fixed_list,
        qnm_free_list,
        fix_mode_params_list,
        free_mode_params_list,
        model_params,
        part=None):
    """Like ``qnm_fit_func_varMa`` but with an arbitrary parameter dict."""
    Q = 0
    for qnm_fixed, fix_mode_params in zip(
            qnm_fixed_list, fix_mode_params_list):
        A, phi = tuple(fix_mode_params)
        omegar = qnm_fixed.omegar
        omegai = qnm_fixed.omegai
        if part is None:
            Q += A * np.exp(-1.j * ((omegar + 1.j * omegai) * t + phi))
        elif part == "real":
            Q += A * np.exp(omegai * t) * np.cos(omegar * t + phi)
        elif part == "imag":
            Q += -A * np.exp(omegai * t) * np.sin(omegar * t + phi)
    for free_mode_params, qnm_free in zip(
            free_mode_params_list, qnm_free_list):
        A, phi = tuple(free_mode_params)
        qnm_free.fix_mode(**model_params)
        omegar = qnm_free.omegar
        omegai = qnm_free.omegai
        if part is None:
            Q += A * np.exp(-1.j * ((omegar + 1.j * omegai) * t + phi))
        elif part == "real":
            Q += A * np.exp(omegai * t) * np.cos(omegar * t + phi)
        elif part == "imag":
            Q += -A * np.exp(omegai * t) * np.sin(omegar * t + phi)
    return Q


def qnm_fit_func_wrapper_var_model(
        t,
        qnm_fixed_list,
        qnm_free_list,
        model,
        *args,
        part=None):
    """Wrapper that unpacks optimisation vector for a generic QNMModel."""
    N_fix = len(qnm_fixed_list)
    N_free = len(qnm_free_list)
    fix_mode_params_list = []
    for i in range(N_fix):
        A = args[0][2 * i]
        phi = args[0][2 * i + 1]
        fix_mode_params_list.append([A, phi])
    free_mode_params_list = []
    for j in range(N_free):
        A = args[0][2 * N_fix + 2 * j]
        phi = args[0][2 * N_fix + 2 * j + 1]
        free_mode_params_list.append([A, phi])
    model_params = {}
    for k, name in enumerate(model.param_names):
        model_params[name] = args[0][2 * (N_fix + N_free) + k]
    return qnm_fit_func_var_model(
        t,
        qnm_fixed_list,
        qnm_free_list,
        fix_mode_params_list,
        free_mode_params_list,
        model_params,
        part=part)


def qnm_fit_func_wrapper_complex_var_model(
        t,
        qnm_fixed_list,
        qnm_free_list,
        model,
        *args):
    """Complex-interweaved wrapper for a generic QNMModel."""
    N = len(t)
    t_real = t[0::2]
    t_imag = t[1::2]
    h_real = qnm_fit_func_wrapper_var_model(
        t_real,
        qnm_fixed_list,
        qnm_free_list,
        model,
        *args,
        part="real")
    h_imag = qnm_fit_func_wrapper_var_model(
        t_imag,
        qnm_fixed_list,
        qnm_free_list,
        model,
        *args,
        part="imag")
    h_riffle = interweave(h_real, h_imag)
    return h_riffle


# ---------------------------------------------------------------------------
# VARPRO helpers for QNMModel-based fits
# ---------------------------------------------------------------------------
# With VARPRO the optimizer only searches over the *model* parameters
# (e.g. M, a, delta).  At every evaluation the complex linear
# coefficients  c_j = A_j exp(-i phi_j)  are solved analytically via
# least-squares, removing them from the search space entirely.
# ---------------------------------------------------------------------------


def _varpro_basis_model(time, qnm_fixed_list, qnm_free_list, model,
                        model_params):
    """Build the complex-exponential basis matrix for VARPRO.

    Returns an (N_t, N_modes) complex ndarray where each column is
    ``exp(-i * omega_j * t)`` for one mode.
    """
    for qnm_free in qnm_free_list:
        qnm_free.fix_mode(**model_params)

    cols = []
    for qnm_fixed in qnm_fixed_list:
        omega = complex(qnm_fixed.omegar) + 1j * complex(qnm_fixed.omegai)
        cols.append(np.exp(-1j * omega * time))
    for qnm_free in qnm_free_list:
        omega = complex(qnm_free.omegar) + 1j * complex(qnm_free.omegai)
        cols.append(np.exp(-1j * omega * time))

    return np.column_stack(cols)          # (N_t, N_modes)


def _varpro_residual_model(model_params_arr, time, y, qnm_fixed_list,
                           qnm_free_list, model):
    """VARPRO residual: only model params are nonlinear.

    Parameters
    ----------
    model_params_arr : 1-D array of model parameter values.
    time : 1-D real array of time samples.
    y : 1-D complex array, the target waveform  ``hr + 1j*hi``.
    qnm_fixed_list, qnm_free_list, model : as elsewhere.

    Returns
    -------
    1-D real array of length ``2 * len(time)``.
    """
    model_params = {name: model_params_arr[k]
                    for k, name in enumerate(model.param_names)}

    basis = _varpro_basis_model(time, qnm_fixed_list, qnm_free_list,
                                model, model_params)

    # Solve for complex linear coefficients  c = A * exp(-i phi)
    c, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)

    residual = y - basis @ c
    return np.concatenate([residual.real, residual.imag])


def _varpro_assemble_popt_model(model_params_arr, time, y,
                                qnm_fixed_list, qnm_free_list, model):
    """Assemble the full ``popt`` vector from VARPRO solution.

    The output format is
    ``[A_fix_0, phi_fix_0, …, A_free_0, phi_free_0, …, model_p0, …]``
    matching the layout expected by
    :meth:`QNMFitVaryingStartingTimeResultModel.process_results`.
    """
    N_fix = len(qnm_fixed_list)
    N_free = len(qnm_free_list)
    n_model = len(model.param_names)

    model_params = {name: model_params_arr[k]
                    for k, name in enumerate(model.param_names)}

    basis = _varpro_basis_model(time, qnm_fixed_list, qnm_free_list,
                                model, model_params)
    c, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)

    # Convention:  h(t) = A * exp(-i*(omega*t + phi))  =>  c = A * exp(-i*phi)
    A_arr = np.abs(c)
    phi_arr = -np.angle(c)

    popt = np.zeros(2 * N_fix + 2 * N_free + n_model)
    for i in range(N_fix):
        popt[2 * i] = A_arr[i]
        popt[2 * i + 1] = phi_arr[i]
    for j in range(N_free):
        popt[2 * N_fix + 2 * j] = A_arr[N_fix + j]
        popt[2 * N_fix + 2 * j + 1] = phi_arr[N_fix + j]
    for k in range(n_model):
        popt[2 * N_fix + 2 * N_free + k] = model_params_arr[k]

    return popt


def _varpro_reconstruct_model(time, popt, qnm_fixed_list, qnm_free_list,
                              model):
    """Reconstruct the complex waveform from a VARPRO popt vector."""
    N_fix = len(qnm_fixed_list)
    N_free = len(qnm_free_list)
    n_model = len(model.param_names)

    model_params_arr = popt[2 * N_fix + 2 * N_free:]
    model_params = {name: model_params_arr[k]
                    for k, name in enumerate(model.param_names)}

    basis = _varpro_basis_model(time, qnm_fixed_list, qnm_free_list,
                                model, model_params)

    # Recover complex coefficients from (A, phi)
    c = np.zeros(N_fix + N_free, dtype=np.complex128)
    for i in range(N_fix):
        A = popt[2 * i]
        phi = popt[2 * i + 1]
        c[i] = A * np.exp(-1j * phi)
    for j in range(N_free):
        A = popt[2 * N_fix + 2 * j]
        phi = popt[2 * N_fix + 2 * j + 1]
        c[N_fix + j] = A * np.exp(-1j * phi)

    return basis @ c


class QNMFitResult:

    def __init__(self, popt, pcov, mismatch,
                 cost=np.nan, grad=np.nan, nfev=np.nan,
                 status=np.nan):
        self.popt = popt
        self.pcov = pcov
        self.mismatch = mismatch
        self.cost = cost
        self.grad = grad
        self.nfev = nfev
        self.status = status


def model_func_optimized(nonlinear_params, t, omegar_fixed, omegai_fixed, mirror_ratio_list, N_free, N_fix, include_mirror):
    omegar_arr = nonlinear_params[0:N_free]
    omegai_arr = nonlinear_params[N_free:2*N_free]
    
    basis = []
    # fixed modes
    for i in range(N_fix):
        omega = omegar_fixed[i] + 1.j * omegai_fixed[i]
        basis.append(jnp.exp(-1.j * omega * t))
        if include_mirror:
            mirror_ratio = mirror_ratio_list[i]
            basis.append(mirror_ratio[0] * jnp.exp(-1.j * (-omega.real + 1.j*omega.imag) * t - mirror_ratio[1]))

    # free modes
    for i in range(N_free):
        omega = omegar_arr[i] + 1.j * omegai_arr[i]
        basis.append(jnp.exp(-1.j * omega * t))

    return jnp.array(basis).T

def residual_func_optimized(nonlinear_params, args, N_free, N_fix, include_mirror):
    t, y, sigma, omegar_fixed, omegai_fixed, mirror_ratio_list, mask = args
    basis = model_func_optimized(nonlinear_params, t, omegar_fixed, omegai_fixed, mirror_ratio_list, N_free, N_fix, include_mirror)
    
    basis_masked = basis * mask[:, None]
    y_masked = y * mask
    
    linear_params, _, _, _ = jnp.linalg.lstsq(basis_masked/sigma[:, None], y_masked/sigma)
    
    y_fit = jnp.dot(basis, linear_params)
    residual = (y - y_fit) * mask
    return jnp.concatenate([residual.real, residual.imag])


def _do_optimization(params0, args, N_free, N_fix, include_mirror, max_nfev):
    """JIT-compiled optimization function."""
    residual = partial(residual_func_optimized, N_free=N_free, N_fix=N_fix, include_mirror=include_mirror)
    solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
    sol = optx.least_squares(residual, solver, params0, 
                             args=args, max_steps=max_nfev,
                             throw=False)
    return sol.value


def _compute_linear_params_and_popt(final_nonlinear_params, time, y, sigma, mask, 
                                     omegar_fixed, omegai_fixed, mirror_ratio_arr,
                                     N_free, N_fix, include_mirror):
    """JIT-compiled function to compute linear parameters and assemble popt."""
    # Compute final basis and linear parameters
    final_basis = model_func_optimized(final_nonlinear_params, time, omegar_fixed, 
                                       omegai_fixed, mirror_ratio_arr, N_free, N_fix, include_mirror)
    final_basis_masked = final_basis * mask[:, None]
    y_masked = y * mask
    final_linear_params, _, _, _ = jnp.linalg.lstsq(final_basis_masked/sigma[:, None], y_masked/sigma)
    
    # Extract nonlinear parameters
    omegar_final = final_nonlinear_params[0:N_free]
    omegai_final = final_nonlinear_params[N_free:2*N_free]
    
    # Assemble popt array
    # The fit convention is h(t) = A * exp(-i*(omega*t + phi)), so the complex
    # coefficient c = A * exp(-i*phi), meaning phi = -angle(c).
    # When include_mirror=True, the basis columns are interleaved:
    #   [prograde_0, mirror_0, prograde_1, mirror_1, ...],
    # so prograde coefficients are at even indices (0, 2, 4, ...).
    popt = jnp.zeros(2*N_fix + 4*N_free)
    if include_mirror:
        prograde_params = final_linear_params[0::2][:N_fix]
        n_basis_fixed = 2 * N_fix
    else:
        prograde_params = final_linear_params[0:N_fix]
        n_basis_fixed = N_fix
    A_fix = jnp.abs(prograde_params)
    phi_fix = -jnp.angle(prograde_params)
    popt = popt.at[0:2*N_fix:2].set(A_fix)
    popt = popt.at[1:2*N_fix:2].set(phi_fix)
    
    free_params = final_linear_params[n_basis_fixed:]
    A_free = jnp.abs(free_params)
    phi_free = -jnp.angle(free_params)
    popt = popt.at[2*N_fix::4].set(A_free)
    popt = popt.at[2*N_fix+1::4].set(phi_free)
    popt = popt.at[2*N_fix+2::4].set(omegar_final)
    popt = popt.at[2*N_fix+3::4].set(omegai_final)
    
    return popt


def _reconstruct_waveform(time, popt, omegar_fixed, omegai_fixed, N_free, N_fix):
    """JIT-compiled waveform reconstruction for non-mirror case."""
    fix_mode_params_list = []
    for i in range(N_fix):
        A = popt[2*i]
        phi = popt[2*i + 1]
        fix_mode_params_list.append([A, phi])
    free_mode_params_list = []
    for j in range(N_free):
        A = popt[2*N_fix + 4*j]
        phi = popt[2*N_fix + 4*j + 1]
        omegar = popt[2*N_fix + 4*j + 2]
        omegai = popt[2*N_fix + 4*j + 3]
        free_mode_params_list.append([A, phi, omegar, omegai])
    
    Q = jnp.zeros(len(time), dtype=jnp.complex128)
    for i in range(N_fix):
        A, phi = fix_mode_params_list[i]
        omegar = omegar_fixed[i]
        omegai = omegai_fixed[i]
        Q = Q + A * jnp.exp(-1.j * ((omegar + 1.j * omegai) * time + phi))
    for free_params in free_mode_params_list:
        A, phi, omegar, omegai = free_params
        Q = Q + A * jnp.exp(-1.j * ((omegar + 1.j * omegai) * time + phi))
    return Q


def _prepare_fit_data(time, hr, hi, sigma, max_len):
    """JIT-compiled function to prepare fit data with padding.
    
    Input arrays must be pre-padded to max_len with zeros.
    This function creates the mask based on which elements are non-zero in sigma.
    """
    # The input arrays are already max_len size (pre-padded with zeros)
    y = hr + 1.j * hi
    
    # Create mask: 1 where sigma is finite, 0 otherwise (padded regions have inf)
    mask = jnp.where(jnp.isinf(sigma), 0.0, 1.0)
    
    return time, hr, hi, y, sigma, mask


# ---------------------------------------------------------------------------
# Schwarzschild (real-waveform) VARPRO functions
# ---------------------------------------------------------------------------
# For Schwarzschild black holes the waveform is purely real:
#   h(t) = sum_j A_j * exp(omega_i_j * t) * cos(omega_r_j * t + phi_j)
#
# We decompose the complex basis exp(-i*omega*t) into two real columns:
#   col_cos = Re(exp(-i*omega*t)) = exp(omega_i*t) * cos(omega_r*t)
#   col_sin = Im(exp(-i*omega*t)) = -exp(omega_i*t) * sin(omega_r*t)
# and fit h_real = a * col_cos + b * col_sin with real coefficients a, b.
# The complex coefficient c = a + i*b then yields A = |c|, phi = -angle(c).

def _model_func_real(nonlinear_params, t, omegar_fixed, omegai_fixed, N_free, N_fix):
    """Build a real-valued basis for Schwarzschild fitting.

    Each mode contributes two real columns (cos and sin parts), so the
    returned matrix has shape (len(t), 2*(N_fix + N_free)).
    """
    omegar_arr = nonlinear_params[0:N_free]
    omegai_arr = nonlinear_params[N_free:2*N_free]

    basis = []
    for i in range(N_fix):
        exp_decay = jnp.exp(omegai_fixed[i] * t)
        basis.append(exp_decay * jnp.cos(omegar_fixed[i] * t))   # Re column
        basis.append(-exp_decay * jnp.sin(omegar_fixed[i] * t))  # Im column
    for i in range(N_free):
        exp_decay = jnp.exp(omegai_arr[i] * t)
        basis.append(exp_decay * jnp.cos(omegar_arr[i] * t))
        basis.append(-exp_decay * jnp.sin(omegar_arr[i] * t))

    return jnp.array(basis).T


def _residual_func_real(nonlinear_params, args, N_free, N_fix):
    """Residual function for Schwarzschild real-valued VARPRO."""
    t, y_real, sigma, omegar_fixed, omegai_fixed, mask = args
    basis = _model_func_real(nonlinear_params, t, omegar_fixed, omegai_fixed, N_free, N_fix)

    basis_masked = basis * mask[:, None]
    y_masked = y_real * mask

    linear_params, _, _, _ = jnp.linalg.lstsq(basis_masked / sigma[:, None], y_masked / sigma)

    y_fit = jnp.dot(basis, linear_params)
    residual = (y_real - y_fit) * mask
    return residual


def _do_optimization_real(params0, args, N_free, N_fix, max_nfev):
    """Optimization driver for Schwarzschild real-valued VARPRO."""
    residual = partial(_residual_func_real, N_free=N_free, N_fix=N_fix)
    solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
    sol = optx.least_squares(residual, solver, params0,
                             args=args, max_steps=max_nfev,
                             throw=False)
    return sol.value


def _compute_linear_params_and_popt_real(final_nonlinear_params, time, y_real, sigma, mask,
                                         omegar_fixed, omegai_fixed, N_free, N_fix):
    """Compute linear params and assemble popt for Schwarzschild real-valued fit.

    The real basis gives two coefficients (a, b) per mode.  Reassembling
    c = a + i*b recovers A = |c| and phi = -angle(c).
    """
    final_basis = _model_func_real(final_nonlinear_params, time, omegar_fixed, omegai_fixed, N_free, N_fix)
    final_basis_masked = final_basis * mask[:, None]
    y_masked = y_real * mask
    final_linear_params, _, _, _ = jnp.linalg.lstsq(final_basis_masked / sigma[:, None], y_masked / sigma)

    omegar_final = final_nonlinear_params[0:N_free]
    omegai_final = final_nonlinear_params[N_free:2*N_free]

    popt = jnp.zeros(2 * N_fix + 4 * N_free)

    # Fixed modes: pairs (a, b) at indices [2*i, 2*i+1]
    # Re(c * B) = Re(c)*Re(B) - Im(c)*Im(B) = a*col_cos + b*col_sin,
    # so a = Re(c) and b = -Im(c), giving c = a - i*b.
    for i in range(N_fix):
        a = final_linear_params[2 * i]
        b = final_linear_params[2 * i + 1]
        c = a - 1.j * b
        A = jnp.abs(c)
        phi = -jnp.angle(c)
        popt = popt.at[2 * i].set(A)
        popt = popt.at[2 * i + 1].set(phi)

    # Free modes: pairs (a, b) at indices [2*N_fix + 2*j, 2*N_fix + 2*j+1]
    # For Schwarzschild, cos(omega_r*t + phi) = cos(-omega_r*t - phi),
    # so positive and negative omega_r are degenerate.  We normalise to
    # omega_r >= 0 by flipping the sign of both omega_r and phi when needed.
    for j in range(N_free):
        a = final_linear_params[2 * N_fix + 2 * j]
        b = final_linear_params[2 * N_fix + 2 * j + 1]
        c = a - 1.j * b
        A = jnp.abs(c)
        phi = -jnp.angle(c)
        omegar_j = omegar_final[j]
        sign = jnp.where(omegar_j < 0, -1.0, 1.0)
        omegar_j = omegar_j * sign
        phi = phi * sign
        popt = popt.at[2 * N_fix + 4 * j].set(A)
        popt = popt.at[2 * N_fix + 4 * j + 1].set(phi)
        popt = popt.at[2 * N_fix + 4 * j + 2].set(omegar_j)
        popt = popt.at[2 * N_fix + 4 * j + 3].set(omegai_final[j])

    return popt


# Create JIT-compiled versions for common configurations
# The static_argnums specify which arguments don't change and can be used for caching
_do_optimization_jit = jit(_do_optimization, static_argnums=(2, 3, 4, 5))
_compute_linear_params_and_popt_jit = jit(_compute_linear_params_and_popt, static_argnums=(8, 9, 10))
_reconstruct_waveform_jit = jit(_reconstruct_waveform, static_argnums=(4, 5))
_prepare_fit_data_jit = jit(_prepare_fit_data, static_argnums=(4,))
_do_optimization_real_jit = jit(_do_optimization_real, static_argnums=(2, 3, 4))
_compute_linear_params_and_popt_real_jit = jit(_compute_linear_params_and_popt_real, static_argnums=(7, 8))


class QNMFitBase:
    """Base class for QNM fitting with shared initialization and utilities."""

    def __init__(
            self,
            h,
            t0,
            qnm_fixed_list=[],
            N_free=0,
            real=False,
            max_nfev=DEFAULT_MAX_NFEV,
            include_mirror=False,
            iota=None,
            psi=None,
            weight_by_amplitude=False,
            sigma=1.,
            **fit_kwargs):
        self.h = h
        self.t0 = t0
        self.qnm_fixed_list = qnm_fixed_list
        self.N_fix = len(qnm_fixed_list)
        self.N_free = N_free
        self.real = real
        self.max_nfev = max_nfev
        self.include_mirror = include_mirror
        self.iota = iota
        self.psi = psi
        self.weight_by_amplitude = weight_by_amplitude
        self.sigma = sigma
        self.fit_kwargs = fit_kwargs
        self.fit_done = False
        self.popt = None
        self.pcov = None
        self.mismatch = None
        self.result = None

    def make_weights(self, hr, hi):
        habs = np.abs(hr + 1.j * hi)
        weight = interweave(habs, habs)
        return np.array(weight)


class QNMFit(QNMFitBase):

    def __init__(
            self,
            h,
            t0,
            N_free,
            qnm_fixed_list=[],
            real=False,
            params0=None,
            max_nfev=DEFAULT_MAX_NFEV,
            A_bound=np.inf,
            weighted=False,
            include_mirror=False,
            mirror_ratio_list=None,
            guess_fixed=[1, 1],
            guess_free=[1, 1, 1, -1],
            max_len=None,
            **fit_kwargs):
        super().__init__(
            h=h, t0=t0, qnm_fixed_list=qnm_fixed_list, N_free=N_free,
            real=real, max_nfev=max_nfev,
            include_mirror=include_mirror, weight_by_amplitude=weighted,
            **fit_kwargs)
        self.A_bound = A_bound
        self.weighted = weighted
        if self.include_mirror and self.N_free != 0:
            raise ValueError("Mirror is only allowed for fixed modes.")
        if self.include_mirror and mirror_ratio_list is None:
            raise ValueError("Mirror ratio list is not provided.")
        self.mirror_ratio_list = mirror_ratio_list
        self.guess_fixed = guess_fixed
        self.guess_free = guess_free
        self.max_len = max_len
        if params0 is not None:
            omegar_initial = params0[2*self.N_fix+2::4]
            omegai_initial = params0[2*self.N_fix+3::4]
            self.params0 = jnp.concatenate([omegar_initial, omegai_initial])
        else:
            if self.N_free > 0:
                omegar_guesses = [self.guess_free[2]] * self.N_free
                omegai_guesses = [self.guess_free[3]] * self.N_free
                self.params0 = jnp.array(omegar_guesses + omegai_guesses)
            else:
                self.params0 = jnp.array([])

    def do_fit(self, return_jcf=False):
        
        time_raw, hr_raw, hi_raw = self.h.postmerger(self.t0)
        
        if self.weighted:
            sigma_raw = self.make_weights(hr_raw, hi_raw)
        else:
            sigma_raw = np.ones(len(hr_raw))

        # Store original length for mismatch calculation
        original_len = len(time_raw)

        if self.max_len is not None:
            # Pre-pad arrays to max_len using numpy (fast, no JIT tracing)
            pad_len = self.max_len - original_len
            if pad_len > 0:
                # Use numpy pad which is fast and doesn't trigger JAX tracing
                time_padded = np.pad(np.asarray(time_raw), (0, pad_len), constant_values=0)
                hr_padded = np.pad(np.asarray(hr_raw), (0, pad_len), constant_values=0)
                hi_padded = np.pad(np.asarray(hi_raw), (0, pad_len), constant_values=0)
                sigma_padded = np.pad(np.asarray(sigma_raw), (0, pad_len), constant_values=np.inf)
            else:
                time_padded = np.asarray(time_raw)
                hr_padded = np.asarray(hr_raw)
                hi_padded = np.asarray(hi_raw)
                sigma_padded = np.asarray(sigma_raw)
            
            # Convert to JAX arrays and create y and mask using JIT
            self.time, self.hr, self.hi, y, sigma, mask = _prepare_fit_data_jit(
                jnp.asarray(time_padded), jnp.asarray(hr_padded), 
                jnp.asarray(hi_padded), jnp.asarray(sigma_padded), self.max_len
            )
        else:
            self.time = time_raw
            self.hr = hr_raw
            self.hi = hi_raw
            y = hr_raw + 1.j * hi_raw
            sigma = jnp.asarray(sigma_raw)
            mask = jnp.ones(len(self.time))

        omegar_fixed = jnp.array([qnm.omegar for qnm in self.qnm_fixed_list])
        omegai_fixed = jnp.array([qnm.omegai for qnm in self.qnm_fixed_list])
        
        if self.include_mirror:
             mirror_ratio_arr = jnp.array(self.mirror_ratio_list)
        else:
             mirror_ratio_arr = jnp.array([])

        if self.real:
            # Schwarzschild: use real-valued VARPRO with cos/sin basis
            y_real = jnp.asarray(self.hr)
            args_real = (self.time, y_real, sigma, omegar_fixed, omegai_fixed, mask)

            final_nonlinear_params = _do_optimization_real_jit(
                self.params0, args_real, self.N_free, self.N_fix, self.max_nfev
            )
            popt = _compute_linear_params_and_popt_real_jit(
                final_nonlinear_params, self.time, y_real, sigma, mask,
                omegar_fixed, omegai_fixed, self.N_free, self.N_fix
            )
        else:
            args = (self.time, y, sigma, omegar_fixed, omegai_fixed, mirror_ratio_arr, mask)

            # Use JIT-compiled optimization function for better performance
            final_nonlinear_params = _do_optimization_jit(
                self.params0, args, self.N_free, self.N_fix, self.include_mirror, self.max_nfev
            )

            # Use JIT-compiled post-processing for linear params and popt
            popt = _compute_linear_params_and_popt_jit(
                final_nonlinear_params, self.time, y, sigma, mask,
                omegar_fixed, omegai_fixed, mirror_ratio_arr,
                self.N_free, self.N_fix, self.include_mirror
            )
        
        self.popt = popt
        self.pcov = jnp.full((len(popt), len(popt)), jnp.nan)

        self.cost = None
        self.grad = jnp.nan
        self.nfev = None
        self.status = None
        
        # Reconstruct waveform
        if self.real:
            reconstruct_h_padded = qnm_fit_func_wrapper(
                self.time, self.qnm_fixed_list, self.N_free, self.popt, part="real")
        elif self.include_mirror:
            reconstruct_h_padded = qnm_fit_func_mirror_wrapper(
                self.time, self.qnm_fixed_list, self.mirror_ratio_list, self.popt)
        else:
            # Use JIT-compiled reconstruction
            reconstruct_h_padded = _reconstruct_waveform_jit(
                self.time, self.popt, omegar_fixed, omegai_fixed, self.N_free, self.N_fix
            )

        # Convert to numpy first, then slice - avoids JAX tracing for different original_len values
        reconstruct_h_np = np.asarray(reconstruct_h_padded)
        hr_np = np.asarray(self.hr)
        hi_np = np.asarray(self.hi)
        
        # Slice to original length for mismatch calculation (numpy slicing, no JAX tracing)
        self.reconstruct_h = reconstruct_h_np[:original_len]
        if self.real:
            h_true_unpadded = hr_np[:original_len]
        else:
            h_true_unpadded = hr_np[:original_len] + 1.j * hi_np[:original_len]
        self.h_true = h_true_unpadded
        self.mismatch = 1 - (np.abs(np.vdot(h_true_unpadded, self.reconstruct_h) / (
            np.linalg.norm(h_true_unpadded) * np.linalg.norm(self.reconstruct_h))))
        
        self.result = QNMFitResult(
            self.popt,
            self.pcov,
            self.mismatch,
            self.cost,
            self.grad,
            self.nfev,
            self.status)
        self.fit_done = True
        
        if return_jcf:
            return None



class QNMFitModel(QNMFitBase):
    """Fit QNM waveform with a parametric frequency model.

    When *model* is ``None`` (default), falls back to the Kerr M/a path.
    For a custom model, pass a :class:`QNMModel` instance together with
    *model_params_guess* and optionally *model_params_bounds*.
    """

    def __init__(
            self,
            h,
            t0,
            qnm_free_list,
            qnm_fixed_list=[],
            retro_def_orbit=True,
            real=False,
            params0=None,
            max_nfev=DEFAULT_MAX_NFEV,
            include_mirror=False,
            iota=None,
            psi=None,
            guess_fixed=[1, 1],
            guess_free=[1, 1],
            guess_M_a=[1, 0.5],
            a_bound=0.99,
            model=None,
            model_params_guess=None,
            model_params_bounds=None,
            **fit_kwargs):
        super().__init__(
            h=h, t0=t0, qnm_fixed_list=qnm_fixed_list,
            N_free=len(qnm_free_list), real=real,
            max_nfev=max_nfev, include_mirror=include_mirror,
            iota=iota, psi=psi, **fit_kwargs)
        self.qnm_free_list = qnm_free_list
        self.params0 = params0
        self.retro_def_orbit = retro_def_orbit
        self.guess_fixed = guess_fixed
        self.guess_free = guess_free
        self.guess_M_a = guess_M_a
        self.a_bound = a_bound
        self.model = model
        self.model_params_guess = model_params_guess
        self.model_params_bounds = model_params_bounds

    # -----------------------------------------------------------------
    # Generalised do_fit for an arbitrary QNMModel
    # -----------------------------------------------------------------

    def _do_fit_model(self):
        """Fit using a user-supplied :class:`QNMModel` with VARPRO.

        Only the model parameters (e.g. M, a, delta) are passed to the
        nonlinear optimiser.  The linear parameters (amplitudes and phases)
        are solved analytically at each iteration via least-squares,
        dramatically reducing the dimensionality of the search.
        """
        model = self.model
        n_model = model.n_params
        self.time, self.hr, self.hi = self.h.postmerger(self.t0)
        time = np.asarray(self.time)
        y = np.asarray(self.hr) + 1j * np.asarray(self.hi)

        # --- Extract or build initial model-parameter guess ---------------
        if hasattr(self.params0, "__iter__"):
            # params0 is a full popt from a previous fit — take model
            # params from the tail.
            model_params0 = np.asarray(self.params0, dtype=float)[-n_model:]
        else:
            model_params0 = np.array(
                [self.model_params_guess[n] for n in model.param_names])

        # --- Build bounds (model parameters only) -------------------------
        default_bounds = model.param_bounds()
        if self.model_params_bounds is not None:
            default_bounds.update(self.model_params_bounds)
        lower, upper = [], []
        for name in model.param_names:
            lo, hi = default_bounds.get(name, (-np.inf, np.inf))
            lower.append(lo)
            upper.append(hi)

        # --- VARPRO: optimise only model params ---------------------------
        sol = scipy_least_squares(
            _varpro_residual_model,
            model_params0,
            args=(time, y, self.qnm_fixed_list, self.qnm_free_list, model),
            bounds=(lower, upper),
            method="trf",
            max_nfev=self.max_nfev,
        )

        # --- Assemble full popt and reconstruct ---------------------------
        self.popt = _varpro_assemble_popt_model(
            sol.x, time, y,
            self.qnm_fixed_list, self.qnm_free_list, model)
        self.pcov = None

        self.reconstruct_h = _varpro_reconstruct_model(
            time, self.popt,
            self.qnm_fixed_list, self.qnm_free_list, model)

        self.h_true = self.hr + 1.j * self.hi
        self.mismatch = 1 - (np.abs(np.vdot(
            self.h_true, self.reconstruct_h) / (
            np.linalg.norm(self.h_true)
            * np.linalg.norm(self.reconstruct_h))))
        self.result = QNMFitResult(self.popt, self.pcov, self.mismatch)
        self.fit_done = True

    # -----------------------------------------------------------------

    def do_fit(self):
        if self.model is not None:
            return self._do_fit_model()

        self.time, self.hr, self.hi = self.h.postmerger(self.t0)
        self._h_interweave = interweave(self.hr, self.hi)
        self._time_interweave = interweave(self.time, self.time)
        if self.real:
            if not hasattr(self.params0, "__iter__"):
                self.params0 = np.array(self.guess_fixed *
                                        self.N_fix +
                                        self.guess_free *
                                        self.N_free +
                                        self.guess_M_a[:1])
            if self.include_mirror:
                fit_func = lambda t, *params: qnm_fit_func_wrapper_varMa_mirror(
                    t, self.qnm_fixed_list, self.qnm_free_list, self.iota, self.psi,
                    self.retro_def_orbit, params, 0, real=True, part="real")
                self.popt, self.pcov = curve_fit(fit_func, np.array(
                    self.time), np.array(
                    self.hr), p0=self.params0, max_nfev=self.max_nfev,
                    method="trf")
                self.reconstruct_h = qnm_fit_func_wrapper_varMa_mirror(
                    self.time, self.qnm_fixed_list, self.qnm_free_list, self.iota, self.psi,
                    self.retro_def_orbit, self.popt,
                    0, real=True, part="real")
            else:
                fit_func = lambda t, *params: qnm_fit_func_wrapper_varMa(
                    t, self.qnm_fixed_list, self.qnm_free_list, self.retro_def_orbit, params, 0, real=True, part="real")
                self.popt, self.pcov = curve_fit(fit_func, np.array(
                    self.time), np.array(
                    self.hr), p0=self.params0, max_nfev=self.max_nfev,
                    method="trf")
                self.reconstruct_h = qnm_fit_func_wrapper_varMa(
                    self.time,
                    self.qnm_fixed_list,
                    self.qnm_free_list,
                    self.retro_def_orbit,
                    self.popt,
                    0,
                    real=True,
                    part="real")
        else:
            if not hasattr(self.params0, "__iter__"):
                self.params0 = np.array(
                    self.guess_fixed *
                    self.N_fix +
                    self.guess_free *
                    self.N_free +
                    self.guess_M_a)
            lower_bound = [-np.inf] * \
                (2 * self.N_fix + 2 * self.N_free + 1) + [-self.a_bound]
            upper_bound = [np.inf] * \
                (2 * self.N_fix + 2 * self.N_free + 1) + [self.a_bound]
            bounds = (np.array(lower_bound), np.array(upper_bound))
            if self.include_mirror:
                fit_func = lambda t, *params: qnm_fit_func_wrapper_complex_varMa_mirror(
                    t, self.qnm_fixed_list, self.qnm_free_list, self.iota, self.psi,
                    self.retro_def_orbit, params)
            else:
                fit_func = lambda t, *params: qnm_fit_func_wrapper_complex_varMa(
                    t, self.qnm_fixed_list, self.qnm_free_list, self.retro_def_orbit, params)
            self.popt, self.pcov = curve_fit(fit_func, np.array(
                self._time_interweave), np.array(
                    self._h_interweave), p0=self.params0,
                bounds=bounds, max_nfev=self.max_nfev,
                method="trf", **self.fit_kwargs)
            if self.include_mirror:
                self.reconstruct_h = qnm_fit_func_wrapper_varMa_mirror(
                    self.time, self.qnm_fixed_list, self.qnm_free_list,
                    self.iota, self.psi,
                    self.retro_def_orbit, self.popt)
            else:
                self.reconstruct_h = qnm_fit_func_wrapper_varMa(
                    self.time, self.qnm_fixed_list, self.qnm_free_list,
                    self.retro_def_orbit, self.popt)
        self.h_true = self.hr + 1.j * self.hi
        self.mismatch = 1 - (np.abs(np.vdot(self.h_true, self.reconstruct_h) / (
            np.linalg.norm(self.h_true) * np.linalg.norm(self.reconstruct_h))))
        self.result = QNMFitResult(self.popt, self.pcov, self.mismatch)
        self.fit_done = True

    def copy_from_result(self, other_result):
        if not self.fit_done:
            self.popt = other_result.popt
            self.pcov = other_result.pcov
            self.time, self.hr, self.hi = self.h.postmerger(self.t0)
            if self.model is not None:
                self.reconstruct_h = _varpro_reconstruct_model(
                    np.asarray(self.time), self.popt,
                    self.qnm_fixed_list, self.qnm_free_list, self.model)
            else:
                self._h_interweave = interweave(self.hr, self.hi)
                self._time_interweave = interweave(self.time, self.time)
                self.reconstruct_h = qnm_fit_func_wrapper(
                    self.time, self.qnm_fixed_list, self.N_free, self.popt)
            self.h_true = self.hr + 1.j * self.hi
            self.mismatch = 1 - (np.abs(np.vdot(self.h_true, self.reconstruct_h) / (
                np.linalg.norm(self.h_true) * np.linalg.norm(self.reconstruct_h))))
            self.result = QNMFitResult(self.popt, self.pcov, self.mismatch)


class QNMFitVarMa(QNMFitModel):
    """Backward-compatible convenience wrapper that uses the Kerr M/a model.

    This is equivalent to ``QNMFitModel`` with ``model=None`` (the default
    Kerr-specific code path).  The ``model``, ``model_params_guess``, and
    ``model_params_bounds`` parameters are **not** accepted here; use
    :class:`QNMFitModel` directly for custom models.
    """

    def __init__(
            self,
            h,
            t0,
            qnm_free_list,
            qnm_fixed_list=[],
            retro_def_orbit=True,
            real=False,
            params0=None,
            max_nfev=DEFAULT_MAX_NFEV,
            include_mirror=False,
            iota=None,
            psi=None,
            guess_fixed=[1, 1],
            guess_free=[1, 1],
            guess_M_a=[1, 0.5],
            a_bound=0.99,
            **fit_kwargs):
        super().__init__(
            h=h, t0=t0, qnm_free_list=qnm_free_list,
            qnm_fixed_list=qnm_fixed_list,
            retro_def_orbit=retro_def_orbit,
            real=real,
            params0=params0, max_nfev=max_nfev,
            include_mirror=include_mirror,
            iota=iota, psi=psi,
            guess_fixed=guess_fixed, guess_free=guess_free,
            guess_M_a=guess_M_a, a_bound=a_bound,
            model=None, model_params_guess=None,
            model_params_bounds=None,
            **fit_kwargs)


def make_initial_guess(
        N_free,
        guess_num,
        A_log_low=-1,
        A_log_hi=1,
        phi_low=0,
        phi_hi=2 * np.pi,
        omega_r_low=-2,
        omega_r_hi=2,
        omega_i_low=0,
        omega_i_hi=-1,
        seed=DEFAULT_SEED,
        A_val=1,
        A_guess_relative=True):
    if not A_guess_relative:
        A_val = 1
    rng = np.random.RandomState(seed)
    A_guesses = A_val * \
        10**(rng.uniform(A_log_low, A_log_hi, size=(guess_num, N_free)))
    phi_guesses = rng.uniform(phi_low, phi_hi, size=(guess_num, N_free))
    omegar_guesses = rng.uniform(
        omega_r_low, omega_r_hi, size=(
            guess_num, N_free))
    omegai_guesses = rng.uniform(
        omega_i_low, omega_i_hi, size=(
            guess_num, N_free))

    guesses_stack = np.empty((guess_num, 4 * N_free), dtype=A_guesses.dtype)
    guesses_stack[:, 0::4] = A_guesses
    guesses_stack[:, 1::4] = phi_guesses
    guesses_stack[:, 2::4] = omegar_guesses
    guesses_stack[:, 3::4] = omegai_guesses

    guess_list = [jnp.array(guess) for guess in guesses_stack]

    return guess_list


class QNMFitVaryingStartingTimeResult:

    def __init__(
            self,
            t0_arr,
            qnm_fixed_list,
            N_free,
            run_string_prefix="Default",
            nonconvergence_cut=False,
            nonconvergence_indx=[],
            initial_num=1,
            include_mirror=False,
            mirror_ratio_list=None,
            iota=None,
            psi=None,
            fit_save_prefix=FIT_SAVE_PATH,
            save_result=True
    ):
        self.t0_arr = t0_arr
        self.qnm_fixed_list = qnm_fixed_list
        self.N_fix = len(self.qnm_fixed_list)
        self.N_free = N_free
        self._popt_full = np.zeros(
            (2 * self.N_fix + 4 * self.N_free, len(self.t0_arr)), dtype=float)
        self.popt_initial = np.zeros(
            (2 * self.N_fix + 4 * self.N_free, initial_num), dtype=float)
        self._mismatch_arr = np.zeros(len(self.t0_arr), dtype=float)
        self.mismatch_initial_arr = np.zeros(initial_num, dtype=float)
        self.cost_arr = np.zeros(len(self.t0_arr), dtype=float)
        self.grad_arr = np.zeros(len(self.t0_arr), dtype=float)
        self.nfev_arr = np.zeros(len(self.t0_arr), dtype=int)
        self.status_arr = np.zeros(len(self.t0_arr), dtype=int)
        self.result_processed = False
        if self.N_fix > 0:
            _qnm_fixed_string_list = sorted(qnms_to_string(qnm_fixed_list))
            self.qnm_fixed_string_ordered = '_'.join(_qnm_fixed_string_list)
            self.run_string = f"{run_string_prefix}_N_{self.N_free}_fix_{self.qnm_fixed_string_ordered}_t0_{t0_arr[0]:.4f}_{t0_arr[-1]:.4f}_{len(t0_arr)}"
        else:
            self.qnm_fixed_string_ordered = ''
            self.run_string = f"{run_string_prefix}_N_{self.N_free}_t0_{t0_arr[0]:.4f}_{t0_arr[-1]:.4f}_{len(t0_arr)}"
        if nonconvergence_cut:
            self.run_string += "_nc"
        self.nonconvergence_indx = nonconvergence_indx
        self.fit_save_prefix = fit_save_prefix
        self.file_path = os.path.join(
            self.fit_save_prefix, f"{self.run_string}_result.pickle")
        self.initila_guess_results = []
        self.include_mirror = include_mirror
        if self.include_mirror:
            self.mirror_ratio_list = mirror_ratio_list
            self.iota = iota
            self.psi = psi
        self.save_results = save_result

    def fill_result(self, i, result):
        self._popt_full[:, i] = result.popt
        self._mismatch_arr[i] = result.mismatch

    def fill_initial_guess(self, i, result):
        self.popt_initial[:, i] = result.popt
        self.mismatch_initial_arr[i] = result.mismatch

    def process_results(self):
        self.popt_full = self._popt_full
        self.mismatch_arr = self._mismatch_arr
        self.A_fix_dict = {}
        self.phi_fix_dict = {}
        self.A_free_dict = {}
        self.phi_free_dict = {}
        self.omega_r_dict = {}
        self.omega_i_dict = {}
        for i in range(0, 2 * self.N_fix, 2):
            self.A_fix_dict[f"A_{self.qnm_fixed_list[i//2].string()}"] = self.popt_full[i]
            self.phi_fix_dict[f"phi_{self.qnm_fixed_list[i//2].string()}"] = self.popt_full[i + 1]
        for i in range(2 * self.N_fix, 2 * self.N_fix + 4 * self.N_free, 4):
            self.A_free_dict[f"A_free_{(i-2*self.N_fix)//4}"] = self.popt_full[i]
            self.phi_free_dict[f"phi_free_{(i-2*self.N_fix)//4}"] = self.popt_full[i + 1]
            self.omega_r_dict[f"omega_r_free_{(i-2*self.N_fix)//4}"] = self.popt_full[i + 2]
            self.omega_i_dict[f"omega_i_free_{(i-2*self.N_fix)//4}"] = self.popt_full[i + 3]
        self.A_dict = {**self.A_fix_dict, **self.A_free_dict}
        self.phi_dict = {**self.phi_fix_dict, **self.phi_free_dict}
        self.results_dict = {
            **self.A_fix_dict,
            **self.A_free_dict,
            **self.phi_fix_dict,
            **self.phi_free_dict,
            **self.omega_r_dict,
            **self.omega_i_dict}
        self.omega_dict = {"real": self.omega_r_dict,
                           "imag": self.omega_i_dict}
        self.result_processed = True
        if self.save_results:
            self.pickle_save()

    def pickle_save(self):
        if not os.path.exists(self.fit_save_prefix):
            os.makedirs(self.fit_save_prefix, exist_ok=True)
        with open(self.file_path, "wb") as f:
            pickle.dump(self, f)

    def pickle_exists(self):
        return os.path.exists(self.file_path)

    def reconstruct_waveform(self, indx, t_arr):
        popt = self.popt_full[:, indx]
        if self.include_mirror:
            Q = qnm_fit_func_mirror_wrapper(
                t_arr,
                self.qnm_fixed_list,
                self.mirror_ratio_list,
                popt,
                part=None)
        else:
            Q = qnm_fit_func_wrapper(
                t_arr, self.qnm_fixed_list, self.N_free, popt, part=None)
        return Q

    def reconstruct_mode_by_mode(self, indx, t_arr):
        Q_fix_list = []
        Q_free_list = []
        popt = self.popt_full[:, indx]
        for j in range(self.N_fix):
            Q = qnm_fit_func_wrapper(
                t_arr, [self.qnm_fixed_list[j]], 0, popt[2 * j:2 * j + 2], part=None)
            Q_fix_list.append(Q)
        for j in range(self.N_free):
            Q = qnm_fit_func_wrapper(
                t_arr, [], 1, popt[2 * self.N_fix + 4 * j:2 * self.N_fix + 4 * j + 4], part=None)
            Q_free_list.append(Q)
        return Q_fix_list, Q_free_list

    def summarize_fixed_mode_flatness(self, **kwargs):
        """Return per-mode flatness summary for fixed-frequency modes.

        This wraps `jaxqualin.selection.summarize_fixed_mode_flatness`.
        """
        from .selection import summarize_fixed_mode_flatness
        return summarize_fixed_mode_flatness(self, **kwargs)

    def fixed_mode_flatness_plot_overlays(self, **kwargs):
        """Return (`bold_dict`, `t_flat_start_dict`) for flatness overlays."""
        from .selection import fixed_mode_flatness_to_plot_overlays
        flatness_summary = self.summarize_fixed_mode_flatness(**kwargs)
        return fixed_mode_flatness_to_plot_overlays(flatness_summary)


class QNMFitVaryingStartingTimeResultModel:

    def __init__(
            self,
            t0_arr,
            qnm_fixed_list,
            qnm_free_list,
            real=False,
            run_string_prefix="Default",
            nonconvergence_cut=False,
            include_mirror=False,
            nonconvergence_indx=[],
            iota=None,
            psi=None,
            fit_save_prefix=FIT_SAVE_PATH,
            save_results=True,
            model=None):
        self.t0_arr = t0_arr
        self.qnm_fixed_list = qnm_fixed_list
        self.qnm_free_list = qnm_free_list
        self.N_fix = len(self.qnm_fixed_list)
        self.N_free = len(qnm_free_list)
        self.real = real
        self.model = model
        if model is not None:
            model_param_len = model.n_params
        elif real:
            model_param_len = 1
        else:
            model_param_len = 2
        self._popt_full = np.zeros(
            (2 * self.N_fix + 2 * self.N_free + model_param_len, len(self.t0_arr)), dtype=float)
        self._mismatch_arr = np.zeros(len(self.t0_arr), dtype=float)
        self.result_processed = False
        _qnm_free_string_list = sorted(qnms_to_string(qnm_fixed_list))
        self.qnm_free_string_ordered = '_'.join(_qnm_free_string_list)
        if self.N_fix > 0:
            _qnm_fixed_string_list = sorted(qnms_to_string(qnm_fixed_list))
            self.qnm_fixed_string_ordered = '_'.join(_qnm_fixed_string_list)
            self.run_string = f"{run_string_prefix}_varMa_free_{self.qnm_free_string_ordered}_fix_{self.qnm_fixed_string_ordered}_t0_{t0_arr[0]:.4f}_{t0_arr[-1]:.4f}_{len(t0_arr)}"
        else:
            self.qnm_fixed_string_ordered = ''
            self.run_string = f"{run_string_prefix}_varMa_free_{self.qnm_free_string_ordered}_t0_{t0_arr[0]:.4f}_{t0_arr[-1]:.4f}_{len(t0_arr)}"
        if nonconvergence_cut:
            self.run_string += "_nc"
        self.nonconvergence_indx = nonconvergence_indx
        self.fit_save_prefix = fit_save_prefix
        self.file_path = os.path.join(
            self.fit_save_prefix, f"{self.run_string}_result.pickle")
        self.include_mirror = include_mirror
        if self.include_mirror:
            self.iota = iota
            self.psi = psi
        self.save_results = save_results

    def fill_result(self, i, result):
        self._popt_full[:, i] = result.popt
        self._mismatch_arr[i] = result.mismatch

    def process_results(self):
        self.popt_full = self._popt_full
        self.mismatch_arr = self._mismatch_arr
        self.A_fix_dict = {}
        self.phi_fix_dict = {}
        self.A_free_dict = {}
        self.phi_free_dict = {}
        self.omega_r_dict = {}
        self.omega_i_dict = {}
        for i in range(0, 2 * self.N_fix, 2):
            self.A_fix_dict[f"A_{self.qnm_fixed_list[i//2].string()}"] = self.popt_full[i]
            self.phi_fix_dict[f"phi_{self.qnm_fixed_list[i//2].string()}"] = self.popt_full[i + 1]
        for i in range(2 * self.N_fix, 2 * self.N_fix + 2 * self.N_free, 2):
            self.A_free_dict[f"A_free_{(i-2*self.N_fix)//2}"] = self.popt_full[i]
            self.phi_free_dict[f"phi_free_{(i-2*self.N_fix)//2}"] = self.popt_full[i + 1]
        j = 2 * self.N_fix + 2 * self.N_free
        if self.model is not None:
            self.model_params_dict = {}
            for k, name in enumerate(self.model.param_names):
                self.model_params_dict[name] = self.popt_full[j + k]
        else:
            M_arr = self.popt_full[j]
            if not self.real:
                a_arr = self.popt_full[j + 1]
            if self.real:
                self.model_params_dict = {"M": M_arr}
            else:
                self.model_params_dict = {"M": M_arr, "a": a_arr}
        # Backward-compatible alias
        self.Ma_dict = self.model_params_dict
        self.A_dict = {**self.A_fix_dict, **self.A_free_dict}
        self.phi_dict = {**self.phi_fix_dict, **self.phi_free_dict}
        self.results_dict = {
            **self.A_fix_dict,
            **self.A_free_dict,
            **self.phi_fix_dict,
            **self.phi_free_dict,
            **self.model_params_dict}
        self.result_processed = True
        if self.save_results:
            self.pickle_save()

    def pickle_save(self):
        if not os.path.exists(self.fit_save_prefix):
            os.makedirs(self.fit_save_prefix, exist_ok=True)
        with open(self.file_path, "wb") as f:
            pickle.dump(self, f)

    def pickle_exists(self):
        return os.path.exists(self.file_path)


# Backward-compatible alias
QNMFitVaryingStartingTimeResultVarMa = QNMFitVaryingStartingTimeResultModel


class QNMFitVaryingStartingTime:
    """
    A class for fitting the postmerger waveform with a varying starting time.

    Attributes:
        t0_arr: array of starting times for fitting. 
        h: waveform object to be fitted. 
        var_M_a: fit for the mass and spin of the black hole. 
        Warning: Not tested yet.
        real: whether to fit a real-valued waveform.
        N_free: number of frequency-free QNMs to include in the model. These
            modes are completely free, i.e. their mode numbers are not fixed
            like those in `qnm_free_list`.
        qnm_fixed_list: list of fixed-frequency QNMs included in the model.
        qnm_free_list: list of free-frequency QNMs of fixed mode numbers to
            include in the model, only used for fitting `M` and `a` when
            `var_M_a = True`.
        N_free: number of free QNMs. 
        run_string_prefix: prefix of the run name for dumping the `pickle`
            file.
        nonconvergence_cut: whether to cut the nonconverged fits.
        nonconvergence_indx: indices of the nonconverged fits. 
        initial_num: number of initial guesses to use for the first starting
            time for frequency-free fits.
        include_mirror: whether to include the mirror modes, for fitting
            waveforms with both waveform polarizations.
        mirror_ratio_list: list of ratios between prograde and mirror mode
            amplitudes.
        iota: inclination angle of the source. 
        psi: polarization angle of the source. 
        save_results: whether to save the results. 
        params0: initial guess for the fit parameters, at least for the
            earliest `t0` fit.
        max_nfev: maximum number of function evaluations for the fit.
        sequential_guess: whether to use the previous fit as the initial
            guess for the next fit.
        load_pickle: whether to load the `pickle` file if it exists.
        fit_save_prefix: prefix of the path to save the `pickle` file. 
        A_bound: maximum value of the amplitude. 
        fit_kwargs: keyword arguments for curve fitting.
        initial_dict: key word arguments for `make_initial_guess` method.
        A_guess_relative: whether to multiply the initial guess of the
            amplitude by the peak strain of the waveform.
        set_seed: random seed for generating the initial guesses. 
        weighted: whether to perform a weighted fit. 
        double_skip: whether to skip the next `2^n` `t0` fits when a fit
            does not converge, where `n` is the number of times the fit did not
            converge consecutively.
        skip_i_init: number of `t0` fits to skip for the first time a
            nonconvergent fit occured.
        result_full: `QNMFitVaryingStartingTimeResult` object for storing
            the fit results.

    Methods:
        get_mirror_ratio_list: get `mirror_ratio_list` from `iota` and
            `psi`.
        initial_guesses: generate initial guesses for the first `t0` fit.
        make_nan_result: generate a `QNMFitVaryingStartingTimeResult` object
        with `nan` values. do_fits: perform the fits.

    """

    t0_arr: np.ndarray
    h: waveform
    var_M_a: bool
    real: bool
    N_free: int
    qnm_fixed_list: List[mode]
    qnm_free_list: List[mode_free]
    N_free: int
    run_string_prefix: str
    nonconvergence_cut: bool
    nonconvergence_indx: List[int]
    initial_num: int
    include_mirror: bool
    mirror_ratio_list: List[float]
    iota: float
    psi: float
    save_results: bool
    params0: np.ndarray
    max_nfev: int
    sequential_guess: bool
    load_pickle: bool
    fit_save_prefix: str
    A_bound: float
    fit_kwargs: Dict[str, Any]
    initial_dict: Dict[str, Any]
    A_guess_relative: bool
    set_seed: int
    weighted: bool
    double_skip: bool
    skip_i_init: int
    result_full: QNMFitVaryingStartingTimeResult

    def __init__(
            self,
            h: waveform,
            t0_arr: np.ndarray,
            N_free: int = 0,
            qnm_fixed_list: List[mode] = [],
            qnm_free_list: List[mode_free] = [],
            var_M_a: bool = False,
            real: bool = False,
            run_string_prefix: str = "Default",
            params0: Optional[np.ndarray] = None,
            max_nfev: int = DEFAULT_MAX_NFEV,
            sequential_guess: bool = True,
            load_pickle: bool = True,
            fit_save_prefix: str = FIT_SAVE_PATH,
            nonconvergence_cut: bool = False,
            A_bound: float = np.inf,
            fit_kwargs: Dict = {},
            initial_num: int = 1,
            random_initial: bool = False,
            initial_dict: Dict = {},
            A_guess_relative: bool = True,
            set_seed: int = DEFAULT_SEED,
            weighted: bool = False,
            double_skip: bool = True,
            include_mirror: bool = False,
            iota: Optional[float] = None,
            psi: Optional[float] = None,
            mirror_ignore_phase: bool = True,
            skip_i_init: int = 1,
            save_results: bool = True,
            fit_config: Optional[FitConfig] = None,
            model=None,
            model_params_guess=None,
            model_params_bounds=None) -> None:
        """
        Initialize the `QNMFitVaryingStartingTime` object.

        Parameters:
            h: waveform object to be fitted.
            t0_arr: array of starting times for fitting.
            N_free: number of frequency-free QNMs to include in the model.
                These modes are completely free, i.e. their mode numbers are not
                fixed like those in `qnm_free_list`.
            qnm_fixed_list: list of fixed-frequency QNMs included in the
                model.
            qnm_free_list: list of free-frequency QNMs of fixed mode numbers
                to include in the model, only used for fitting `M` and `a` when
                `var_M_a = True`.
            var_M_a: fit for the mass and spin of the black hole. Warning:
                Not tested yet.
            real: whether to fit a real-valued waveform.
            run_string_prefix: prefix of the run name for dumping the
                `pickle` file.
            params0: initial guess for the fit parameters, at least for the
                earliest `t0` fit.
            max_nfev: maximum number of function evaluations for the fit.
            sequential_guess: whether to use the previous fit as the initial
                guess for the next fit.
            load_pickle: whether to load the `pickle` file if it exists.
            fit_save_prefix: prefix of the path to save the `pickle` file.
            nonconvergence_cut: whether to cut the nonconverged fits.
            A_bound: maximum value of the amplitude.
            fit_kwargs: keyword arguments for curve fitting.
            initial_num: number of initial guesses to use for the first
                starting time for frequency-free fits.
            random_initial: whether to generate random initial guesses for
                the first starting time for frequency-free fits.
            initial_dict: key word arguments for `make_initial_guess`
                method.
            A_guess_relative: whether to multiply the initial guess of the
                amplitude by the peak strain of the waveform.
            set_seed: random seed for generating the initial guesses.
            weighted: whether to perform a weighted fit.
            double_skip: whether to skip the next `2^n` `t0` fits when a fit
                does not converge, where `n` is the number of times the fit did
                not converge consecutively.
            include_mirror: whether to include the mirror modes, for fitting
                waveforms with both waveform polarizations.
            iota: inclination angle of the source.
            psi: polarization angle of the source.
            mirror_ignore_phase: whether to ignore the phase difference
                between the prograde and mirror modes.
            skip_i_init: number of `t0` fits to skip for the first time a
                nonconvergent fit occured.
            save_results: whether to save the results.
            fit_config: optional FitConfig dataclass. If provided, overrides
                the individual max_nfev, weighted, real,
                include_mirror, iota, and psi parameters.
            model: optional QNMModel instance for custom parametric models.
            model_params_guess: dict of initial guesses for model params.
            model_params_bounds: dict overriding default model param bounds.
        """
        if fit_config is not None:
            self.fit_config = fit_config
        else:
            self.fit_config = FitConfig(
                max_nfev=max_nfev,
                weight_by_amplitude=weighted,
                real=real,
                include_mirror=include_mirror,
                iota=iota,
                psi=psi,
            )
        max_nfev = self.fit_config.max_nfev
        weighted = self.fit_config.weight_by_amplitude
        real = self.fit_config.real
        include_mirror = self.fit_config.include_mirror
        iota = self.fit_config.iota
        psi = self.fit_config.psi

        self.h = h
        if A_guess_relative:
            A_rel = np.abs(h.h[0])
        else:
            A_rel = 1
        self.t0_arr = t0_arr
        self.N_fix = len(qnm_fixed_list)
        self.var_M_a = var_M_a
        if var_M_a:
            warnings.warn(
                "var_M_a is not tested yet, proceed with caution",
                UserWarning)
            self.N_free = len(qnm_free_list)
            self.qnm_free_list = qnm_free_list
        else:
            self.N_free = N_free
        self.qnm_fixed_list = qnm_fixed_list
        self.params0 = params0
        self.max_nfev = max_nfev
        if not hasattr(self.params0, "__iter__"):
            if model is not None:
                guess_model = [model_params_guess[n]
                               for n in model.param_names]
                self.params0 = jnp.array(
                    [A_rel, 1] * self.N_fix + [A_rel, 1] * self.N_free + guess_model)
            elif var_M_a:
                if real:
                    self.params0 = jnp.array(
                        [A_rel, 1] * self.N_fix + [A_rel, 1] * self.N_free + [1])
                else:
                    self.params0 = jnp.array(
                        [A_rel, 1] * self.N_fix + [A_rel, 1] * self.N_free + [1, 0.5])
            else:
                self.params0 = jnp.array(
                    [A_rel, 1] * self.N_fix + [A_rel, 1, 1, -1] * self.N_free)
        self.sequential_guess = sequential_guess
        self.run_string_prefix = run_string_prefix
        self.load_pickle = load_pickle
        self.fit_save_prefix = fit_save_prefix
        self.real = real
        if self.real:
            logger.info("Real-waveform mode enabled.")
        self.nonconvergence_cut = nonconvergence_cut
        self.A_bound = A_bound
        self.fit_kwargs = fit_kwargs
        self.initial_num = initial_num
        self.random_initial = (
            random_initial and not self.var_M_a and self.N_free != 0)
        self.initial_dict = initial_dict
        self.A_guess_relative = A_guess_relative
        self.set_seed = set_seed
        self.weighted = weighted
        self.double_skip = double_skip
        self.include_mirror = include_mirror
        if self.include_mirror and self.N_free != 0:
            raise ValueError(
                "Cannot include mirror if there are free parameters")
        self.iota = iota
        self.psi = psi
        if self.include_mirror and (self.iota is None or self.psi is None):
            raise ValueError(
                "Must specify iota and phi to include mirror mode")
        if self.include_mirror and not self.var_M_a:
            self.mirror_ratio_list = self.get_mirror_ratio_list()
        else:
            self.mirror_ratio_list = None

        self.skip_i_init = skip_i_init
        self.save_results = save_results
        self.model = model
        self.model_params_guess = model_params_guess
        self.model_params_bounds = model_params_bounds
        if self.model is not None:
            self.var_M_a = True

    def get_mirror_ratio_list(self) -> List[float]:
        """
        Get the ratios between the prograde and mirror modes from `iota` and `psi`.

        Returns:
            list of ratios between prograde and mirror mode amplitudes.

        """
        self.mirror_ratio_list = make_mirror_ratio_list(self.qnm_fixed_list, self.iota, psi = self.psi)
        return self.mirror_ratio_list

    def initial_guesses(self) -> Tuple[int, List[QNMFit], List[np.ndarray]]:
        """
        Generate initial guesses for the first `t0` fit.

        Returns:
            best_guess_index: index of the best initial guess.
            qnm_fit_list: list of `QNMFit` objects for the initial guesses.
            guess_list: list of initial guess parameters used.
        """
        A_val = np.abs(self.h.h[0])
        guess_list = make_initial_guess(self.N_free, self.initial_num,
                                        A_guess_relative=self.A_guess_relative,
                                        seed=self.set_seed, A_val=A_val,
                                        **self.initial_dict)
        qnm_fit_list = []
        desc = f"Runname: {self.run_string_prefix}, making initial guesses for N_free = {self.N_free}. Status"
        for j, guess in tqdm(
                enumerate(guess_list), desc=desc, total=len(guess_list), **_tqdm_kwargs()):
            qnm_fit = QNMFit(
                self.h,
                self.t0_arr[0],
                self.N_free,
                qnm_fixed_list=self.qnm_fixed_list,
                real=self.real,
                params0=guess,
                max_nfev=self.max_nfev,
                A_bound=self.A_bound,
                weighted=self.weighted,
                include_mirror=self.include_mirror,
                mirror_ratio_list=self.mirror_ratio_list,
                max_len=self._max_len_for_fit,
                **self.fit_kwargs)
            try:
                qnm_fit.do_fit()
            except RuntimeError:
                logger.warning(f"{j}-th initial guess fit did not reach tolerance.")
                qnm_fit = None
            qnm_fit_list.append(qnm_fit)

        mismatches = []
        for i in range(self.initial_num):
            if qnm_fit_list[i] is None:
                mismatches.append(np.nan)
            else:
                mismatches.append(qnm_fit_list[i].result.mismatch)
        mismatches = np.array(mismatches)
        try:
            best_guess_index = np.nanargmin(mismatches)
        except ValueError:
            best_guess_index = None

        return best_guess_index, qnm_fit_list, guess_list

    def make_nan_result(self) -> None:
        """
        Generate a `QNMFitVaryingStartingTimeResult` object with `nan` values.
        """

        nan_mismatch = np.nan
        if self.var_M_a:
            if self.real:
                nan_popt = np.full(
                    self.N_fix * 2 + self.N_free * 2 + 1, np.nan)
                nan_pcov = nan_popt
            else:
                nan_popt = np.full(
                    self.N_fix * 2 + self.N_free * 2 + 2, np.nan)
                nan_pcov = nan_popt
        else:
            nan_popt = np.full(
                self.N_fix * 2 + self.N_free * 4, np.nan)
            nan_pcov = nan_popt
        nan_cost = np.nan
        nan_grad = np.empty(self.N_fix * 2 + self.N_free * 4)
        nan_grad[:] = np.nan
        nan_nfev = self.max_nfev
        max_status = 0
        nan_result = QNMFitResult(
            nan_popt, nan_pcov, nan_mismatch,
            nan_cost, nan_grad, nan_nfev, max_status)

        return nan_result

    def _run_warmup(self):
        """Handle the warmup fit logic to initialize JAX/optimistix."""
        global _FULL_WARMUP_DONE
        max_len = self._max_len_for_fit

        if not _FULL_WARMUP_DONE and not self.var_M_a:
            n_warmup = min(5, len(self.t0_arr))
            for warmup_idx in range(n_warmup):
                warmup_fit = QNMFit(
                    self.h, self.t0_arr[warmup_idx], self.N_free,
                    qnm_fixed_list=self.qnm_fixed_list,
                    real=self.real,
                    params0=self.params0,
                    max_nfev=self.max_nfev,
                    A_bound=self.A_bound,
                    weighted=self.weighted,
                    include_mirror=self.include_mirror,
                    mirror_ratio_list=self.mirror_ratio_list,
                    max_len=max_len,
                    **self.fit_kwargs)
                warmup_fit.do_fit()
                _ = warmup_fit.popt.block_until_ready()
                if warmup_idx == 0:
                    self._warmup_result = warmup_fit
            _FULL_WARMUP_DONE = True
        elif not self.var_M_a:
            warmup_fit = QNMFit(
                self.h, self.t0_arr[0], self.N_free,
                qnm_fixed_list=self.qnm_fixed_list,
                real=self.real,
                params0=self.params0,
                max_nfev=self.max_nfev,
                A_bound=self.A_bound,
                weighted=self.weighted,
                include_mirror=self.include_mirror,
                mirror_ratio_list=self.mirror_ratio_list,
                max_len=max_len,
                **self.fit_kwargs)
            warmup_fit.do_fit()
            self._warmup_result = warmup_fit

    def _run_fit_at_t0(self, i, _t0):
        """Handle fitting at a single t0 value.

        Uses and updates instance state: _skip_i, _skip_consect,
        _current_params0, _qnm_fit_result_temp, not_converged,
        nonconvergence_indx.
        """
        max_len = self._max_len_for_fit

        if self.var_M_a:
            if self.model is not None:
                qnm_fit = QNMFitModel(
                    self.h,
                    _t0,
                    self.qnm_free_list,
                    qnm_fixed_list=self.qnm_fixed_list,
                    real=self.real,
                    params0=self._current_params0,
                    max_nfev=self.max_nfev,
                    include_mirror=self.include_mirror,
                    iota=self.iota,
                    psi=self.psi,
                    model=self.model,
                    model_params_guess=self.model_params_guess,
                    model_params_bounds=self.model_params_bounds,
                    **self.fit_kwargs)
            else:
                qnm_fit = QNMFitVarMa(
                    self.h,
                    _t0,
                    self.qnm_free_list,
                    qnm_fixed_list=self.qnm_fixed_list,
                    real=self.real,
                    params0=self._current_params0,
                    max_nfev=self.max_nfev,
                    include_mirror=self.include_mirror,
                    iota=self.iota,
                    psi=self.psi,
                    **self.fit_kwargs)
        else:
            qnm_fit = QNMFit(
                self.h,
                _t0,
                self.N_free,
                qnm_fixed_list=self.qnm_fixed_list,
                real=self.real,
                params0=self._current_params0,
                max_nfev=self.max_nfev,
                A_bound=self.A_bound,
                weighted=self.weighted,
                include_mirror=self.include_mirror,
                mirror_ratio_list=self.mirror_ratio_list,
                max_len=max_len,
                **self.fit_kwargs)

        if self.nonconvergence_cut and self.not_converged:
            qnm_fit.copy_from_result(self._qnm_fit_result_temp)
        else:
            try:
                if i == 0 and self.random_initial:
                    if self._initial_converged:
                        qnm_fit = self._qnm_initial_fit_list[self._best_guess_index]
                    else:
                        raise RuntimeError
                elif i == 0 and not self.var_M_a and hasattr(self, '_warmup_result'):
                    qnm_fit = self._warmup_result
                else:
                    if self._skip_consect < self._skip_i and self.double_skip:
                        raise RuntimeError
                    else:
                        qnm_fit.do_fit()
            except RuntimeError:
                if self._skip_consect < self._skip_i:
                    logger.debug(f"skipped t0 = {_t0}.")
                else:
                    logger.debug(
                        f"fit did not reach tolerance at t0 = {_t0}.")
                qnm_fit.result = self.make_nan_result()
                self.nonconvergence_indx.append(i)
                self.not_converged = True
                if self.double_skip:
                    if self._skip_consect >= self._skip_i:
                        self._skip_consect = 0
                        if self._skip_i == 0:
                            self._skip_i = self.skip_i_init
                        else:
                            self._skip_i *= 2
                    self._skip_consect += 1
            else:
                self._skip_consect = 0
                self._skip_i = 0
                if self.sequential_guess:
                    self._current_params0 = qnm_fit.result.popt

        self.result_full.fill_result(i, qnm_fit.result)
        self._qnm_fit_result_temp = qnm_fit.result

    def do_fits(self):
        """
        Perform the fits.

        """

        self._skip_i = 0
        self._skip_consect = 0
        self.not_converged = False
        self.nonconvergence_indx = []
        self._time_longest, _, _ = self.h.postmerger(self.t0_arr[0])
        max_len = len(self._time_longest)
        self._max_len_for_fit = max_len
        if self.var_M_a:
            _ResultClass = (QNMFitVaryingStartingTimeResultModel
                            if self.model is not None
                            else QNMFitVaryingStartingTimeResultVarMa)
            self.result_full = _ResultClass(
                self.t0_arr,
                self.qnm_fixed_list,
                self.qnm_free_list,
                self.real,
                run_string_prefix=self.run_string_prefix,
                nonconvergence_cut=self.nonconvergence_cut,
                include_mirror=self.include_mirror,
                iota=self.iota,
                psi=self.psi,
                fit_save_prefix=self.fit_save_prefix,
                save_results=self.save_results,
                model=self.model)
        else:
            self.result_full = QNMFitVaryingStartingTimeResult(
                self.t0_arr,
                self.qnm_fixed_list,
                self.N_free,
                run_string_prefix=self.run_string_prefix,
                nonconvergence_cut=self.nonconvergence_cut,
                initial_num=self.initial_num,
                include_mirror=self.include_mirror,
                mirror_ratio_list=self.mirror_ratio_list,
                iota=self.iota,
                psi=self.psi,
                fit_save_prefix=self.fit_save_prefix,
                save_result=self.save_results)
        loaded_results = False
        if self.result_full.pickle_exists() and self.load_pickle:
            try:
                _file_path = self.result_full.file_path
                with open(_file_path, "rb") as f:
                    self.result_full = pickle.load(f)
                logger.info(
                    f"Loaded fit {self.result_full.run_string} from an old run.")
                loaded_results = True
            except EOFError:
                logger.warning("EOFError when loading pickle for fit. Doing new fit now...")
                loaded_results = False
        if not loaded_results:
            self._run_warmup()

            self._initial_converged = None
            self._qnm_initial_fit_list = None
            self._best_guess_index = None
            if self.random_initial:
                self._best_guess_index, self._qnm_initial_fit_list, guess_list = self.initial_guesses()
                if self._best_guess_index is None:
                    self._initial_converged = False
                else:
                    self.result_full.guess_list = guess_list
                    for i, qnm_initial_fit in enumerate(self._qnm_initial_fit_list):
                        if qnm_initial_fit is None:
                            fit_result = self.make_nan_result()
                        else:
                            fit_result = qnm_initial_fit.result
                        self.result_full.fill_initial_guess(i, fit_result)
                    self._initial_converged = True

            self._current_params0 = self.params0
            self._qnm_fit_result_temp = None

            if self.N_free == 0:
                desc = f"Runname: {self.run_string_prefix}, fitting with the following modes: "
                mode_string_list = qnms_to_string(self.qnm_fixed_list)
                desc += ', '.join(mode_string_list)
                desc += ". Status"
            elif len(self.qnm_fixed_list) == 0:
                desc = f"Runname: {self.run_string_prefix}, fitting for N_free = {self.N_free}. Status"
            else:
                desc = f"Runname: {self.run_string_prefix}, fitting with the following modes: "
                mode_string_list = qnms_to_string(self.qnm_fixed_list)
                desc += ', '.join(mode_string_list)
                desc += f"and N_free = {self.N_free}. Status"
            for i, _t0 in tqdm(
                enumerate(
                    self.t0_arr), desc=desc, total=len(
                    self.t0_arr), **_tqdm_kwargs()):
                self._run_fit_at_t0(i, _t0)
            self.result_full.nonconvergence_indx = self.nonconvergence_indx
            self.result_full.process_results()


def fit_effective(omega_fund, A_merger, phi_merger, Mf, h):
    t_comp = np.concatenate((h.time, h.time))
    h_comp = np.concatenate((h.hr, h.hi))

    def fit_func(t_comp, c2, c3, d3, d4): return \
        effective_ringdown_for_fit(
            omega_fund, A_merger, phi_merger, Mf, t_comp, c2, c3, d3, d4)
    popt, pcov = curve_fit(fit_func, t_comp, h_comp, maxfev=10000)
    return popt, pcov


def effective_ringdown(
        omega_fund,
        A_merger,
        phi_merger,
        Mf,
        t,
        c2,
        c3,
        d3,
        d4,
        part="complex"):
    c1 = -A_merger * np.imag(omega_fund) * np.cosh(c3)**2 / c2
    c4 = A_merger - c1 * np.tanh(c3)
    d2 = 2 * c2
    d1 = Mf * (1 + d3 + d4) / (d2 * (d3 + 2 * d4)) * \
        (np.real(omega_fund) - phi_merger)
    A = c1 * np.tanh(c2 * t + c3) + c4
    phi = - d1 * np.log((1 + d3 * np.exp(-d2 * t) + d4 *
                        np.exp(-2 * d2 * t)) / (1 + d3 + d4))
    if part == "complex":
        return A * np.exp(1.j * phi) * np.exp(-1.j *
                                              (omega_fund * t + phi_merger))
    elif part == "real":
        return np.real(A * np.exp(1.j * phi) *
                       np.exp(-1.j * (omega_fund * t + phi_merger)))
    elif part == "imag":
        return np.imag(A * np.exp(1.j * phi) *
                       np.exp(-1.j * (omega_fund * t + phi_merger)))
    else:
        raise ValueError("part must be complex, real or imag")
        return


def effective_ringdown_for_fit(
        omega_fund,
        A_merger,
        phi_merger,
        Mf,
        t_comp,
        c2,
        c3,
        d3,
        d4):
    fit_params = (c2, c3, d3, d4)
    N = int(len(t_comp) / 2)
    h_real = effective_ringdown(omega_fund,
                                A_merger,
                                phi_merger,
                                Mf,
                                t_comp[:N],
                                *fit_params,
                                part="real")
    h_imag = effective_ringdown(omega_fund,
                                A_merger,
                                phi_merger,
                                Mf,
                                t_comp[N:],
                                *fit_params,
                                part="imag")
    h_comp = np.concatenate((h_real, h_imag))
    return h_comp


def fit_effective_2(h, A_fund, phi_fund, omega_fund, t_match):
    t_comp = np.concatenate((h.time, h.time))
    h_comp = np.concatenate((h.hr, h.hi))

    def fit_func(t_comp, c1, c2, d1, d2): return \
        effective_ringdown_for_fit_2(
            t_comp, A_fund, phi_fund, omega_fund, t_match, c1, c2, d1, d2)
    popt, pcov = curve_fit(fit_func, t_comp, h_comp, maxfev=1000000, bounds=(
        [-np.inf, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
    return popt, pcov


def effective_ringdown_2(
        t,
        A_fund,
        phi_fund,
        omega_fund,
        t_match,
        c1,
        c2,
        d1,
        d2,
        part="complex"):
    A = -c1 * (np.tanh((t - t_match) / c2) - 1) / 2 + A_fund
    # d1*np.log(1+d2*np.exp(-d3*(t-t_match)))
    phi = phi_fund - d1 * (np.tanh((t - t_match) / d2) - 1) / 2
    if part == "complex":
        return A * np.exp(-1.j * (omega_fund * t + phi))
    elif part == "real":
        return np.real(A * np.exp(-1.j * (omega_fund * t + phi)))
    elif part == "imag":
        return np.imag(A * np.exp(-1.j * (omega_fund * t + phi)))
    else:
        raise ValueError("part must be complex, real or imag")
        return


def effective_ringdown_for_fit_2(
        t_comp,
        A_fund,
        phi_fund,
        omega_fund,
        t_match,
        c1,
        c2,
        d1,
        d2):
    fit_params = (c1, c2, d1, d2)
    N = int(len(t_comp) / 2)
    h_real = effective_ringdown_2(
        t_comp[:N], A_fund, phi_fund, omega_fund, t_match, *fit_params, part="real")
    h_imag = effective_ringdown_2(
        t_comp[N:], A_fund, phi_fund, omega_fund, t_match, *fit_params, part="imag")
    h_comp = np.concatenate((h_real, h_imag))
    return h_comp


def estimate_mass_and_spin(Psi, qnm_free_list,
                           run_string_prefix,
                           tstart=30,
                           tend=50,
                           one_t=False,
                           gamma=None,
                           gamma_scale=False,
                           real=False,
                           qnm_fixed_list=[],
                           t0_arr=np.linspace(0, 100, num=51),
                           load_pickle=True,
                           fit_save_prefix=FIT_SAVE_PATH):

    qnm_fitter = QNMFitVaryingStartingTime(Psi,
                                           t0_arr,
                                           qnm_fixed_list=qnm_fixed_list,
                                           qnm_free_list=qnm_free_list,
                                           real=real,
                                           run_string_prefix=run_string_prefix,
                                           var_M_a=True,
                                           load_pickle=load_pickle,
                                           fit_save_prefix=fit_save_prefix)

    qnm_fitter.do_fits()

    M = qnm_fitter.result_full.Ma_dict["M"]
    tstartindx = bisect_left(t0_arr, tstart)
    tendindx = bisect_left(t0_arr, tend)
    if one_t:
        M_mean = M[tstartindx]
        M_std = 0
    else:
        M_win = M[tstartindx:tendindx]
        M_mean = np.mean(M_win)
        M_std = np.std(M_win)

    if not real:
        a = qnm_fitter.result_full.Ma_dict["a"]
        if one_t:
            a_mean = a[tstartindx]
            a_std = 0
        else:
            a_win = a[tstartindx:tendindx]
            a_mean = np.mean(a_win)
            a_std = np.std(a_win)
        return M_mean, M_std, a_mean, a_std

    return M_mean, M_std
