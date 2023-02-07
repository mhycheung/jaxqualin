import numpy as np
import jax.numpy as jnp
from jaxfit import CurveFit
import scipy
from scipy.optimize import curve_fit
from utils import *
from QuasinormalMode import *
from tqdm import tqdm
import os
import pickle
from copy import copy

from jax.config import config
config.update("jax_enable_x64", True)

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
FIT_SAVE_PATH = os.path.join(ROOT_PATH, "pickle/fits")


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
        retro = False,
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
    for free_mode_params, qnm_free in zip(free_mode_params_list, qnm_free_list):
        A, phi = tuple(free_mode_params)
        qnm_free.fix_mode(M, a, retro = retro)
        omegar = qnm_free.omegar
        omegai = qnm_free.omegai
        if part is None:
            Q += A * np.exp(-1.j * ((omegar + 1.j * omegai) * t + phi))
        elif part == "real":
            Q += A * np.exp(omegai * t) * np.cos(omegar * t + phi)
        elif part == "imag":
            Q += -A * np.exp(omegai * t) * np.sin(omegar * t + phi)
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
        A = args[0][2 * N_fix + 4 * j]
        phi = args[0][2 * N_fix + 4 * j + 1]
        omegar = args[0][2 * N_fix + 4 * j + 2]
        omegai = args[0][2 * N_fix + 4 * j + 3]
        free_mode_params_list.append([A, phi, omegar, omegai])
    return qnm_fit_func(t, qnm_fixed_list, fix_mode_params_list,
                        free_mode_params_list, part=part)

def qnm_fit_func_wrapper_varMa(t, qnm_fixed_list, qnm_free_list, retro, *args, Schwarzschild = False, part=None):
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
    if Schwarzschild:
        return qnm_fit_func_varMa(t, qnm_fixed_list, qnm_free_list, fix_mode_params_list,
                        free_mode_params_list, M, 0., retro = retro, part=part)
    else:
        a = args[0][2 * (N_fix + N_free) + 1] 
        return qnm_fit_func_varMa(t, qnm_fixed_list, qnm_free_list, fix_mode_params_list,
                            free_mode_params_list, M, a, retro = retro, part=part)

# https://stackoverflow.com/questions/50203879/curve-fitting-of-complex-data


def qnm_fit_func_wrapper_complex(t, qnm_fixed_list, N_free, *args, Schwarzschild = False):
    N = len(t)
    t_real = t[0::2]
    t_imag = t[1::2]
    h_real = qnm_fit_func_wrapper(
        t_real, qnm_fixed_list, N_free, *args, part="real")
    if Schwarzschild:
        h_imag = jnp.zeros(int(N/2))
    else:
        h_imag = qnm_fit_func_wrapper(
            t_imag, qnm_fixed_list, N_free, *args, part="imag")
    h_riffle = interweave(h_real, h_imag)
    return h_riffle

def qnm_fit_func_wrapper_complex_varMa(t, qnm_fixed_list, qnm_free_list, retro, *args):
    N = len(t)
    t_real = t[0::2]
    t_imag = t[1::2]
    h_real = qnm_fit_func_wrapper_varMa(
        t_real, qnm_fixed_list, qnm_free_list, retro, *args, part="real")
    h_imag = qnm_fit_func_wrapper_varMa(
        t_imag, qnm_fixed_list, qnm_free_list, retro, *args, part="imag")
    h_riffle = interweave(h_real, h_imag)
    return h_riffle


class QNMFitResult:

    def __init__(self, popt, pcov, mismatch):
        self.popt = popt
        self.pcov = pcov
        self.mismatch = mismatch


class QNMFit:

    def __init__(
            self,
            h,
            t0,
            N_free,
            qnm_fixed_list=[],
            Schwarzschild = False,
            jcf=CurveFit(),
            params0=None,
            max_nfev=1000000,
            A_bound = np.inf):
        self.h = h
        self.t0 = t0
        self.N_free = N_free
        self.qnm_fixed_list = qnm_fixed_list
        self.params0 = params0
        self.N_fix = len(qnm_fixed_list)
        self.Schwarzschild = Schwarzschild
        self.jcf = jcf
        self.max_nfev = max_nfev
        self.fit_done = False
        self.A_bound = A_bound

    def do_fit(self):
        self.time, self.hr, self.hi = self.h.postmerger(self.t0)
        self._h_interweave = interweave(self.hr, self.hi)
        self._time_interweave = interweave(self.time, self.time)
        if not hasattr(self.params0, "__iter__"):
            self.params0 = jnp.array(
                [1, 1] * self.N_fix + [1, 1, 1, -1] * self.N_free)
        upper_bound = [self.A_bound, np.inf] * self.N_fix + ([self.A_bound] + 3 * [np.inf]) * self.N_free
        lower_bound = [-self.A_bound, -np.inf] * self.N_fix + ([-self.A_bound] + 3 * [-np.inf]) * self.N_free
        bounds = (np.array(lower_bound), np.array(upper_bound))
        self.popt, self.pcov = self.jcf.curve_fit(
        # self.popt, self.pcov = scipy.optimize.curve_fit(
            lambda t, *params: qnm_fit_func_wrapper_complex(
                t, self.qnm_fixed_list, self.N_free, params, Schwarzschild = self.Schwarzschild), np.array(
                self._time_interweave), np.array(
                self._h_interweave), bounds = bounds, p0=self.params0, max_nfev=self.max_nfev,
                method = "trf")
        if self.Schwarzschild:
            self.reconstruct_h = qnm_fit_func_wrapper(
                self.time, self.qnm_fixed_list, self.N_free, self.popt, part = "real")
        else:
            self.reconstruct_h = qnm_fit_func_wrapper(
                self.time, self.qnm_fixed_list, self.N_free, self.popt)
        self.h_true = self.hr + 1.j * self.hi
        self.mismatch = 1 - (np.abs(np.vdot(self.h_true, self.reconstruct_h) / (
            np.linalg.norm(self.h_true) * np.linalg.norm(self.reconstruct_h))))
        self.result = QNMFitResult(self.popt, self.pcov, self.mismatch)
        self.fit_done = True
        
    def copy_from_result(self, other_result):
        if self.fit_done == False:
            self.popt = other_result.popt
            self.pcov = other_result.pcov
            self.time, self.hr, self.hi = self.h.postmerger(self.t0)
            self._h_interweave = interweave(self.hr, self.hi)
            self._time_interweave = interweave(self.time, self.time)
            self.reconstruct_h = qnm_fit_func_wrapper(
                self.time, self.qnm_fixed_list, self.N_free, self.popt)
            self.h_true = self.hr + 1.j * self.hi
            self.mismatch = 1 - (np.abs(np.vdot(self.h_true, self.reconstruct_h) / (
                np.linalg.norm(self.h_true) * np.linalg.norm(self.reconstruct_h))))
            self.result = QNMFitResult(self.popt, self.pcov, self.mismatch)
            
            
class QNMFitVarMa:

    def __init__(
            self,
            h,
            t0,
            qnm_free_list,
            qnm_fixed_list=[],
            retro = False,
            Schwarzschild = False,
            jcf=CurveFit(),
            params0=None,
            max_nfev=1000000):
        self.h = h
        self.t0 = t0
        self.N_free = len(qnm_free_list)
        self.qnm_free_list = qnm_free_list
        self.qnm_fixed_list = qnm_fixed_list
        self.params0 = params0
        self.N_fix = len(qnm_fixed_list)
        # self.jcf = jcf
        self.max_nfev = max_nfev
        self.fit_done = False
        self.retro = retro
        self.Schwarzschild = Schwarzschild

    def do_fit(self):
        self.time, self.hr, self.hi = self.h.postmerger(self.t0)
        self._h_interweave = interweave(self.hr, self.hi)
        self._time_interweave = interweave(self.time, self.time)
        if self.Schwarzschild:
            if not hasattr(self.params0, "__iter__"):
                self.params0 = np.array(
                    [1, 1] * self.N_fix + [1, 1] * self.N_free + [1])
            fit_func = lambda t, *params: qnm_fit_func_wrapper_varMa(
            t, self.qnm_fixed_list, self.qnm_free_list, self.retro, params, 0, Schwarzschild = True, part = "real")
            self.popt, self.pcov = curve_fit(fit_func, np.array(
                    self.time), np.array(
                    self.hr), p0=self.params0, max_nfev=self.max_nfev,
                    method = "trf")
            self.reconstruct_h = qnm_fit_func_wrapper_varMa(
                self.time, self.qnm_fixed_list, self.qnm_free_list, self.retro, self.popt,
                0, Schwarzschild = True, part = "real")
        else:
            if not hasattr(self.params0, "__iter__"):
                self.params0 = np.array(
                    [1, 1] * self.N_fix + [1, 1] * self.N_free + [1, 0.5])
            lower_bound = [-np.inf] * (2 * self.N_fix + 2 * self.N_free + 1) + [-0.99]
            upper_bound = [np.inf] * (2 * self.N_fix + 2 * self.N_free + 1) + [0.99]
            bounds = (np.array(lower_bound), np.array(upper_bound))
            fit_func = lambda t, *params: qnm_fit_func_wrapper_complex_varMa(
            t, self.qnm_fixed_list, self.qnm_free_list, self.retro, params)
            self.popt, self.pcov = curve_fit(fit_func, np.array(
                    self._time_interweave), np.array(
                    self._h_interweave), p0=self.params0, 
                    bounds = bounds, max_nfev=self.max_nfev,
                    method = "trf")
            self.reconstruct_h = qnm_fit_func_wrapper_varMa(
                self.time, self.qnm_fixed_list, self.qnm_free_list, self.retro, self.popt)
        self.h_true = self.hr + 1.j * self.hi
        self.mismatch = 1 - (np.abs(np.vdot(self.h_true, self.reconstruct_h) / (
            np.linalg.norm(self.h_true) * np.linalg.norm(self.reconstruct_h))))
        self.result = QNMFitResult(self.popt, self.pcov, self.mismatch)
        self.fit_done = True
        
    def copy_from_result(self, other_result):
        if self.fit_done == False:
            self.popt = other_result.popt
            self.pcov = other_result.pcov
            self.time, self.hr, self.hi = self.h.postmerger(self.t0)
            self._h_interweave = interweave(self.hr, self.hi)
            self._time_interweave = interweave(self.time, self.time)
            self.reconstruct_h = qnm_fit_func_wrapper(
                self.time, self.qnm_fixed_list, self.N_free, self.popt)
            self.h_true = self.hr + 1.j * self.hi
            self.mismatch = 1 - (np.abs(np.vdot(self.h_true, self.reconstruct_h) / (
                np.linalg.norm(self.h_true) * np.linalg.norm(self.reconstruct_h))))
            self.result = QNMFitResult(self.popt, self.pcov, self.mismatch)
        


class QNMFitVaryingStartingTimeResult:

    def __init__(
            self,
            t0_arr,
            qnm_fixed_list,
            N_free,
            run_string_prefix="Default",
            nonconvergence_cut = False,
            nonconvergence_indx = []):
        self.t0_arr = t0_arr
        self.qnm_fixed_list = qnm_fixed_list
        self.N_fix = len(self.qnm_fixed_list)
        self.N_free = N_free
        self._popt_full = np.zeros(
            (2 * self.N_fix + 4 * self.N_free, len(self.t0_arr)), dtype=float)
        self._mismatch_arr = np.zeros(len(self.t0_arr), dtype=float)
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
        self.file_path = os.path.join(FIT_SAVE_PATH, f"{self.run_string}_result.pickle")

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
        self.pickle_save()

    def pickle_save(self):
        with open(self.file_path, "wb") as f:
            pickle.dump(self, f)

    def pickle_exists(self):
        return os.path.exists(self.file_path)
    
    def reconstruct_waveform(self, indx, t_arr):
        popt = self.popt_full[:, indx]
        Q = qnm_fit_func_wrapper(t_arr, self.qnm_fixed_list, self.N_free, popt, part = None)
        return Q
    
    def reconstruct_mode_by_mode(self, indx, t_arr):
        Q_fix_list = []
        Q_free_list = []
        popt = self.popt_full[:, indx]
        for j in range(self.N_fix):
            Q = qnm_fit_func_wrapper(t_arr, [self.qnm_fixed_list[j]], 0, popt[2*j:2*j+2], part = None)
            Q_fix_list.append(Q)
        for j in range(self.N_free):
            Q = qnm_fit_func_wrapper(t_arr, [], 1, popt[2 * self.N_fix + 4*j:2 * self.N_fix + 4*j + 4], part = None)
            Q_free_list.append(Q)
        return Q_fix_list, Q_free_list
    
class QNMFitVaryingStartingTimeResultVarMa:

    def __init__(
            self,
            t0_arr,
            qnm_fixed_list,
            qnm_free_list,
            Schwarzschild = False,
            run_string_prefix="Default",
            nonconvergence_cut = False,
            nonconvergence_indx = []):
        self.t0_arr = t0_arr
        self.qnm_fixed_list = qnm_fixed_list
        self.qnm_free_list = qnm_free_list
        self.N_fix = len(self.qnm_fixed_list)
        self.N_free = len(self.qnm_free_list)
        self.Schwarzschild = Schwarzschild
        if Schwarzschild:
            M_a_len = 1
        else:
            M_a_len = 2
        self._popt_full = np.zeros(
            (2 * self.N_fix + 2 * self.N_free + M_a_len, len(self.t0_arr)), dtype=float)
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
        self.file_path = os.path.join(FIT_SAVE_PATH, f"{self.run_string}_result.pickle")

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
        M_arr = self.popt_full[j]
        if not self.Schwarzschild:
            a_arr = self.popt_full[j + 1]
        self.A_dict = {**self.A_fix_dict, **self.A_free_dict}
        self.phi_dict = {**self.phi_fix_dict, **self.phi_free_dict}
        if self.Schwarzschild:
            self.Ma_dict = {"M": M_arr}
        else:
            self.Ma_dict = {"M": M_arr, "a": a_arr}
        self.results_dict = {
            **self.A_fix_dict,
            **self.A_free_dict,
            **self.phi_fix_dict,
            **self.phi_free_dict,
            **self.Ma_dict}
        self.result_processed = True
        self.pickle_save()

    def pickle_save(self):
        with open(self.file_path, "wb") as f:
            pickle.dump(self, f)

    def pickle_exists(self):
        return os.path.exists(self.file_path)

class QNMFitVaryingStartingTime:

    def __init__(
            self,
            h,
            t0_arr,
            N_free = 0,
            qnm_fixed_list=[],
            qnm_free_list=[],
            var_M_a = False,
            retro = False,
            Schwarzschild = False,
            run_string_prefix="Default",
            params0=None,
            max_nfev=1000000,
            sequential_guess=True,
            load_pickle = True,
            nonconvergence_cut = False,
            A_bound = np.inf):
        self.h = h
        self.t0_arr = t0_arr
        self.N_fix = len(qnm_fixed_list)
        self.var_M_a = var_M_a
        if var_M_a:
            self.N_free = len(qnm_free_list)
            self.qnm_free_list = qnm_free_list
        else:
            self.N_free = N_free
        self.qnm_fixed_list = qnm_fixed_list
        self.params0 = params0
        self.max_nfev = max_nfev
        if not hasattr(self.params0, "__iter__"):
            if var_M_a:
                if Schwarzschild:
                    self.params0 = jnp.array(
                        [1, 1] * self.N_fix + [1, 1] * self.N_free + [1])
                else:
                    self.params0 = jnp.array(
                        [1, 1] * self.N_fix + [1, 1] * self.N_free + [1, 0.5])
            else:
                self.params0 = jnp.array(
                    [1, 1] * self.N_fix + [1, 1, 1, -1] * self.N_free)
        self.sequential_guess = sequential_guess
        self.run_string_prefix = run_string_prefix
        self.load_pickle = load_pickle
        self.retro = retro
        self.Schwarzschild = Schwarzschild
        self.nonconvergence_cut = nonconvergence_cut
        self.A_bound = A_bound

    def do_fits(self):
        self.not_converged = False
        self.nonconvergence_indx = []
        self._time_longest, _, _ = self.h.postmerger(self.t0_arr[0])
        _jcf = CurveFit(flength=2 * len(self._time_longest))
        if self.var_M_a:
            self.result_full = QNMFitVaryingStartingTimeResultVarMa(
                self.t0_arr,
                self.qnm_fixed_list,
                self.qnm_free_list,
                self.Schwarzschild,
                run_string_prefix=self.run_string_prefix,
                nonconvergence_cut = self.nonconvergence_cut)
        else:
            self.result_full = QNMFitVaryingStartingTimeResult(
                self.t0_arr,
                self.qnm_fixed_list,
                self.N_free,
                run_string_prefix=self.run_string_prefix,
                nonconvergence_cut = self.nonconvergence_cut)
        if self.result_full.pickle_exists() and self.load_pickle:
            _file_path = self.result_full.file_path
            with open(_file_path, "rb") as f:
                self.result_full = pickle.load(f)
            print(
                f"reloaded fit {self.result_full.run_string} from an old run.")
        else:
            _params0 = self.params0
            for i, _t0 in tqdm(enumerate(self.t0_arr)):
                if self.var_M_a:
                    qnm_fit = QNMFitVarMa(
                        self.h,
                        _t0,
                        self.qnm_free_list,
                        qnm_fixed_list=self.qnm_fixed_list,
                        Schwarzschild = self.Schwarzschild,
                        jcf=_jcf,
                        params0=_params0,
                        max_nfev=self.max_nfev)
                else:
                    qnm_fit = QNMFit(
                        self.h,
                        _t0,
                        self.N_free,
                        qnm_fixed_list=self.qnm_fixed_list,
                        Schwarzschild = self.Schwarzschild,
                        jcf=_jcf,
                        params0=_params0,
                        max_nfev=self.max_nfev,
                        A_bound = self.A_bound)
                if self.nonconvergence_cut and self.not_converged:
                    qnm_fit.copy_from_result(qnm_fit_result_temp)
                else:
                    try:
                        qnm_fit.do_fit()
                    except RuntimeError:
                        print(f"fit did not reach tolerance at t0 = {_t0}.")
                        qnm_fit.copy_from_result(qnm_fit_result_temp)
                        self.nonconvergence_indx.append(i)
                        self.not_converged = True
                self.result_full.fill_result(i, qnm_fit.result)
                qnm_fit_result_temp = qnm_fit.result
                if self.sequential_guess:
                    _params0 = qnm_fit.result.popt
            self.result_full.nonconvergence_indx = self.nonconvergence_indx
            self.result_full.process_results()
