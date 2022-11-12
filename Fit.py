import numpy as np
import jax.numpy as jnp
from jaxfit import CurveFit
import scipy
from utils import *
from QuasinormalMode import *
from tqdm import tqdm
import os
import pickle
from copy import copy

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

# https://stackoverflow.com/questions/50203879/curve-fitting-of-complex-data


def qnm_fit_func_wrapper_complex(t, qnm_fixed_list, N_free, *args):
    N = len(t)
    t_real = t[0::2]
    t_imag = t[1::2]
    h_real = qnm_fit_func_wrapper(
        t_real, qnm_fixed_list, N_free, *args, part="real")
    h_imag = qnm_fit_func_wrapper(
        t_imag, qnm_fixed_list, N_free, *args, part="imag")
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
            jcf=CurveFit(),
            params0=None,
            max_nfev=100000):
        self.h = h
        self.t0 = t0
        self.N_free = N_free
        self.qnm_fixed_list = qnm_fixed_list
        self.params0 = params0
        self.N_fix = len(qnm_fixed_list)
        self.jcf = jcf
        self.max_nfev = max_nfev
        self.fit_done = False

    def do_fit(self):
        self.time, self.hr, self.hi = self.h.postmerger(self.t0)
        self._h_interweave = interweave(self.hr, self.hi)
        self._time_interweave = interweave(self.time, self.time)
        if not hasattr(self.params0, "__iter__"):
            self.params0 = jnp.array(
                [1, 1] * self.N_fix + [1, 1, 1, -1] * self.N_free)

        self.popt, self.pcov = self.jcf.curve_fit(
        # self.popt, self.pcov = scipy.optimize.curve_fit(
            lambda t, *params: qnm_fit_func_wrapper_complex(
                t, self.qnm_fixed_list, self.N_free, params), np.array(
                self._time_interweave), np.array(
                self._h_interweave), p0=self.params0, max_nfev=self.max_nfev,
                method = "trf")
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


class QNMFitVaryingStartingTimeResult:

    def __init__(
            self,
            t0_arr,
            qnm_fixed_list,
            N_free,
            run_string_prefix="Default"):
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


class QNMFitVaryingStartingTime:

    def __init__(
            self,
            h,
            t0_arr,
            N_free,
            qnm_fixed_list=[],
            run_string_prefix="Default",
            params0=None,
            max_nfev=100000,
            sequential_guess=True):
        self.h = h
        self.t0_arr = t0_arr
        self.N_fix = len(qnm_fixed_list)
        self.N_free = N_free
        self.qnm_fixed_list = qnm_fixed_list
        self.params0 = params0
        self.max_nfev = max_nfev
        if not hasattr(self.params0, "__iter__"):
            self.params0 = jnp.array(
                [1, 1] * self.N_fix + [1, 1, 1, -1] * self.N_free)
        self.sequential_guess = sequential_guess
        self.run_string_prefix = run_string_prefix

    def do_fits(self):
        self._time_longest, _, _ = self.h.postmerger(self.t0_arr[0])
        _jcf = CurveFit(flength=2 * len(self._time_longest))
        self.result_full = QNMFitVaryingStartingTimeResult(
            self.t0_arr,
            self.qnm_fixed_list,
            self.N_free,
            run_string_prefix=self.run_string_prefix)
        if self.result_full.pickle_exists():
            _file_path = self.result_full.file_path
            with open(_file_path, "rb") as f:
                self.result_full = pickle.load(f)
            print(
                f"reloaded fit {self.result_full.run_string} from an old run.")
        else:
            _params0 = self.params0
            for i, _t0 in tqdm(enumerate(self.t0_arr)):
                qnm_fit = QNMFit(
                    self.h,
                    _t0,
                    self.N_free,
                    qnm_fixed_list=self.qnm_fixed_list,
                    jcf=_jcf,
                    params0=_params0,
                    max_nfev=self.max_nfev)
                try:
                    qnm_fit.do_fit()
                except RuntimeError:
                    print(f"fit did not reach tolerance at t0 = {_t0}.")
                    qnm_fit.copy_from_result(qnm_fit_result_temp)
                self.result_full.fill_result(i, qnm_fit.result)
                qnm_fit_result_temp = qnm_fit.result
                if self.sequential_guess:
                    _params0 = qnm_fit.result.popt
            self.result_full.process_results()
