from .utils import max_consecutive_trues
from .qnmode import mode, potential_modes, qnms_to_string, remove_duplicated_modes, lower_overtone_present, lower_l_mode_present
from .fit import QNMFitVaryingStartingTime, FIT_SAVE_PATH, DEFAULT_SEED
from .waveforms import waveform, get_waveform_SXS, get_relevant_lm_waveforms_SXS, relevant_modes_dict_to_lm_tuple, make_eff_ringdown_waveform_from_param, DEFAULT_T_END

import logging
import numpy as np
import pickle
import json
import os

from typing import List, Tuple, Union, Optional, Dict, Any

logger = logging.getLogger(__name__)

SETTING_PATH = os.getcwd()
MODE_SEARCHERS_SAVE_PATH = os.path.join(
    os.getcwd(), ".jaxqualin_cache/mode_searchers")

DEFAULT_N_LIST = [5, 6, 7, 8, 9, 10]
DEFAULT_ALPHA = 0.05
DEFAULT_TAU_AGNOSTIC = 10
DEFAULT_P_AGNOSTIC = 0.95
DEFAULT_EPSILON_STABLE = 0.2
DEFAULT_TAU_STABLE = 10
DEFAULT_P_STABLE = 0.95
DEFAULT_A_TOL = 1e-3
DEFAULT_BETA_A = 1.0
DEFAULT_BETA_PHI = 1.5


def _merge_kwargs(defaults, overrides):
    """Merge default kwargs with user overrides."""
    result = dict(defaults)
    result.update(overrides)
    return result


def _check_no_cce(cce_flag):
    """Raise if CCE mode is requested (not yet supported)."""
    if cce_flag:
        raise NotImplementedError("CCE is not yet supported")


class IterativeFlatnessChecker:

    def __init__(self, h, t0_arr, M, a, l, m, found_modes, **kwargs_in):
        self.h = h
        self.t0_arr = t0_arr
        self.l = l
        self.m = m
        self.M = M
        self.a = a
        self.found_modes = found_modes
        self.fitter_list = []
        self.kwargs = _merge_kwargs({
            "run_string_prefix": "Default",
            "epsilon_stable": DEFAULT_EPSILON_STABLE,
            "tau_stable": DEFAULT_TAU_STABLE,
            "retro_def_orbit": True,
            "load_pickle": True,
            "confusion_tol": 0.03,
            "p_stable": DEFAULT_P_STABLE,
            "A_tol": DEFAULT_A_TOL,
            "beta_A": DEFAULT_BETA_A,
            "beta_phi": DEFAULT_BETA_PHI,
            "CCE": False,
            "fit_save_prefix": FIT_SAVE_PATH}, kwargs_in)
        self.run_string_prefix = self.kwargs["run_string_prefix"]
        self.epsilon_stable = self.kwargs["epsilon_stable"]
        self.confusion_tol = self.kwargs["confusion_tol"]
        self.tau_stable = self.kwargs["tau_stable"]
        self.tau_stable_length = int(
            self.tau_stable / (self.t0_arr[1] - self.t0_arr[0]) + 1)
        self.p_stable = self.kwargs["p_stable"]
        self.A_tol = self.kwargs["A_tol"]
        self.beta_A = self.kwargs["beta_A"]
        self.beta_phi = self.kwargs["beta_phi"]
        self.CCE = self.kwargs["CCE"]
        _check_no_cce(self.CCE)
        self.fit_save_prefix = self.kwargs["fit_save_prefix"]

        self.retro_def_orbit = self.kwargs["retro_def_orbit"]
        self.load_pickle = self.kwargs["load_pickle"]

    def do_iterative_flatness_check(self):

        if self.retro_def_orbit and self.a < 0:
            fund_mode_string = f"-{self.l}.{self.m}.0"
        else:
            fund_mode_string = f"{self.l}.{self.m}.0"
        current_modes = self.found_modes
        current_modes_string = qnms_to_string(current_modes)
        if fund_mode_string not in current_modes_string:
            current_modes.append(
                mode(
                    fund_mode_string,
                    self.M,
                    self.a,
                    retro_def_orbit=self.retro_def_orbit))
        i = 0
        discard_mode = True
        more_than_one_mode = True
        if self.CCE:
            skip_i_init = 10
        else:
            skip_i_init = 1

        while discard_mode and more_than_one_mode:
            self.fitter_list.append(QNMFitVaryingStartingTime(
                self.h,
                self.t0_arr,
                0,
                current_modes,
                run_string_prefix=self.run_string_prefix,
                load_pickle=self.load_pickle,
                skip_i_init=skip_i_init,
                fit_save_prefix=self.fit_save_prefix))
            current_modes_string = qnms_to_string(current_modes)
            fund_mode_indx = current_modes_string.index(fund_mode_string)
            fitter = self.fitter_list[i]
            fitter.do_fits()
            fluc_least_list = []
            fluc_least_indx_list = []
            start_flat_indx_list = []
            result_full = fitter.result_full
            popt_full = result_full.popt_full

            collapsed = np.full(popt_full.shape[1], False)

            if self.CCE:
                collapse_n = 10
            else:
                collapse_n = 1

            for kk in range(popt_full.shape[1] - collapse_n):
                diff = popt_full[:, kk + collapse_n] - popt_full[:, kk]
                collapsed[kk + collapse_n] = np.all(np.abs(diff) < 1e-15)

            for j in range(len(current_modes)):
                A_fix_j_arr = np.array(
                    np.abs(list(fitter.result_full.A_fix_dict["A_" + current_modes_string[j]])))
                A_fix_j_arr = np.where(collapsed, np.nan, A_fix_j_arr)
                phi_fix_j_arr = np.array(
                    list(fitter.result_full.phi_fix_dict["phi_" + current_modes_string[j]]))
                phi_fix_j_arr = np.where(collapsed, np.nan, phi_fix_j_arr)
                fluc_least_indx, _fluc_least, start_flat_indx = flattest_region_quadrature(
                    self.tau_stable_length,
                    A_fix_j_arr, phi_fix_j_arr,
                    quantile_range=self.p_stable,
                    med_min=self.A_tol,
                    fluc_tol=self.epsilon_stable,
                    weight_1=self.beta_A, weight_2=self.beta_phi)
                fluc_least_list.append(_fluc_least)
                fluc_least_indx_list.append(fluc_least_indx)
                start_flat_indx_list.append(start_flat_indx)
            if len(current_modes) <= 1:
                break
            fluc_least_list_no_fund = fluc_least_list.copy()
            del fluc_least_list_no_fund[fund_mode_indx]
            worst_mode_indx = fluc_least_list.index(
                max(fluc_least_list_no_fund))
            bad_mode_indx_list = []
            bad_mode_fluc_list = []
            for ii, fluc_least in enumerate(fluc_least_list):
                if fluc_least > self.epsilon_stable:
                    bad_mode_indx_list.append(ii)
                    bad_mode_fluc_list.append(fluc_least)
            discard_mode = fluc_least_list[worst_mode_indx] > self.epsilon_stable
            worst_mode = current_modes[worst_mode_indx]
            worst_l, worst_m = worst_mode.sum_lm()
            sacrifice_mode = False
            sacrifice_fluc = 0
            if worst_l == self.l and np.abs(worst_m) == np.abs(self.m):
                for jj in bad_mode_indx_list:
                    if jj == worst_mode_indx:
                        continue
                    bad_mode = current_modes[jj]
                    bad_mode_l, bad_mode_m = bad_mode.sum_lm()
                    if bad_mode_l == self.l and np.abs(
                            bad_mode_m) == np.abs(self.m):
                        continue
                    if np.abs(
                            bad_mode.omega -
                            worst_mode.omega) < self.confusion_tol:
                        sacrifice_mode = True
                        if fluc_least_list[jj] > sacrifice_fluc:
                            sacrifice_fluc = fluc_least_list[jj]
                            sacrifice_mode_indx = jj
            if discard_mode:
                if sacrifice_mode:
                    logger.info(
                        f"Although the {current_modes[worst_mode_indx].string()} mode fluctuates the most, "
                        f"the {current_modes[sacrifice_mode_indx].string()} mode is sacrificed instead.")
                    del current_modes[sacrifice_mode_indx]
                else:
                    logger.info(
                        f"discarding {current_modes[worst_mode_indx].string()} mode because it failed flatness test.")
                    del current_modes[worst_mode_indx]
            more_than_one_mode = len(current_modes) > 1
            i += 1
        self.fluc_least_indx_list = fluc_least_indx_list
        self.start_flat_indx_list = start_flat_indx_list
        self.found_modes_screened = current_modes


class ModeSelectorAllFree:
    def __init__(
            self,
            result_full,
            potential_mode_list,
            alpha_r=DEFAULT_ALPHA,
            alpha_i=DEFAULT_ALPHA,
            tau_agnostic=DEFAULT_TAU_AGNOSTIC,
            p_agnostic=DEFAULT_P_AGNOSTIC,
            N_max=10):
        self.result_full = result_full
        self.potential_mode_list = potential_mode_list
        self.alpha_r = alpha_r
        self.alpha_i = alpha_i
        self.tau_agnostic = tau_agnostic
        self.p_agnostic = p_agnostic
        self.passed_mode_list = []
        self.passed_mode_indx = []
        self.N_max = N_max

    def select_modes(self):
        t_approach_duration_list = []
        for i, mode in enumerate(self.potential_mode_list):
            min_distance = closest_free_mode_distance(self.result_full, mode,
                                                      alpha_r=self.alpha_r,
                                                      alpha_i=self.alpha_i)
            start_indx, end_indx = max_consecutive_trues(
                min_distance < 1, tol=self.p_agnostic)
            t0_arr = self.result_full.t0_arr
            t_approach_duration = t0_arr[end_indx] - t0_arr[start_indx]
            if t_approach_duration > self.tau_agnostic:
                self.passed_mode_list.append(mode)
                self.passed_mode_indx.append(i)
                t_approach_duration_list.append(t_approach_duration)
        while len(self.passed_mode_list) > self.N_max:
            del_indx = t_approach_duration_list.index(
                min(t_approach_duration_list))
            del self.passed_mode_list[del_indx]
            del self.passed_mode_indx[del_indx]
            del t_approach_duration_list[del_indx]

    def do_selection(self):
        self.select_modes()


class ModeSearchAllFreeLM:
    def __init__(
            self,
            h,
            M,
            a,
            relevant_lm_list=[],
            t0_arr=np.linspace(
                0,
                50,
                num=501),
            N=5,
            **kwargs_in):
        self.h = h
        self.l = self.h.l
        self.m = self.h.m
        self.M = M
        self.a = a
        self.relevant_lm_list = relevant_lm_list
        self.t0_arr = t0_arr
        self.N = N
        self.kwargs = _merge_kwargs({
            "retro_def_orbit": True,
            "run_string_prefix": "Default",
            "load_pickle": True,
            "a_recoil_tol": 0.,
            "recoil_n_max": 0,
            "alpha_r": DEFAULT_ALPHA,
            "alpha_i": DEFAULT_ALPHA,
            "tau_agnostic": DEFAULT_TAU_AGNOSTIC,
            "p_agnostic": DEFAULT_P_AGNOSTIC,
            'fit_kwargs': {},
            "initial_num": 1,
            "random_initial": False,
            "initial_dict": {},
            "A_guess_relative": True,
            "set_seed": 1234,
            'fit_save_prefix': FIT_SAVE_PATH,
            'BBH_potential_modes': True,
            'potential_modes_custom': []}, kwargs_in)
        self.retro_def_orbit = self.kwargs["retro_def_orbit"]
        self.run_string_prefix = self.kwargs["run_string_prefix"]
        self.a_recoil_tol = self.kwargs["a_recoil_tol"]
        self.alpha_r = self.kwargs["alpha_r"]
        self.alpha_i = self.kwargs["alpha_i"]
        self.tau_agnostic = self.kwargs["tau_agnostic"]
        self.p_agnostic = self.kwargs["p_agnostic"]
        self.recoil_n_max = self.kwargs["recoil_n_max"]
        if self.kwargs["BBH_potential_modes"]:
            if self.a >= self.a_recoil_tol:
                self.potential_modes_full = potential_modes(
                    self.l,
                    self.m,
                    self.M,
                    self.a,
                    self.relevant_lm_list,
                    retro_def_orbit=self.retro_def_orbit)
            else:
                self.potential_modes_full = potential_modes(self.l, self.m, self.M, self.a, [(
                    self.l, self.m)], retro_def_orbit=self.retro_def_orbit, recoil_n_max=self.recoil_n_max)
        else:
            self.potential_modes_full = []
        self.potential_modes_full.extend(self.kwargs["potential_modes_custom"])
        self.potential_modes_full = remove_duplicated_modes(self.potential_modes_full)
        self.potential_modes = self.potential_modes_full.copy()
        self.load_pickle = self.kwargs["load_pickle"]
        self.fit_kwargs = self.kwargs["fit_kwargs"]
        self.initial_num = self.kwargs["initial_num"]
        self.random_initial = self.kwargs["random_initial"]
        self.initial_dict = self.kwargs["initial_dict"]
        self.A_guess_relative = self.kwargs["A_guess_relative"]
        self.set_seed = self.kwargs["set_seed"]
        self.fit_save_prefix = self.kwargs["fit_save_prefix"]

    def mode_search_all_free(self):
        N = self.N
        self.found_modes = []
        self.full_fit = QNMFitVaryingStartingTime(
            self.h,
            self.t0_arr,
            N,
            self.found_modes,
            run_string_prefix=self.run_string_prefix,
            load_pickle=self.load_pickle,
            fit_kwargs=self.fit_kwargs,
            initial_num=self.initial_num,
            random_initial=self.random_initial,
            initial_dict=self.initial_dict,
            A_guess_relative=self.A_guess_relative,
            set_seed=self.set_seed,
            fit_save_prefix=self.fit_save_prefix)
        self.full_fit.do_fits()
        self.mode_selector = ModeSelectorAllFree(
            self.full_fit.result_full,
            self.potential_modes,
            alpha_r=self.alpha_r,
            alpha_i=self.alpha_i,
            tau_agnostic=self.tau_agnostic,
            p_agnostic=self.p_agnostic,
            N_max=N)
        self.mode_selector.do_selection()
        jump_mode_indx = []
        for j in range(len(self.mode_selector.passed_mode_list)):
            if not lower_overtone_present(
                    self.mode_selector.passed_mode_list[j],
                    self.mode_selector.passed_mode_list + self.found_modes):
                jump_mode_indx.append(j)
            if not lower_l_mode_present(
                    self.l,
                    self.m,
                    self.relevant_lm_list,
                    self.mode_selector.passed_mode_list[j],
                    self.mode_selector.passed_mode_list +
                    self.found_modes):
                jump_mode_indx.append(j)
        for k in sorted(list(set(jump_mode_indx)), reverse=True):
            del self.mode_selector.passed_mode_list[k]
        self.found_modes.extend(self.mode_selector.passed_mode_list)
        print_string = f"Runname: {self.run_string_prefix}, N_free = {N}, potential modes: "
        print_string += ', '.join(qnms_to_string(
            self.mode_selector.passed_mode_list))
        logger.info(print_string)
        for j in sorted(self.mode_selector.passed_mode_indx, reverse=True):
            del self.potential_modes[j]

    def do_mode_search(self):
        self.mode_search_all_free()


class ModeSearchAllFreeVaryingN:
    """
    A class that performs a mode search for a given waveform, varying the number
    of free modes used in the fit.

    Attributes:
        h: The waveform to be fit.
        l: The harmonic number l of the waveform.
        m: The harmonic number m of the waveform.
        M: The mass of the black hole.
        a: The dimensionless spin of the black hole.
        relevant_lm_list: A list of tuples of the form (l, m) that specifies
            which recoil modes are relevant for the waveform.
        t0_arr: array of starting times for fitting.
        N_list: A list of integers that specifies the number of free modes
            to be used in each mode searcher in `mode_searchers`.
        kwargs: A dictionary of keyword arguments.
        flatness_checker_kwargs: A dictionary of keyword arguments for the
            `IterativeFlatnessChecker` class.
        mode_searcher_kwargs: A dictionary of keyword arguments for the
            `ModeSearchAllFreeLM` class.
        mode_searchers: A list of `ModeSearchAllFreeLM` objects for mode
            searching with different number of free modes.
        found_modes_final: A list of `mode` objects that contains the final
            list of modes found by the best mode searcher.
        run_string_prefix: A string that is used as a prefix for the run
            name for dumping the `pickle` file.
        load_pickle: A boolean that specifies whether to load the `pickle`
            file.
        CCE: A boolean that specifies whether the waveform is a CCE
            waveform. This is not implemented yet.
        fixed_fitters: A list of `QNMFitVaryingStartingTime` objects that
            contains the final list of fitters used for the flatness checkers in
            each mode searcher.
        flatness_checkers: A list of `IterativeFlatnessChecker` objects that
            contains the list of flatness checkers used for the mode searchers.
        best_run_indx: An integer that specifies the index of the mode
            searcher that found the most number of modes.

    Methods:
        init_searchers: Initializes the mode searchers. 
        do_mode_searches: Performs the mode searches.
    """

    h: waveform
    l: int
    m: int
    M: float
    a: float
    relevant_lm_list: List[Tuple[int, int]]
    t0_arr: np.ndarray
    N_list: List[int]
    kwargs: Dict[str, Any]
    flatness_checker_kwargs: Dict[str, Any]
    mode_searcher_kwargs: Dict[str, Any]
    mode_searchers: List[ModeSearchAllFreeLM]
    found_modes_final: List[mode]
    run_string_prefix: str
    load_pickle: bool
    CCE: bool
    fixed_fitters: List[QNMFitVaryingStartingTime]
    flatness_checkers: List[IterativeFlatnessChecker]
    best_run_indx: int

    def __init__(
            self,
            h: waveform,
            M: float,
            a: float,
            relevant_lm_list: List[Tuple[int, int]] = [],
            t0_arr: np.ndarray = np.linspace(
                0,
                50,
                num=501),
            flatness_checker_kwargs: Dict[str, Any] = {},
            mode_searcher_kwargs: Dict[str, Any] = {},
            **kwargs_in: Any) -> None:
        """
        Initialize the `ModeSearchAllFreeVaryingN` class.

        Parameters:
            h: The waveform to be fit.
            M: The mass of the black hole.
            a: The dimensionless spin of the black hole.
            relevant_lm_list: A list of tuples of the form (l, m) that
                specifies which recoil modes are relevant for the waveform.
            t0_arr: array of starting times for fitting.
            flatness_checker_kwargs: A dictionary of keyword arguments for
                the `IterativeFlatnessChecker` class.
            mode_searcher_kwargs: A dictionary of keyword arguments for the
                `ModeSearchAllFreeLM` class.
            **kwargs_in: keyword arguments.
        """
        self.h = h
        self.l = self.h.l
        self.m = self.h.m
        self.M = M
        self.a = a
        self.relevant_lm_list = relevant_lm_list
        self.t0_arr = t0_arr
        kwargs = _merge_kwargs({'run_string_prefix': 'Default',
                  'load_pickle': True,
                  'N_list': DEFAULT_N_LIST,
                  'CCE': False,
                  'retro_def_orbit': True}, kwargs_in)
        self.N_list = kwargs['N_list']
        self.kwargs = kwargs
        self.flatness_checker_kwargs = flatness_checker_kwargs
        self.mode_searcher_kwargs = mode_searcher_kwargs
        self.mode_searchers = []
        self.init_searchers()
        self.found_modes_final = []
        self.run_string_prefix = kwargs["run_string_prefix"]
        self.load_pickle = self.kwargs["load_pickle"]
        self.CCE = self.kwargs["CCE"]
        _check_no_cce(self.CCE)

    def init_searchers(self) -> None:
        """
        Initializes the mode searchers.
        """
        for _N_init in self.N_list:
            self.mode_searchers.append(
                ModeSearchAllFreeLM(
                    self.h,
                    self.M,
                    self.a,
                    self.relevant_lm_list,
                    N=_N_init,
                    t0_arr=self.t0_arr,
                    **self.mode_searcher_kwargs,
                    **self.kwargs))

    def do_mode_searches(self) -> None:
        """
        Performs the mode searches.
        """
        self.fixed_fitters = []
        self.flatness_checkers = []
        if self.CCE:
            skip_i_init = 10
        else:
            skip_i_init = 1
        for i, mode_searcher in enumerate(self.mode_searchers):
            mode_searcher.do_mode_search()
            self.flatness_checkers.append(
                IterativeFlatnessChecker(
                    self.h,
                    self.t0_arr,
                    self.M,
                    self.a,
                    self.l,
                    self.m,
                    mode_searcher.found_modes,
                    **self.flatness_checker_kwargs,
                    **self.kwargs))
            flatness_checker = self.flatness_checkers[i]
            logger.info(
                f'Performing amplitude and phase flatness check for N_free = {self.N_list[i]}')
            flatness_checker.do_iterative_flatness_check()
            flatness_checker.found_modes_screened
            self.fixed_fitters.append(flatness_checker.fitter_list[-1])
            if len(mode_searcher.found_modes) >= len(self.found_modes_final):
                self.best_run_indx = i
                self.found_modes_final = mode_searcher.found_modes
            logger.info(
                f"Runname: {self.run_string_prefix}, N_free = {self.N_list[i]}, found the following {len(mode_searcher.found_modes)} modes: "
                + ', '.join(qnms_to_string(mode_searcher.found_modes)))

    def summarize_final_modes(self, **kwargs):
        """Return per-mode final results for the selected best run."""
        return summarize_mode_searcher_final_modes(self, **kwargs)


class ModeSearchAllFreeVaryingNSXS:
    """
    A class that performs a mode search for a given SXS waveform, varying the
    number of free modes used in the fit.

    Attributes:
        SXS_num: The SXS number of the waveform. 
        l: The harmonic number l of the
        waveform. m: The harmonic number m of the waveform. 
        t0_arr: array of starting times for fitting. 
        N_list: A list of integers that specifies the number of free modes
            to be used in each mode searcher in `mode_searchers`.
        postfix_string: A string that is appended to the run name for
            dumping the `pickle` file.
        CCE: A boolean that specifies whether the waveform is a CCE
            waveform. This is not implemented yet.
        kwargs: A dictionary of keyword arguments. 
        retro_def_orbit: Whether to define retrograde modes
            with respect to the orbital frame (`True`) or remnant black hole
            frame (`False`). See the methods paper for details. Defaults to
            True.
        relevant_lm_list_override: A boolean that specifies whether to
            override the `relevant_lm_list` attribute of the
            `ModeSearchAllFreeVaryingN` class.
        relevant_lm_list: A list of tuples of the form (l, m) that specifies
            which recoil modes are relevant for the waveform. Used if
            `relevant_lm_list_override` is `True`.
        h: The waveform to be fit. 
        M: The mass of the black hole. 
        a: The dimensionless spin of the black hole. 
        Lev: The resolution level of the SXS simulation. 
        N_list_string: A string that is used as a suffix for the run name for
            dumping the `pickle` file.
        run_string_fitter: A string that is used as a prefix for the run
            name for dumping the `pickle` file for the fitters.
        run_string: A string that is used as a prefix for the run name for
            dumping the `pickle` file for the mode searcher.
        run_string_full: A string that is used as a prefix for the run name
            for dumping the `pickle` file for the mode searcher, including the
            `postfix_string`.
        file_path: The path to the `pickle` file. 
        load_pickle: A boolean that specifies whether to load the `pickle`
            file for the fitters.
        mode_searcher_load_pickle: A boolean that specifies whether to load
            the `pickle` file for the mode searcher.
        set_seed: An integer that specifies the seed for the random number
            generator.
        save_mode_searcher: A boolean that specifies whether to save the
            mode searcher to a `pickle` file.
        mode_searcher_vary_N: A `ModeSearchAllFreeVaryingN` object that
            performs the mode search.
        found_modes_final: A list of `mode` objects that contains the final
            list of modes found by the best mode searcher.
        download: A boolean that specifies whether to download the waveform,
            for `sxs.load`.

    Methods:
        mode_search_varying_N_sxs: Performs the mode searches.
        do_mode_search_varying_N: Performs the mode searches and dumps the
            class instance to a `pickle` file. 
        get_waveform: Loads the waveform from the SXS catalog. 
        pickle_save: Dumps the class instance to a `pickle` file.
        pickle_load: Check whether a `pickle` file exists and can be loaded.

    """

    SXS_num: str
    l: int
    m: int
    t0_arr: np.ndarray
    N_list: List[int]
    postfix_string: str
    CCE: bool
    kwargs: Dict[str, Any]
    retro_def_orbit: bool
    relevant_lm_list_override: bool
    relevant_lm_list: List[Tuple[int, int]]
    h: waveform
    M: float
    a: float
    Lev: int
    N_list_string: str
    run_string_fitter: str
    run_string: str
    run_string_full: str
    file_path: str
    load_pickle: bool
    mode_searcher_load_pickle: bool
    set_seed: int
    save_mode_searcher: bool
    mode_searcher_vary_N: ModeSearchAllFreeVaryingN
    found_modes_final: List[mode]
    download: Optional[bool]

    def __init__(
            self,
            SXS_num: str,
            l: int,
            m: int,
            t0_arr: np.ndarray = np.linspace(
                0,
                50,
                num=501),
            **kwargs_in: Any) -> None:
        """
        Initialize the `ModeSearchAllFreeVaryingNSXS` class.

        Parameters:
            SXS_num: The SXS number of the waveform.
            l: The harmonic number l of the waveform.
            m: The harmonic number m of the waveform.
            t0_arr: array of starting times for fitting.
            **kwargs_in: keyword arguments.
        """
        self.SXS_num = SXS_num
        self.l = l
        self.m = m
        self.t0_arr = t0_arr
        kwargs = _merge_kwargs({'load_pickle': True,
                  'mode_searcher_load_pickle': True,
                  'save_mode_searcher': True,
                  'N_list': DEFAULT_N_LIST,
                  'postfix_string': '',
                  'mode_searchers_save_path': MODE_SEARCHERS_SAVE_PATH,
                  'set_seed_SXS': True,
                  'default_seed': DEFAULT_SEED,
                  'CCE': False,
                  'relevant_lm_list': [],
                  't_end': DEFAULT_T_END,
                  'retro_def_orbit': True,
                  'run_string_fitter': None,
                  'run_string': None,
                  'download': None}, kwargs_in)
        self.N_list = kwargs['N_list']
        self.postfix_string = kwargs['postfix_string']
        self.CCE = kwargs['CCE']
        _check_no_cce(self.CCE)
        self.kwargs = kwargs
        self.retro_def_orbit = self.kwargs['retro_def_orbit']

        if len(self.kwargs['relevant_lm_list']) != 0:
            self.relevant_lm_list_override = True
            self.relevant_lm_list = self.kwargs['relevant_lm_list']
        else:
            self.relevant_lm_list_override = False

        self.N_list_string = '_'.join(list(map(str, self.N_list)))
        if kwargs["run_string_fitter"] is None:
            self.run_string_fitter = f"SXS{self.SXS_num}_lm_{self.l}.{self.m}"
        else:
            self.run_string_fitter = kwargs["run_string_fitter"]
        if kwargs["run_string"] is None:
            self.run_string = f"SXS{self.SXS_num}_lm_{self.l}.{self.m}_N_{self.N_list_string}"
        else:
            self.run_string = kwargs["run_string"]
        save_path = self.kwargs["mode_searchers_save_path"]
        if self.postfix_string == '':
            self.run_string_full = self.run_string
        else:
            self.run_string_full = f"{self.run_string}_{self.postfix_string}"
        self.file_path = os.path.join(
            save_path, f"ModeSearcher_{self.run_string_full}.pickle")
        self.load_pickle = self.kwargs["load_pickle"]
        self.mode_searcher_load_pickle = self.kwargs["mode_searcher_load_pickle"]
        if self.kwargs['set_seed_SXS']:
            self.set_seed = int(self.SXS_num)
        else:
            self.set_seed = self.kwargs['default_seed']
        self.save_mode_searcher = self.kwargs['save_mode_searcher']
        self.t_end = self.kwargs['t_end']
        self.download = kwargs["download"]
        self.get_waveform()

    def mode_search_varying_N_sxs(self) -> None:
        """
        Performs the mode searches.
        """
        kwargs = self.kwargs.copy()
        kwargs.pop('relevant_lm_list')
        self.mode_searcher_vary_N = ModeSearchAllFreeVaryingN(
            self.h,
            self.M,
            self.a,
            self.relevant_lm_list,
            t0_arr=self.t0_arr,
            set_seed=self.set_seed,
            run_string_prefix=self.run_string_fitter,
            **kwargs)
        self.mode_searcher_vary_N.do_mode_searches()
        self.found_modes_final = self.mode_searcher_vary_N.found_modes_final
        logger.info(
            f"Runname: {self.run_string}, final list of modes: "
            + ', '.join(qnms_to_string(self.found_modes_final)))

    def do_mode_search_varying_N(self) -> None:
        """
        Performs the mode searches and dumps the class instance to a `pickle` file.
        """
        self.mode_search_varying_N_sxs()
        if self.save_mode_searcher:
            self.pickle_save()

    def summarize_final_modes(self, **kwargs):
        """Return per-mode final results for the selected best run."""
        if not hasattr(self, "mode_searcher_vary_N"):
            raise ValueError(
                "No mode-search result found. Run do_mode_search_varying_N() first.")
        return self.mode_searcher_vary_N.summarize_final_modes(**kwargs)

    def get_waveform(self) -> None:
        """
        Loads the waveform from the SXS catalog.
        """
        _check_no_cce(self.CCE)
        relevant_modes_dict = get_relevant_lm_waveforms_SXS(
            self.SXS_num, CCE=self.CCE, t_end=self.t_end)
        if not self.relevant_lm_list_override:
            self.relevant_lm_list = relevant_modes_dict_to_lm_tuple(
                relevant_modes_dict)
        peaktime_dom = list(relevant_modes_dict.values())[0].peaktime
        # if self.CCE:
        #     # self.h, self.M, self.a, self.Lev = get_waveform_CCE(
        #     #     self.SXS_num, self.l, self.m)
        # else:
        self.h, self.M, self.a, self.Lev = get_waveform_SXS(
            self.SXS_num,
            self.l,
            self.m,
            t_end=self.t_end,
            download=self.download)
        self.h.update_peaktime(peaktime_dom)

    def pickle_save(self) -> None:
        """
        Dump the class instance to a `pickle` file.
        """
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "wb") as f:
            pickle.dump(self, f)

    def pickle_exists(self) -> bool:
        """
        Check whether a `pickle` file exists and can be loaded.
        """
        if self.mode_searcher_load_pickle:
            return os.path.exists(self.file_path)
        else:
            return False


class ModeSearchAllFreeVaryingNSXSAllRelevant:

    def __init__(
            self,
            SXS_num,
            t0_arr=np.linspace(
                0,
                50,
                num=501),
            **kwargs_in):
        self.SXS_num = SXS_num
        self.t0_arr = t0_arr
        self.kwargs = _merge_kwargs({'load_pickle': True,
                  'mode_searcher_load_pickle': True,
                  'N_list': DEFAULT_N_LIST,
                  'postfix_string': '',
                  't_end': DEFAULT_T_END,
                  'CCE': False}, kwargs_in)
        self.load_pickle = self.kwargs['load_pickle']
        self.mode_searcher_load_pickle = self.kwargs['mode_searcher_load_pickle']
        self.N_list = self.kwargs['N_list']
        self.postfix_string = self.kwargs['postfix_string']
        self.CCE = self.kwargs['CCE']
        _check_no_cce(self.CCE)
        self.get_relevant_lm_list()
        self.get_relevant_lm_mode_searcher_varying_N()

    def do_all_searches(self):
        for _i, _searcher in enumerate(
                self.relevant_lm_mode_searcher_varying_N):
            if _searcher.pickle_exists() and self.mode_searcher_load_pickle:
                _file_path = _searcher.file_path
                with open(_file_path, "rb") as f:
                    self.relevant_lm_mode_searcher_varying_N[_i] = pickle.load(
                        f)
                logger.info(
                    f"Loaded lm = {self.relevant_lm_list[_i][0]}.{self.relevant_lm_list[_i][1]} from an old run.")
            else:
                self.relevant_lm_mode_searcher_varying_N[_i].do_mode_search_varying_N(
                )

    def get_relevant_lm_list(self):
        relevant_modes_dict = get_relevant_lm_waveforms_SXS(
            self.SXS_num, CCE=self.CCE, t_end=self.kwargs['t_end'])
        self.relevant_lm_list = relevant_modes_dict_to_lm_tuple(
            relevant_modes_dict)

    def get_relevant_lm_mode_searcher_varying_N(self):
        self.relevant_lm_mode_searcher_varying_N = []
        for lm in self.relevant_lm_list:
            l, m = lm
            if self.CCE:
                _run_string_prefix = f"CCE{self.SXS_num}_lm_{l}.{m}"
            else:
                _run_string_prefix = f"SXS{self.SXS_num}_lm_{l}.{m}"
            self.relevant_lm_mode_searcher_varying_N. append(
                ModeSearchAllFreeVaryingNSXS(
                    self.SXS_num,
                    l,
                    m,
                    t0_arr=self.t0_arr,
                    run_string_prefix=_run_string_prefix,
                    **self.kwargs))


def closest_free_mode_distance(result_full, mode, alpha_r=1, alpha_i=1):
    omega_r_dict = result_full.omega_dict["real"]
    omega_i_dict = result_full.omega_dict["imag"]
    omega_r_arr = np.array(list(omega_r_dict.values()))
    omega_i_arr = np.array(list(omega_i_dict.values()))
    scaled_distance_arr = np.sqrt(((omega_r_arr - mode.omegar) / \
                           alpha_r)**2 + ((omega_i_arr - mode.omegai) / alpha_i)**2)
    min_distance = np.nanmin(scaled_distance_arr, axis=0)
    return min_distance

def closest_free_mode_distance_cov(result_full, mode, cov, n_sig = 1):
    omega_r_dict = result_full.omega_dict["real"]
    omega_i_dict = result_full.omega_dict["imag"]
    cov_inv = np.linalg.inv(cov)
    scaled_omega_r_arr = []
    scaled_omega_i_arr = []
    for i in range(len(omega_r_dict)):
        omega_arr = np.array([omega_r_dict[f'omega_r_free_{i}'] - mode.omegar,
                              omega_i_dict[f'omega_i_free_{i}'] - mode.omegai])
        omega_arr_adj = cov_inv @ omega_arr
        scaled_omega_r_arr.append(omega_arr_adj[0])
        scaled_omega_i_arr.append(omega_arr_adj[1])
    scaled_omega_r_arr = np.array(scaled_omega_r_arr)
    scaled_omega_i_arr = np.array(scaled_omega_i_arr)
    scaled_distance_arr = np.sqrt(scaled_omega_r_arr**2 * cov[0,0] / n_sig**2 
                                  + scaled_omega_i_arr**2 * cov[1,1] / n_sig**2)
    min_distance = np.nanmin(scaled_distance_arr, axis=0)
    return min_distance

def flattest_region_quadrature(length, arr1, arr2, quantile_range=0.95,
                               normalize_1_by=None, normalize_2_by=2 * np.pi,
                               med_min=1e-3, weight_1=1, weight_2=1.5,
                               fluc_tol=0.1,
                               return_median=False):
    if len(arr1) != len(arr2):
        raise Exception("The length of the two arrays do not match")
    nan_tol = 1 - quantile_range
    total_length = len(arr1)
    quantile_low = (1 - quantile_range) / 2
    quantile_hi = 1 - quantile_low
    fluc_least = np.inf
    fluc_least_indx = 0
    start_flat_indx = -1
    for i in range(total_length - length):
        arr1_in_range = arr1[i:i + length]
        arr2_in_range = arr2[i:i + length]
        if length > 0:
            arr1_nan_frac = np.sum(np.isnan(arr1_in_range)) / length
            arr2_nan_frac = np.sum(np.isnan(arr2_in_range)) / length
        else:
            arr1_nan_frac = 1
            arr2_nan_frac = 1
        quantile_adj = min(arr1_nan_frac / 2, nan_tol / 2)
        hi1 = np.nanquantile(arr1_in_range, min(1, quantile_hi + quantile_adj))
        low1 = np.nanquantile(
            arr1_in_range, max(
                0, quantile_low - quantile_adj))
        med1 = max(np.nanquantile(arr1_in_range, 0.5), med_min)
        if normalize_1_by is None:
            normalize1 = med1
        else:
            normalize1 = normalize_1_by
        fluc1 = (hi1 - low1) / normalize1

        hi2 = np.nanquantile(arr2_in_range, min(1, quantile_hi + quantile_adj))
        low2 = np.nanquantile(
            arr2_in_range, max(
                0, quantile_low - quantile_adj))
        med2 = max(np.nanquantile(arr2_in_range, 0.5), med_min)
        if normalize_2_by is None:
            normalize2 = med2
        else:
            normalize2 = normalize_2_by
        fluc2 = (hi2 - low2) / normalize2

        fluc = np.sqrt((fluc1 * weight_1)**2 + (fluc2 * weight_2)**2)

        if fluc < fluc_tol and arr1_nan_frac < nan_tol and start_flat_indx < 0:
            start_flat_indx = i

        if fluc < fluc_least and arr1_nan_frac < nan_tol:
            fluc_least = fluc
            fluc_least_indx = i

    if return_median:
        return (fluc_least_indx, fluc_least,
                np.nanquantile(arr1[fluc_least_indx:fluc_least_indx + length], 0.5),
                np.nanquantile(arr2[fluc_least_indx:fluc_least_indx + length], 0.5))
    return fluc_least_indx, fluc_least, start_flat_indx


def start_of_flat_region(length, arr1, arr2, **kwargs):
    _, _, start_flat_indx = flattest_region_quadrature(length, arr1, arr2, **kwargs)
    if start_flat_indx < 0:
        return np.nan
    return start_flat_indx


def _mode_strings_from_result_fixed(result_full):
    if hasattr(result_full, "qnm_fixed_list") and result_full.qnm_fixed_list is not None:
        return qnms_to_string(result_full.qnm_fixed_list)
    return [key.removeprefix("A_") for key in result_full.A_fix_dict.keys()]


def _window_length_from_delta_t(t0_arr, delta_t):
    if len(t0_arr) < 2:
        raise ValueError("t0_arr must have at least 2 points.")
    if delta_t <= 0:
        raise ValueError("delta_t must be positive.")
    dt = float(np.median(np.diff(t0_arr)))
    if dt <= 0:
        raise ValueError("t0_arr must be strictly increasing.")
    window_length = int(delta_t / dt + 1)
    if window_length <= 0 or window_length >= len(t0_arr):
        raise ValueError(
            "Computed window length must be in [1, len(t0_arr)-1]. "
            "Adjust delta_t or provide a denser t0_arr."
        )
    return window_length


def _wrap_to_pi(arr):
    return (arr + np.pi) % (2 * np.pi) - np.pi


def _phase_quantiles(phi_window, quantile_low=0.05, quantile_high=0.95, wrap_phase=True):
    """Compute phase quantiles robustly, optionally using circular wrapping."""
    phi_window = np.asarray(phi_window)
    if not wrap_phase:
        phi_med = float(np.nanquantile(phi_window, 0.5))
        phi_low = float(np.nanquantile(phi_window, quantile_low))
        phi_high = float(np.nanquantile(phi_window, quantile_high))
        return phi_low, phi_med, phi_high, (phi_med - phi_low), (phi_high - phi_med)

    phi_valid = phi_window[~np.isnan(phi_window)]
    if phi_valid.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Center on circular mean angle, then quantile wrapped offsets.
    phase_center = float(np.angle(np.mean(np.exp(1j * phi_valid))))
    phi_offset = _wrap_to_pi(phi_valid - phase_center)
    phi_low_offset = float(np.nanquantile(phi_offset, quantile_low))
    phi_med_offset = float(np.nanquantile(phi_offset, 0.5))
    phi_high_offset = float(np.nanquantile(phi_offset, quantile_high))
    phi_low = float(_wrap_to_pi(phase_center + phi_low_offset))
    phi_med = float(_wrap_to_pi(phase_center + phi_med_offset))
    phi_high = float(_wrap_to_pi(phase_center + phi_high_offset))
    return phi_low, phi_med, phi_high, (phi_med_offset - phi_low_offset), (phi_high_offset - phi_med_offset)


def summarize_fixed_mode_flatness(
        result_full,
        delta_t=None,
        flatness_length=None,
        quantile_range=DEFAULT_P_STABLE,
        med_min=DEFAULT_A_TOL,
        weight_1=DEFAULT_BETA_A,
        weight_2=DEFAULT_BETA_PHI,
        fluc_tol=DEFAULT_EPSILON_STABLE,
        wrap_phase=True):
    """Compute per-mode flatness summary for a fixed-frequency fit result.

    Parameters:
        result_full: `QNMFitVaryingStartingTimeResult`-like object with
            `t0_arr`, `A_fix_dict`, and `phi_fix_dict`.
        delta_t: Flatness window width in time units. Ignored if
            `flatness_length` is provided.
        flatness_length: Optional explicit window length in index units.
        quantile_range: Quantile range used for fluctuation estimate.
        med_min: Floor for amplitude/phase normalization medians.
        weight_1: Amplitude fluctuation weight.
        weight_2: Phase fluctuation weight.
        fluc_tol: Threshold for earliest acceptable flatness window.
        wrap_phase: If True, compute phase quantiles using circular wrapping.

    Returns:
        Dict keyed by mode string. Each value includes:
            - flattest window start/end index and times
            - flattest fluctuation
            - median amplitude/phase within flattest window
            - earliest acceptable flat-window start index/time
    """
    if flatness_length is not None and delta_t is not None:
        raise ValueError("Provide either delta_t or flatness_length, not both.")

    t0_arr = np.asarray(result_full.t0_arr)
    if flatness_length is None:
        if delta_t is None:
            delta_t = DEFAULT_TAU_STABLE
        flatness_length = _window_length_from_delta_t(t0_arr, delta_t)
    elif flatness_length <= 0 or flatness_length >= len(t0_arr):
        raise ValueError("flatness_length must be in [1, len(t0_arr)-1].")

    mode_strings = _mode_strings_from_result_fixed(result_full)
    summary = {}
    for mode_string in mode_strings:
        A_arr = np.abs(np.asarray(result_full.A_fix_dict[f"A_{mode_string}"]))
        phi_arr = np.asarray(result_full.phi_fix_dict[f"phi_{mode_string}"])

        flattest_start_idx, fluc_least, earliest_start_idx = flattest_region_quadrature(
            flatness_length,
            A_arr,
            phi_arr,
            quantile_range=quantile_range,
            med_min=med_min,
            weight_1=weight_1,
            weight_2=weight_2,
            fluc_tol=fluc_tol)

        flattest_end_exclusive = flattest_start_idx + flatness_length
        flattest_end_inclusive = min(flattest_end_exclusive - 1, len(t0_arr) - 1)
        A_window = A_arr[flattest_start_idx:flattest_end_exclusive]
        phi_window = phi_arr[flattest_start_idx:flattest_end_exclusive]

        if earliest_start_idx < 0:
            earliest_start_idx_out = np.nan
            earliest_start_time = np.nan
        else:
            earliest_start_idx_out = int(earliest_start_idx)
            earliest_start_time = float(t0_arr[earliest_start_idx])

        summary[mode_string] = {
            "window_length": int(flatness_length),
            "window_delta_t": float(t0_arr[flattest_end_inclusive] - t0_arr[flattest_start_idx]),
            "flattest_start_index": int(flattest_start_idx),
            "flattest_end_index_exclusive": int(flattest_end_exclusive),
            "flattest_start_time": float(t0_arr[flattest_start_idx]),
            "flattest_end_time": float(t0_arr[flattest_end_inclusive]),
            "flattest_fluctuation": float(fluc_least),
            "flattest_amplitude_median": float(np.nanquantile(A_window, 0.5)),
            "flattest_amplitude_low": float(np.nanquantile(A_window, 0.05)),
            "flattest_amplitude_high": float(np.nanquantile(A_window, 0.95)),
            "flattest_phase_median": np.nan,
            "flattest_phase_low": np.nan,
            "flattest_phase_high": np.nan,
            "earliest_flat_start_index": earliest_start_idx_out,
            "earliest_flat_start_time": earliest_start_time,
        }
        summary[mode_string]["flattest_amplitude_minus"] = (
            summary[mode_string]["flattest_amplitude_median"]
            - summary[mode_string]["flattest_amplitude_low"]
        )
        summary[mode_string]["flattest_amplitude_plus"] = (
            summary[mode_string]["flattest_amplitude_high"]
            - summary[mode_string]["flattest_amplitude_median"]
        )
        summary[mode_string]["flattest_phase_minus"] = (
            summary[mode_string]["flattest_phase_median"]
            - summary[mode_string]["flattest_phase_low"]
        )
        summary[mode_string]["flattest_phase_plus"] = (
            summary[mode_string]["flattest_phase_high"]
            - summary[mode_string]["flattest_phase_median"]
        )
        phi_low, phi_med, phi_high, phi_minus, phi_plus = _phase_quantiles(
            phi_window, quantile_low=0.05, quantile_high=0.95, wrap_phase=wrap_phase)
        summary[mode_string]["flattest_phase_low"] = phi_low
        summary[mode_string]["flattest_phase_median"] = phi_med
        summary[mode_string]["flattest_phase_high"] = phi_high
        summary[mode_string]["flattest_phase_minus"] = float(phi_minus)
        summary[mode_string]["flattest_phase_plus"] = float(phi_plus)
    return summary


def fixed_mode_flatness_to_plot_overlays(flatness_summary):
    """Convert `summarize_fixed_mode_flatness` output to plot overlay dicts."""
    bold_dict = {}
    t_flat_start_dict = {}
    for mode_string, mode_summary in flatness_summary.items():
        bold_dict[mode_string] = (
            mode_summary["flattest_start_index"],
            mode_summary["flattest_end_index_exclusive"])
        if not np.isnan(mode_summary["earliest_flat_start_time"]):
            t_flat_start_dict[mode_string] = mode_summary["earliest_flat_start_time"]
    return bold_dict, t_flat_start_dict


def summarize_mode_searcher_final_modes(
        mode_searcher_vary_N,
        quantile_low=0.05,
        quantile_high=0.95,
        wrap_phase=True):
    """Summarize final-mode flatness outputs from a mode-search result.

    Uses the selected best run in `ModeSearchAllFreeVaryingN` and returns a
    per-mode dictionary containing mode presence, flattest window times, median
    amplitude/phase, asymmetric uncertainty ranges (upper/lower quantiles), and
    earliest flatness start time.
    """
    best_run_indx = mode_searcher_vary_N.best_run_indx
    flatness_checker = mode_searcher_vary_N.flatness_checkers[best_run_indx]
    result_full = mode_searcher_vary_N.fixed_fitters[best_run_indx].result_full
    mode_strings = qnms_to_string(mode_searcher_vary_N.found_modes_final)
    t0_arr = result_full.t0_arr
    window_length = flatness_checker.tau_stable_length

    summary = {}
    for mode_string, flattest_start_idx, earliest_start_idx in zip(
            mode_strings,
            flatness_checker.fluc_least_indx_list,
            flatness_checker.start_flat_indx_list):
        flattest_start_idx = int(flattest_start_idx)
        flattest_end_exclusive = flattest_start_idx + window_length
        flattest_end_inclusive = min(flattest_end_exclusive - 1, len(t0_arr) - 1)
        A_arr = np.abs(np.asarray(result_full.A_fix_dict[f"A_{mode_string}"]))
        phi_arr = np.asarray(result_full.phi_fix_dict[f"phi_{mode_string}"])
        A_window = A_arr[flattest_start_idx:flattest_end_exclusive]
        phi_window = phi_arr[flattest_start_idx:flattest_end_exclusive]

        if earliest_start_idx < 0:
            earliest_start_idx_out = np.nan
            earliest_start_time = np.nan
        else:
            earliest_start_idx_out = int(earliest_start_idx)
            earliest_start_time = float(t0_arr[earliest_start_idx])

        A_med = float(np.nanquantile(A_window, 0.5))
        A_low = float(np.nanquantile(A_window, quantile_low))
        A_high = float(np.nanquantile(A_window, quantile_high))
        phi_low, phi_med, phi_high, phi_minus, phi_plus = _phase_quantiles(
            phi_window,
            quantile_low=quantile_low,
            quantile_high=quantile_high,
            wrap_phase=wrap_phase,
        )

        summary[mode_string] = {
            "window_length": int(window_length),
            "flattest_start_index": flattest_start_idx,
            "flattest_end_index_exclusive": int(flattest_end_exclusive),
            "flattest_start_time": float(t0_arr[flattest_start_idx]),
            "flattest_end_time": float(t0_arr[flattest_end_inclusive]),
            "flattest_amplitude_median": A_med,
            "flattest_amplitude_low": A_low,
            "flattest_amplitude_high": A_high,
            "flattest_amplitude_minus": A_med - A_low,
            "flattest_amplitude_plus": A_high - A_med,
            "flattest_phase_median": phi_med,
            "flattest_phase_low": phi_low,
            "flattest_phase_high": phi_high,
            "flattest_phase_minus": float(phi_minus),
            "flattest_phase_plus": float(phi_plus),
            "earliest_flat_start_index": earliest_start_idx_out,
            "earliest_flat_start_time": earliest_start_time,
            "is_present": True,
        }
    return summary


def eff_mode_search(
        inject_params,
        runname,
        retro_def_orbit=True,
        load_pickle=True,
        delay=True,
        **kwargs):

    Mf = inject_params['Mf']
    af = inject_params['af']
    relevant_lm_list = inject_params['relevant_lm_list']
    h_eff = make_eff_ringdown_waveform_from_param(inject_params, delay=delay)
    mode_searcher = ModeSearchAllFreeVaryingN(
        h_eff,
        Mf,
        af,
        relevant_lm_list=relevant_lm_list,
        retro_def_orbit=retro_def_orbit,
        run_string_prefix=runname,
        load_pickle=load_pickle,
        **kwargs)
    mode_searcher.do_mode_searches()

    return mode_searcher


def read_json_eff_mode_search(
        i,
        batch_runname,
        retro_def_orbit=True,
        load_pickle=True,
        delay=True,
        setting_path=SETTING_PATH,
        **kwargs):

    with open(f"{setting_path}/{batch_runname}.json", 'r') as f:
        inject_params_full = json.load(f)

    runname = f"{batch_runname}_{i:03d}"
    mode_searcher = eff_mode_search(
        inject_params_full[runname],
        runname,
        retro_def_orbit=retro_def_orbit,
        load_pickle=load_pickle,
        delay=delay,
        **kwargs)

    return mode_searcher


def read_json_for_param_dict(i, batch_runname, setting_path=SETTING_PATH):

    with open(f"{setting_path}/{batch_runname}.json", 'r') as f:
        inject_params_full = json.load(f)

    runname = f"{batch_runname}_{i:03d}"

    return inject_params_full[runname]
