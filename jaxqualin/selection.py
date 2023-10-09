from .utils import *
from .qnmode import *
from .fit import *
from .waveforms import *

import numpy as np
import pickle
import json
import os

from typing import List, Tuple, Union, Optional, Dict, Any

SETTING_PATH = os.getcwd()
MODE_SEARCHERS_SAVE_PATH = os.path.join(
    os.getcwd(), ".jaxqualin_cache/mode_searchers")


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
        kwargs = {
            "run_string_prefix": "Default",
            "epsilon_stable": 0.2,
            "tau_stable": 10,
            "retro_def_orbit": True,
            "load_pickle": True,
            "confusion_tol": 0.03,
            "p_stable": 0.95,
            "A_tol": 1e-3,
            "beta_A": 1.0,
            "beta_phi": 1.5,
            "CCE": False,
            "fit_save_prefix": FIT_SAVE_PATH}
        kwargs.update(kwargs_in)
        self.kwargs = kwargs
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
        if self.CCE:
            raise NotImplementedError
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
                    print(
                        f"Although the {current_modes[worst_mode_indx].string()} mode fluctuates the most, "
                        f"the {current_modes[sacrifice_mode_indx].string()} mode is sacrificed instead.")
                    del current_modes[sacrifice_mode_indx]
                else:
                    print(
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
            alpha_r=0.05,
            alpha_i=0.05,
            tau_agnostic=10,
            p_agnostic=0.95,
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
        kwargs = {
            "retro_def_orbit": True,
            "run_string_prefix": "Default",
            "load_pickle": True,
            "a_recoil_tol": 0.,
            "recoil_n_max": 0,
            "alpha_r": 0.05,
            "alpha_i": 0.05,
            "tau_agnostic": 10,
            "p_agnostic": 0.95,
            'fit_kwargs': {},
            "initial_num": 1,
            "random_initial": False,
            "initial_dict": {},
            "A_guess_relative": True,
            "set_seed": 1234,
            'fit_save_prefix': FIT_SAVE_PATH}
        kwargs.update(kwargs_in)
        self.kwargs = kwargs
        self.retro_def_orbit = self.kwargs["retro_def_orbit"]
        self.run_string_prefix = self.kwargs["run_string_prefix"]
        self.a_recoil_tol = self.kwargs["a_recoil_tol"]
        self.alpha_r = self.kwargs["alpha_r"]
        self.alpha_i = self.kwargs["alpha_i"]
        self.tau_agnostic = self.kwargs["tau_agnostic"]
        self.p_agnostic = self.kwargs["p_agnostic"]
        self.recoil_n_max = self.kwargs["recoil_n_max"]
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
        # print(qnms_to_string(self.mode_selector.passed_mode_list))
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
        # print(list(set(_jump_mode_indx)))
        for k in sorted(list(set(jump_mode_indx)), reverse=True):
            del self.mode_selector.passed_mode_list[k]
        self.found_modes.extend(self.mode_selector.passed_mode_list)
        print_string = f"Runname: {self.run_string_prefix}, N_free = {N}, potential modes: "
        print_string += ', '.join(qnms_to_string(
            self.mode_selector.passed_mode_list))
        print(print_string)
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
        kwargs = {'run_string_prefix': 'Default',
                  'load_pickle': True,
                  'N_list': [5, 6, 7, 8, 9, 10],
                  'CCE': False,
                  'retro_def_orbit': True}
        kwargs.update(kwargs_in)
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
        if self.CCE:
            raise NotImplementedError

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
            print(
                f'Performing amplitude and phase flatness check for N_free = {self.N_list[i]}')
            flatness_checker.do_iterative_flatness_check()
            flatness_checker.found_modes_screened
            self.fixed_fitters.append(flatness_checker.fitter_list[-1])
            if len(mode_searcher.found_modes) >= len(self.found_modes_final):
                self.best_run_indx = i
                self.found_modes_final = mode_searcher.found_modes
            print(
                f"Runname: {self.run_string_prefix}, N_free = {self.N_list[i]}, found the following {len(mode_searcher.found_modes)} modes: ")
            print(', '.join(qnms_to_string(mode_searcher.found_modes)))


class ModeSearchAllFreeVaryingNSXS:
    """
    A class that performs a mode search for a given SXS waveform, varying the
    number of free modes used in the fit.

    Attributes:
        SXSnum: The SXS number of the waveform. 
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

    Methods:
        mode_search_varying_N_sxs: Performs the mode searches.
        do_mode_search_varying_N: Performs the mode searches and dumps the
            class instance to a `pickle` file. 
        get_waveform: Loads the waveform from the SXS catalog. 
        pickle_save: Dumps the class instance to a `pickle` file.
        pickle_load: Check whether a `pickle` file exists and can be loaded.

    """

    SXSnum: str
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

    def __init__(
            self,
            SXSnum: str,
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
            SXSnum: The SXS number of the waveform.
            l: The harmonic number l of the waveform.
            m: The harmonic number m of the waveform.
            t0_arr: array of starting times for fitting.
            **kwargs_in: keyword arguments.
        """
        self.SXSnum = SXSnum
        self.l = l
        self.m = m
        self.t0_arr = t0_arr
        kwargs = {'load_pickle': True,
                  'mode_searcher_load_pickle': True,
                  'save_mode_searcher': True,
                  'N_list': [5, 6, 7, 8, 9, 10],
                  'postfix_string': '',
                  'mode_searchers_save_path': MODE_SEARCHERS_SAVE_PATH,
                  'set_seed_SXS': True,
                  'default_seed': 1234,
                  'CCE': False,
                  'relevant_lm_list': [],
                  'retro_def_orbit': True,
                  'run_string_fitter': None,
                  'run_string': None}
        kwargs.update(kwargs_in)
        self.N_list = kwargs['N_list']
        self.postfix_string = kwargs['postfix_string']
        self.CCE = kwargs['CCE']
        if self.CCE:
            raise NotImplementedError
        self.kwargs = kwargs
        self.retro_def_orbit = self.kwargs['retro_def_orbit']

        if len(self.kwargs['relevant_lm_list']) == 0:
            self.relevant_lm_list_override = True
            self.relevant_lm_list = self.kwargs['relevant_lm_list']
        else:
            self.relevant_lm_list_override = False

        self.get_waveform()
        self.N_list_string = '_'.join(list(map(str, self.N_list)))
        if kwargs["run_string_fitter"] is None:
            self.run_string_fitter = f"SXS{self.SXSnum}_lm_{self.l}.{self.m}"
        else:
            self.run_string_fitter = kwargs["run_string_fitter"]
        if kwargs["run_string"] is None:
            self.run_string = f"SXS{self.SXSnum}_lm_{self.l}.{self.m}_N_{self.N_list_string}"
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
            self.set_seed = int(self.SXSnum)
        else:
            self.set_seed = self.kwargs['default_seed']
        self.save_mode_searcher = self.kwargs['save_mode_searcher']

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
        print(f"Runname: {self.run_string}, final list of modes: ")
        print(', '.join(qnms_to_string(self.found_modes_final)))

    def do_mode_search_varying_N(self) -> None:
        """
        Performs the mode searches and dumps the class instance to a `pickle` file.
        """
        self.mode_search_varying_N_sxs()
        if self.save_mode_searcher:
            self.pickle_save()

    def get_waveform(self) -> None:
        """
        Loads the waveform from the SXS catalog.
        """
        if self.CCE:
            raise NotImplementedError
        relevant_modes_dict = get_relevant_lm_waveforms_SXS(
            self.SXSnum, CCE=self.CCE)
        if not self.relevant_lm_list_override:
            self.relevant_lm_list = relevant_modes_dict_to_lm_tuple(
                relevant_modes_dict)
        peaktime_dom = list(relevant_modes_dict.values())[0].peaktime
        # if self.CCE:
        #     # self.h, self.M, self.a, self.Lev = get_waveform_CCE(
        #     #     self.SXSnum, self.l, self.m)
        # else:
        self.h, self.M, self.a, self.Lev = get_waveform_SXS(
            self.SXSnum, self.l, self.m)
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
            SXSnum,
            t0_arr=np.linspace(
                0,
                50,
                num=501),
            **kwargs_in):
        self.SXSnum = SXSnum
        self.t0_arr = t0_arr
        kwargs = {'load_pickle': True,
                  'mode_searcher_load_pickle': True,
                  'N_list': [5, 6, 7, 8, 9, 10],
                  'postfix_string': '',
                  'CCE': False}
        kwargs.update(kwargs_in)
        self.kwargs = kwargs
        self.load_pickle = kwargs['load_pickle']
        self.mode_searcher_load_pickle = kwargs['mode_searcher_load_pickle']
        self.N_list = kwargs['N_list']
        self.postfix_string = kwargs['postfix_string']
        self.CCE = kwargs['CCE']
        if self.CCE:
            raise NotImplementedError
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
                print(
                    f"Loaded lm = {self.relevant_lm_list[_i][0]}.{self.relevant_lm_list[_i][1]} from an old run.")
            else:
                self.relevant_lm_mode_searcher_varying_N[_i].do_mode_search_varying_N(
                )

    def get_relevant_lm_list(self):
        relevant_modes_dict = get_relevant_lm_waveforms_SXS(
            self.SXSnum, CCE=self.CCE)
        self.relevant_lm_list = relevant_modes_dict_to_lm_tuple(
            relevant_modes_dict)

    def get_relevant_lm_mode_searcher_varying_N(self):
        self.relevant_lm_mode_searcher_varying_N = []
        for lm in self.relevant_lm_list:
            l, m = lm
            if self.CCE:
                _run_string_prefix = f"CCE{self.SXSnum}_lm_{l}.{m}"
            else:
                _run_string_prefix = f"SXS{self.SXSnum}_lm_{l}.{m}"
            self.relevant_lm_mode_searcher_varying_N. append(
                ModeSearchAllFreeVaryingNSXS(
                    self.SXSnum,
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
    scaled_distance_arr = ((omega_r_arr - mode.omegar) / \
                           alpha_r)**2 + ((omega_i_arr - mode.omegai) / alpha_i)**2
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


def start_of_flat_region(length, arr1, arr2, quantile_range=0.95,
                         normalize_1_by=None, normalize_2_by=2 * np.pi,
                         med_min=1e-3, weight_1=1, weight_2=1.5,
                         fluc_tol=0.1):
    if len(arr1) != len(arr2):
        raise Exception("The length of the two arrays do not match")
    nan_tol = 1 - quantile_range
    total_length = len(arr1)
    quantile_low = (1 - quantile_range) / 2
    quantile_hi = 1 - quantile_low
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

        if fluc < fluc_tol and arr1_nan_frac < nan_tol:
            start_flat_indx = i
            return start_flat_indx

    return np.nan


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
