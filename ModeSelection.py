from utils import *
from QuasinormalMode import *
from Fit import *
from Waveforms import *
import numpy as np
import pickle
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
MODE_SEARCHERS_SAVE_PATH = os.path.join(ROOT_PATH, "pickle/mode_searchers")


class IterativeFlatnessChecker:
    
    def __init__(self, h, t0_arr, found_modes, **kwargs_in):
        self.h = h
        self.t0_arr = t0_arr
        self.found_modes = found_modes
        self.fitter_list = []
        kwargs = {"run_string_prefix": "Default", "tolerance": 0.2,
                  "flatness_range": 10}
        kwargs.update(kwargs_in)
        self.kwargs = kwargs
        self.run_string_prefix = self.kwargs["run_string_prefix"]
        self.tolerance = self.kwargs["tolerance"]
        self.flatness_range = self.kwargs["flatness_range"] 
        self.flatness_length = int(self.flatness_range/(self.t0_arr[1] - self.t0_arr[0])+1)
        
    def do_iterative_flatness_check(self):
        _current_modes = self.found_modes
        i = 0
        _discard_mode = True
        _more_than_one_mode = True
        while _discard_mode and _more_than_one_mode:
            self.fitter_list.append(QNMFitVaryingStartingTime(
                    self.h,
                    self.t0_arr,
                    0,
                    _current_modes,
                    run_string_prefix=self.run_string_prefix))
            _fitter = self.fitter_list[i]
            _fitter.do_fits()
            _fluc_least_list = []
            _fluc_least_indx_list = []
            _fluc_least_list = []
            _fluc_least_indx_list = []
            for j in range(len(_current_modes)):
                _A_fix_j_arr = np.abs(list(_fitter.result_full.A_fix_dict.values())[j])
                _phi_fix_j_arr = list(_fitter.result_full.phi_fix_dict.values())[j]
                _fluc_least_indx, _fluc_least = flattest_region_quadrature(
                    self.flatness_length,
                    _A_fix_j_arr, _phi_fix_j_arr, 
                    quantile_range = 0.95, weight_2 = 1.5)
                _fluc_least_list.append(_fluc_least)
                _fluc_least_indx_list.append(_fluc_least_indx)
            _worst_mode_indx = _fluc_least_list.index(max(_fluc_least_list))
            _discard_mode = _fluc_least_list[_worst_mode_indx] > self.tolerance
            # print(_fluc_least_list)
            if _discard_mode:
                print(f"discarding {_current_modes[_worst_mode_indx].string()} mode because it failed flatness test")
                del _current_modes[_worst_mode_indx]
            _more_than_one_mode = len(_current_modes)>1
            i += 1
        self.fluc_least_indx_list = _fluc_least_indx_list
        self.found_modes_screened = _current_modes


class ModeSelectorAllFree:
    def __init__(
            self,
            result_full,
            potential_mode_list,
            omega_r_tol=0.05,
            omega_i_tol=0.05,
            t_tol=10,
            fraction_tol=0.95,
            **kwargs):
        self.result_full = result_full
        self.potential_mode_list = potential_mode_list
        self.omega_r_tol = omega_r_tol
        self.omega_i_tol = omega_i_tol
        self.t_tol = t_tol
        self.fraction_tol = fraction_tol
        self.passed_mode_list = []
        self.passed_mode_indx = []

    def select_modes(self):
        for i, _mode in enumerate(self.potential_mode_list):
            min_distance = closest_free_mode_distance(self.result_full, _mode,
                                                      r_scale=self.omega_r_tol,
                                                      i_scale=self.omega_i_tol)
            _start_indx, _end_indx = max_consecutive_trues(
                min_distance < 1, tol=self.fraction_tol)
            _t0_arr = self.result_full.t0_arr
            if _t0_arr[_end_indx] - _t0_arr[_start_indx] > self.t_tol:
                self.passed_mode_list.append(_mode)
                self.passed_mode_indx.append(i)

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
            N_init=5,
            N_step=3,
            iterations=2,
            **kwargs_in):
        self.h = h
        self.l = self.h.l
        self.m = self.h.m
        self.M = M
        self.a = a
        self.relevant_lm_list = relevant_lm_list
        self.t0_arr = t0_arr
        self.N_init = N_init
        self.N_step = N_step
        self.iterations = iterations
        kwargs = {"retro": False, "run_string_prefix": "Default"}
        kwargs.update(kwargs_in)
        self.kwargs = kwargs
        self.retro = self.kwargs["retro"]
        self.run_string_prefix = self.kwargs["run_string_prefix"]
        if self.a >= 0.3:
            self.potential_modes_full = potential_modes(
                self.l, self.m, self.M, self.a, self.relevant_lm_list, retro = self.retro)
        else:
            self.potential_modes_full = potential_modes(
                self.l, self.m, self.M, self.a, [(self.l, self.m)], retro = self.retro)
        self.potential_modes = self.potential_modes_full.copy()

    def mode_search_all_free(self):
        _N = self.N_init
        self.found_modes = []
        for i in range(self.iterations):
            if i > 0:
                _N = self.N_step
            self.full_fit = QNMFitVaryingStartingTime(
                self.h,
                self.t0_arr,
                _N,
                self.found_modes,
                run_string_prefix=self.run_string_prefix)
            self.full_fit.do_fits()
            self.mode_selector = ModeSelectorAllFree(
                self.full_fit.result_full, self.potential_modes, **self.kwargs)
            self.mode_selector.do_selection()
            print(qnms_to_string(self.mode_selector.passed_mode_list))
            _jump_mode_indx = []
            for j in range(len(self.mode_selector.passed_mode_list)):
                if not lower_overtone_present(
                        self.mode_selector.passed_mode_list[j],
                        self.mode_selector.passed_mode_list + self.found_modes):
                    _jump_mode_indx.append(j)
                if not lower_l_mode_present(self.l, self.m, 
                                            self.relevant_lm_list, 
                                            self.mode_selector.passed_mode_list[j], 
                                            self.mode_selector.passed_mode_list + self.found_modes):
                    _jump_mode_indx.append(j)
            print(list(set(_jump_mode_indx)))
            for k in sorted(list(set(_jump_mode_indx)), reverse=True):
                del self.mode_selector.passed_mode_list[k]
            if len(self.mode_selector.passed_mode_list) == 0:
                break
            self.found_modes.extend(self.mode_selector.passed_mode_list)
            print(qnms_to_string(self.mode_selector.passed_mode_list))
            for j in sorted(self.mode_selector.passed_mode_indx, reverse=True):
                del self.potential_modes[j]

    def do_mode_search(self):
        self.mode_search_all_free()


class ModeSearchAllFreeLMSXS:
    def __init__(
            self,
            SXSnum,
            l,
            m,
            N_init=5,
            N_step=3,
            iterations=2,
            **kwargs):
        self.SXSnum = SXSnum
        self.l = l
        self.m = m
        self.N_init = N_init
        self.N_step = N_step
        self.iterations = iterations
        self.get_waveform()
        self.kwargs = kwargs

    def mode_search_all_free_sxs(self):
        self.mode_searcher = ModeSearchAllFreeLM(
            self.h,
            self.M,
            self.a,
            self.relevant_lm_list,
            N_init=self.N_init,
            N_step=self.N_step,
            iterations=self.iterations,
            retro = self.retro,
            **self.kwargs)
        self.mode_searcher.do_mode_search()
        self.found_modes = self.mode_searcher.found_modes

    def do_mode_search(self):
        self.mode_search_all_free_sxs()

    def get_waveform(self):
        _relevant_modes_dict, self.retro = get_relevant_lm_waveforms_SXS(self.SXSnum)
        self.relevant_lm_list = relevant_modes_dict_to_lm_tuple(
            _relevant_modes_dict)
        peaktime_dom = list(_relevant_modes_dict.values())[0].peaktime
        self.h, self.M, self.a, self.Lev, _retro = get_waveform_SXS(
            self.SXSnum, self.l, self.m)
        self.h.update_peaktime(peaktime_dom)


class ModeSearchAllFreeVaryingN:
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
            N_list=[
                6,
                8,
                10,
                12],
            **kwargs):
        self.h = h
        self.l = self.h.l
        self.m = self.h.m
        self.M = M
        self.a = a
        self.relevant_lm_list = relevant_lm_list
        self.t0_arr = t0_arr
        self.N_list = N_list
        self.kwargs = kwargs
        self.mode_searchers = []
        self.init_searchers()
        self.found_modes_final = []
        if not ("run_string_prefix" in list(kwargs.keys())):
            self.run_string_prefix = "Default"
        else:
            self.run_string_prefix = kwargs["run_string_prefix"]

    def init_searchers(self):
        for _N_init in self.N_list:
            self.mode_searchers.append(
                ModeSearchAllFreeLM(
                    self.h,
                    self.M,
                    self.a,
                    self.relevant_lm_list,
                    N_init=_N_init,
                    iterations=1,
                    **self.kwargs))

    def do_mode_searches(self):
        self.fixed_fitters = []
        self.flatness_checkers = []
        for i, _mode_searcher in enumerate(self.mode_searchers):
            _mode_searcher.do_mode_search()
            self.flatness_checkers.append(IterativeFlatnessChecker(self.h, self.t0_arr,
            _mode_searcher.found_modes, run_string_prefix = self.run_string_prefix))
            _flatness_checker = self.flatness_checkers[i]
            _flatness_checker.do_iterative_flatness_check()
            _flatness_checker.found_modes_screened
            self.fixed_fitters.append(
                QNMFitVaryingStartingTime(
                    self.h,
                    self.t0_arr,
                    0,
                    qnm_fixed_list=_flatness_checker.found_modes_screened,
                    run_string_prefix=self.run_string_prefix))
            self.fixed_fitters[i].do_fits()
            if len(_mode_searcher.found_modes) >= len(self.found_modes_final):
                self.best_run_indx = i
                self.found_modes_final = _mode_searcher.found_modes


class ModeSearchAllFreeVaryingNSXS:

    def __init__(
            self,
            SXSnum,
            l,
            m,
            t0_arr=np.linspace(
                0,
                50,
                num=501),
            N_list=[
                6,
                8,
                10,
                12],
            **kwargs):
        self.SXSnum = SXSnum
        self.l = l
        self.m = m
        self.t0_arr = t0_arr
        self.N_list = N_list
        self.kwargs = kwargs
        self.get_waveform()
        self.N_list_string = '_'.join(list(map(str, self.N_list)))
        self.run_string = f"SXS{self.SXSnum}_lm_{self.l}.{self.m}_N_{self.N_list_string}"
        self.file_path = os.path.join(MODE_SEARCHERS_SAVE_PATH, f"ModeSearcher_{self.run_string}.pickle")
        if "load_pickle" not in self.kwargs:
            self.load_pickle = True
        else:
            self.load_pickle = self.kwargs["load_pickle"]

    def mode_search_varying_N_sxs(self):
        self.mode_searcher_vary_N = ModeSearchAllFreeVaryingN(
            self.h,
            self.M,
            self.a,
            self.relevant_lm_list,
            t0_arr=self.t0_arr,
            N_list=self.N_list,
            retro = self.retro,
            **self.kwargs)
        self.mode_searcher_vary_N.do_mode_searches()
        self.found_modes_final = self.mode_searcher_vary_N.found_modes_final

    def do_mode_search_varying_N(self):
        self.mode_search_varying_N_sxs()
        self.pickle_save()

    def get_waveform(self):
        _relevant_modes_dict, self.retro = get_relevant_lm_waveforms_SXS(self.SXSnum)
        self.relevant_lm_list = relevant_modes_dict_to_lm_tuple(
            _relevant_modes_dict)
        peaktime_dom = list(_relevant_modes_dict.values())[0].peaktime
        self.h, self.M, self.a, self.Lev, _retro = get_waveform_SXS(
            self.SXSnum, self.l, self.m)
        self.h.update_peaktime(peaktime_dom)

    def pickle_save(self):
        with open(self.file_path, "wb") as f:
            pickle.dump(self, f)

    def pickle_exists(self):
        if self.load_pickle:
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
            N_list=[
                6,
                8,
                10,
                12],
            **kwargs):
        self.SXSnum = SXSnum
        self.t0_arr = t0_arr
        self.N_list = N_list
        self.kwargs = kwargs
        self.get_relevant_lm_list()
        self.get_relevant_lm_mode_searcher_varying_N()

    def do_all_searches(self):
        for _i, _searcher in enumerate(
                self.relevant_lm_mode_searcher_varying_N):
            if _searcher.pickle_exists():
                _file_path = _searcher.file_path
                with open(_file_path, "rb") as f:
                    self.relevant_lm_mode_searcher_varying_N[_i] = pickle.load(
                        f)
                print(
                    f"reloaded lm = {self.relevant_lm_list[_i][0]}.{self.relevant_lm_list[_i][0]} from an old run.")
            else:
                self.relevant_lm_mode_searcher_varying_N[_i].do_mode_search_varying_N(
                )

    def get_relevant_lm_list(self):
        _relevant_modes_dict, self.retro = get_relevant_lm_waveforms_SXS(self.SXSnum)
        self.relevant_lm_list = relevant_modes_dict_to_lm_tuple(
            _relevant_modes_dict)

    def get_relevant_lm_mode_searcher_varying_N(self):
        self.relevant_lm_mode_searcher_varying_N = []
        for _lm in self.relevant_lm_list:
            _l, _m = _lm
            _run_string_prefix = f"SXS{self.SXSnum}_lm_{_l}.{_m}"
            self.relevant_lm_mode_searcher_varying_N. append(
                ModeSearchAllFreeVaryingNSXS(
                    self.SXSnum,
                    _l,
                    _m,
                    t0_arr=self.t0_arr,
                    N_list=self.N_list,
                    run_string_prefix=_run_string_prefix,
                    **self.kwargs))


def closest_free_mode_distance(result_full, mode, r_scale=1, i_scale=1):
    omega_r_dict = result_full.omega_dict["real"]
    omega_i_dict = result_full.omega_dict["imag"]
    omega_r_arr = np.array(list(omega_r_dict.values()))
    omega_i_arr = np.array(list(omega_i_dict.values()))
    scaled_distance_arr = ((omega_r_arr - mode.omegar) / \
                           r_scale)**2 + ((omega_i_arr - mode.omegai) / i_scale)**2
    min_distance = np.min(scaled_distance_arr, axis=0)
    return min_distance

def flattest_region(length, arr, quantile_range = 0.95, normalize_by = None):
    total_length = len(arr)
    quantile_low = (1 - quantile_range)/2
    quantile_hi = 1 - quantile_low
    fluc_least = np.inf
    fluc_least_indx = 0
    for i in range(total_length - length):
        arr_in_range = arr[i:i+length]
        hi = np.quantile(arr_in_range,quantile_hi)
        low = np.quantile(arr_in_range,quantile_low)
        med = np.quantile(arr_in_range, 0.5)
        if normalize_by is None:
            normalize = med
        else:
            normalize = normalize_by
        fluc = (hi - low)/normalize
        if fluc < fluc_least:
            fluc_least = fluc
            fluc_least_indx = i
    return fluc_least_indx, fluc_least

def flattest_region_quadrature(length, arr1, arr2, quantile_range = 0.95, 
                               normalize_1_by = None, normalize_2_by = 2*np.pi, 
                               med_min = 1e-3, weight_1 = 1, weight_2 = 1):
    if len(arr1) != len(arr2):
        raise Exception("The length of the two arrays do not match")
    total_length = len(arr1)
    quantile_low = (1 - quantile_range)/2
    quantile_hi = 1 - quantile_low
    fluc_least = np.inf
    fluc_least_indx = 0
    for i in range(total_length - length):
        arr1_in_range = arr1[i:i+length]
        arr2_in_range = arr2[i:i+length]
        hi1 = np.quantile(arr1_in_range,quantile_hi)
        low1 = np.quantile(arr1_in_range,quantile_low)
        med1 = max(np.quantile(arr1_in_range, 0.5), med_min)
        if normalize_1_by is None:
            normalize1 = med1
        else:
            normalize1 = normalize_1_by
        fluc1 = (hi1 - low1)/normalize1
        
        hi2 = np.quantile(arr2_in_range,quantile_hi)
        low2 = np.quantile(arr2_in_range,quantile_low)
        med2 = max(np.quantile(arr2_in_range, 0.5), med_min)
        if normalize_2_by is None:
            normalize2 = med2
        else:
            normalize2 = normalize_2_by
        fluc2 = (hi2 - low2)/normalize2
        
        fluc = np.sqrt((fluc1*weight_1)**2 + (fluc2*weight_2)**2)
        
        if fluc < fluc_least:
            fluc_least = fluc
            fluc_least_indx = i
            hi1_best, low1_best, med1_best, normalize1_best, fluc1_best, hi2_best, low2_best, med2_best, normalize2_best, fluc2_best = (hi1, low1, med1, normalize1, fluc1, hi2, low2, med2, normalize2, fluc2)
    # print(hi1_best, low1_best, med1_best, normalize1_best, fluc1_best, hi2_best, low2_best, med2_best, normalize2_best, fluc2_best)
    return fluc_least_indx, fluc_least
    