from utils import *
from QuasinormalMode import *
from Fit import *
from Waveforms import *
import numpy as np
import pickle
import os

class ModeSelectorAllFree:
    def __init__(self, result_full, potential_mode_list, omega_r_tol = 0.05, omega_i_tol = 0.05, t_tol = 10,
                 fraction_tol = 0.95):
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
                                                      r_scale = self.omega_r_tol,
                                                      i_scale = self.omega_i_tol)
            _start_indx, _end_indx = max_consecutive_trues(min_distance < 1, tol = self.fraction_tol)
            _t0_arr = self.result_full.t0_arr
            if _t0_arr[_end_indx] - _t0_arr[_start_indx] > self.t_tol:
                self.passed_mode_list.append(_mode)
                self.passed_mode_indx.append(i)
    
    def do_selection(self):
        self.select_modes()
        
class ModeSearchAllFreeLM:
    def __init__(self, h, M, a, relevant_lm_list = [], t0_arr = np.linspace(0,50,num = 501), N_init = 5,
                  N_step = 3, iterations = 2, **kwargs):
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
        self.potential_modes_full = potential_modes(self.l, self.m, self.M, self.a, self.relevant_lm_list)
        self.potential_modes = self.potential_modes_full.copy()
        self.kwargs = kwargs
    
    def mode_search_all_free(self):
        _N = self.N_init
        self.found_modes = []
        for i in range(self.iterations):
            if i > 0:
                _N = self.N_step
            self.full_fit = QNMFitVaryingStartingTime(self.h, self.t0_arr, _N, self.found_modes)
            self.full_fit.do_fits()
            self.mode_selector = ModeSelectorAllFree(self.full_fit.result_full, 
                                                     self.potential_modes, **self.kwargs)
            self.mode_selector.do_selection()
            print(qnms_to_string(self.mode_selector.passed_mode_list))
            _jump_mode_indx = []
            for j in range(len(self.mode_selector.passed_mode_list)):
                if not lower_overtone_present(self.mode_selector.passed_mode_list[j], 
                                      self.mode_selector.passed_mode_list + self.found_modes):
                    _jump_mode_indx.append(j)
            print(_jump_mode_indx)
            for k in sorted(_jump_mode_indx, reverse=True):
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
    def __init__(self, SXSnum, l, m, N_init = 5, N_step = 3, iterations = 2, **kwargs):
        self.SXSnum = SXSnum
        self.l = l
        self.m = m
        self.N_init = N_init
        self.N_step = N_step
        self.iterations = iterations
        self.get_waveform()
        self.kwargs = kwargs
        
    def mode_search_all_free_sxs(self):
        self.mode_searcher = ModeSearchAllFreeLM(self.h, self.M, self.a, self.relevant_lm_list, N_init = self.N_init,
                                                 N_step = self.N_step, iterations = self.iterations, **self.kwargs)
        self.mode_searcher.do_mode_search()
        self.found_modes = self.mode_searcher.found_modes
        
    def do_mode_search(self):
        self.mode_search_all_free_sxs()
    
    def get_waveform(self):
        _relevant_modes_dict = get_relevant_lm_waveforms_SXS(self.SXSnum)
        self.relevant_lm_list = relevant_modes_dict_to_lm_tuple(_relevant_modes_dict)
        peaktime_dom = list(_relevant_modes_dict.values())[0].peaktime
        self.h, self.M, self.a, self.Lev = get_waveform_SXS(self.SXSnum, self.l, self.m)
        self.h.update_peaktime(peaktime_dom)

class ModeSearchAllFreeVaryingN:
    def __init__(self, h, M, a, relevant_lm_list = [], t0_arr = np.linspace(0,50,num = 501), N_list = [6, 8, 10, 12], **kwargs):
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
        
    def init_searchers(self):
        for _N_init in self.N_list:
            self.mode_searchers.append(ModeSearchAllFreeLM(self.h, self.M, self.a, self.relevant_lm_list, N_init = _N_init,
                                                 iterations = 1, **self.kwargs))
            
    def do_mode_searches(self):
        self.fixed_fitters = []
        for i, _mode_searcher in enumerate(self.mode_searchers):
            _mode_searcher.do_mode_search()
            self.fixed_fitters.append(QNMFitVaryingStartingTime(self.h, self.t0_arr, 0, 
                                                           qnm_fixed_list = _mode_searcher.found_modes))
            self.fixed_fitters[i].do_fits()
            if len(_mode_searcher.found_modes) >= len(self.found_modes_final):
                self.best_run_indx = i
                self.found_modes_final = _mode_searcher.found_modes
                
class ModeSearchAllFreeVaryingNSXS:
    
    def __init__(self, SXSnum, l, m, t0_arr = np.linspace(0,50,num = 501), N_list = [6, 8, 10, 12], **kwargs):
        self.SXSnum = SXSnum
        self.l = l
        self.m = m
        self.t0_arr = t0_arr
        self.N_list = N_list
        self.kwargs = kwargs
        self.get_waveform()
        self.N_list_string = '_'.join(list(map(str, self.N_list)))
        self.run_string = f"SXS{self.SXSnum}_lm_{self.l}.{self.m}_N_{self.N_list_string}"
        self.file_path= f"./pickle/ModeSearcher_{self.run_string}.pickle"
        
    def mode_search_varying_N_sxs(self):
        self.mode_searcher_vary_N = ModeSearchAllFreeVaryingN(self.h, self.M, self.a, 
                                                              self.relevant_lm_list, 
                                                              t0_arr = self.t0_arr, 
                                                              N_list = self.N_list,
                                                              **self.kwargs)
        self.mode_searcher_vary_N.do_mode_searches()
        self.found_modes_final = self.mode_searcher_vary_N.found_modes_final
        
    def do_mode_search_varying_N(self):
        self.mode_search_varying_N_sxs()
        self.pickle_save()
    
    def get_waveform(self):
        _relevant_modes_dict = get_relevant_lm_waveforms_SXS(self.SXSnum)
        self.relevant_lm_list = relevant_modes_dict_to_lm_tuple(_relevant_modes_dict)
        peaktime_dom = list(_relevant_modes_dict.values())[0].peaktime
        self.h, self.M, self.a, self.Lev = get_waveform_SXS(self.SXSnum, self.l, self.m)
        self.h.update_peaktime(peaktime_dom)
        
    def pickle_save(self):
        with open(self.file_path, "wb") as f:
            pickle.dump(self, f)
            
    def pickle_exists(self):
        return os.path.exists(self.file_path)
            
            
class ModeSearchAllFreeVaryingNSXSAllRelevant:
    
    def __init__(self, SXSnum, t0_arr = np.linspace(0,50,num = 501), N_list = [6, 8, 10, 12], **kwargs):
        self.SXSnum = SXSnum
        self.t0_arr = t0_arr
        self.N_list = N_list
        self.kwargs = kwargs
        self.get_relevant_lm_list()
        self.get_relevant_lm_mode_searcher_varying_N()
    
    def do_all_searches(self):
        for _i, _searcher in enumerate(self.relevant_lm_mode_searcher_varying_N):
            if _searcher.pickle_exists():
                _file_path = _searcher.file_path
                with open(_file_path, "rb") as f:
                    self.relevant_lm_mode_searcher_varying_N[_i] = pickle.load(f)
                print(f"reloaded lm = {self.relevant_lm_list[_i][0]}.{self.relevant_lm_list[_i][0]}\
                from an old run.")
            else:
                self.relevant_lm_mode_searcher_varying_N[_i].do_mode_search_varying_N()
    
    def get_relevant_lm_list(self):
        _relevant_modes_dict = get_relevant_lm_waveforms_SXS(self.SXSnum)
        self.relevant_lm_list = relevant_modes_dict_to_lm_tuple(_relevant_modes_dict)
        
    def get_relevant_lm_mode_searcher_varying_N(self):
        self.relevant_lm_mode_searcher_varying_N = []
        for _lm in self.relevant_lm_list:
            _l, _m = _lm
            self.relevant_lm_mode_searcher_varying_N.\
            append(ModeSearchAllFreeVaryingNSXS(self.SXSnum, _l, _m,
                                                t0_arr = self.t0_arr,
                                                N_list = self.N_list,
                                                **self.kwargs))
                
    
def closest_free_mode_distance(result_full, mode, r_scale = 1, i_scale = 1):
    omega_r_dict = result_full.omega_dict["real"]
    omega_i_dict = result_full.omega_dict["imag"]
    omega_r_arr = np.array(list(omega_r_dict.values()))
    omega_i_arr = np.array(list(omega_i_dict.values()))
    scaled_distance_arr = ((omega_r_arr - mode.omegar)/r_scale)**2 + ((omega_i_arr - mode.omegai)/i_scale)**2
    min_distance = np.min(scaled_distance_arr, axis=0)
    return min_distance
    
    
    
    
    
    