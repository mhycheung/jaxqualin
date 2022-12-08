from Waveforms import *
import glob
import re
from bisect import bisect_right
from Fit import *
import csv

GRChombo_root = "/home/mark/dev/GR_Chombo_QNM/"
GRChombo_data_root = GRChombo_root + "NewHeadOnData/"


def read_mass():
    massgammas = {}
    masserrorgammas = {}
    with open(GRChombo_root + "masses.csv", newline = '') as csvfile:
        spamreader = csv.reader(csvfile)
        for i, row in enumerate(spamreader):
            if i == 0:
                continue
            massgammas[row[0]] = float(row[1])
            masserrorgammas[row[0]] = float(row[2])
    return massgammas, masserrorgammas

def read_data(gamma, l, radius = 1200, box = 4096, base = 960, gamma_scale = False):
    dirname = glob.glob(GRChombo_data_root + f"gamma{gamma}/*box{box}_base{base}*")[0]
    time = []
    Psi = []
    with open(dirname + f"/Weyl_integral_{l}0.dat", "r") as f:
        lines = f.readlines()
        strings = re.split(' +', lines[1].strip())
        radii = list(map(float,strings[3:]))
        col = radii.index(1200) + 1
        for line in lines[2:]:
            strings = re.split(' +', line.strip())
            time.append(float(strings[0]))
            Psi.append(float(strings[col]))
    if gamma_scale == True:
        time = np.array(time)
    else:
        time = np.array(time)/gamma
    Psi = np.array(Psi) + 0.j
    return waveform(time, Psi, l = l, m = 0)

def estimate_mass(Psi, qnm_free_list, 
                  run_string_prefix,
                  tstart = 30,
                  tend = 50,
                  one_t = False,
                  gamma = None,
                  gamma_scale = False,
                  qnm_fixed_list = [],
                  t0_arr = np.linspace(0, 100, num = 51)):
                  
    qnm_fitter = QNMFitVaryingStartingTime(Psi,
                t0_arr,
                qnm_fixed_list = qnm_fixed_list,
                qnm_free_list = qnm_free_list,
                Schwarzschild = True,
                run_string_prefix=run_string_prefix,
                var_M_a = True,
                load_pickle=True)
    
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
    
    return M_mean, M_std

def estimate_mass_gamma_l(gamma, l,
                          tstart = 30,
                          tend = 50,
                          one_t = False,
                          gamma_scale = False,
                          t0_arr = np.linspace(0,100,num=51)):
    Psi = read_data(gamma, l, gamma_scale = gamma_scale)
    qnm_free_list = long_str_to_qnms_free(f"{l}.2.0")
    if gamma_scale:
        run_string_prefix = f"GRChombo_gamma_{gamma}_scaled_l_{l}"
    else:
        run_string_prefix = f"GRChombo_gamma_{gamma}_l_{l}"

    M_mean, M_std = estimate_mass(Psi, qnm_free_list, 
                      run_string_prefix,
                      tstart = tstart,
                      tend = tend,
                      one_t = one_t,
                      gamma = gamma,
                      gamma_scale = gamma_scale,
                      qnm_fixed_list = [],
                      t0_arr = t0_arr)
    
    return M_mean, M_std
    
    
def estimate_mass_for_all_simulations(gammas, ls,
                                      tstart = 30,
                                      tend = 50,
                                      one_t = False,
                                      gamma_scale = False,
                                      t0_arr = np.linspace(0,100,num=51)):
    massqnmdict = {}
    massqnmstddict = {}
    for gamma in gammas:
        massqnmdict[str(gamma)] = {}
        massqnmstddict[str(gamma)] = {}
        for l in ls:
            M_mean, M_std = estimate_mass_gamma_l(gamma, l, 
                                  tstart = tstart,
                                  tend = tend,
                                  one_t = one_t,
                                  gamma_scale = gamma_scale,
                                  t0_arr = t0_arr)
            massqnmdict[str(gamma)][str(l)] = M_mean
            massqnmstddict[str(gamma)][str(l)] = M_std
    return massqnmdict, massqnmstddict
            
    
    