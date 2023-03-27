from Fit import *
from QuasinormalMode import *
from Waveforms import *
from ModeSelection import *
import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import Pool, Process

import os
import argparse

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH_ROOT = os.path.join(ROOT_PATH, f"plots/")
PLOT_PATH = os.path.join(PLOT_PATH_ROOT, "convergence/")

fit_results = []

def do_one_fit(N_fix, guess_list, batch_runname, l, m, t0_arr, N_free, qnm_fixed_list,
               load_pickle, run_string_prefix, fit_kwargs, jcf, return_jcf,
               eff, eff_i, delay, i):
    
    global fit_results

    if eff:
        inject_params = read_json_for_param_dict(eff_i, batch_runname)
        h = make_eff_ringdown_waveform_from_param(inject_params, delay = delay)
    else:
        SXS_num = batch_runname
        h, Mf, af, _, retro = get_waveform_SXS(SXS_num, l, m)
    params0 = jnp.array([1, 1] * N_fix + guess_list[i])
    print(N_free)
    fitter = QNMFitVaryingStartingTime(h, t0_arr, N_free = N_free,
                                    qnm_fixed_list = qnm_fixed_list, 
                                    load_pickle = load_pickle, jcf = jcf,
                                    params0 = params0, 
                                    run_string_prefix = f"{run_string_prefix}_{i}",
                                    fit_kwargs = fit_kwargs)
    fitter.do_fits()

    fit_results.append[(i, fitter)]

def main():

    global fit_results

    parser = argparse.ArgumentParser()

    parser.add_argument('batch_runname')
    parser.add_argument('l', type = int)
    parser.add_argument('m', type = int)
    parser.add_argument('N_free', type = int)
    parser.add_argument('guess_num', type = int)
    parser.add_argument('runname')

    parser.add_argument("-n", "--num", dest = "num", action = "store", 
                        default = -1, type = int)

    parser.add_argument("-eff", "--eff", dest = "eff", action = 'store_true')
    parser.add_argument("-i", "--eff_i", dest = "eff_i", action = "store", 
                        type = int)
    parser.add_argument("-nd", "--no_delay", dest = "delay", action = "store_false")
    parser.add_argument("-l", "--load-pickle", dest = "load_pickle", 
                        action='store_true')
    parser.add_argument("-w", "--weighted", dest = "weighted", 
                        action='store_true')

    parser.add_argument("-nAr", "--no_A_relative", dest = "A_rel", 
                        action='store_false')                    

    parser.add_argument("-tl", "--t_low", dest = "tl", action = "store",
                        default = 0, type = float)
    parser.add_argument("-th", "--t_high", dest = "th", action = "store",
                        default = 50, type = float)
    parser.add_argument("-tn", "--t_num", dest = "tn", action = "store",
                        default = 501, type = int)

    parser.add_argument("-s", "--seed", dest = "seed", action = "store",
                        default = 1234, type = int)
    parser.add_argument("-Al", "--A_low", dest = "Al", action = "store",
                        default = -1, type = float)
    parser.add_argument("-Ah", "--A_high", dest = "Ah", action = "store",
                        default = 1, type = float)
    parser.add_argument("-pl", "--phi_low", dest = "pl", action = "store",
                        default = 0, type = float)
    parser.add_argument("-ph", "--phi_high", dest = "ph", action = "store",
                        default = 2*np.pi, type = float)
    parser.add_argument("-rl", "--omega_r_low", dest = "rl", action = "store",
                        default = -2, type = float)
    parser.add_argument("-rh", "--omega_r_high", dest = "rh", action = "store",
                        default = 2, type = float)
    parser.add_argument("-il", "--omega_i_low", dest = "il", action = "store",
                        default = 0, type = float)
    parser.add_argument("-ih", "--omega_i_high", dest = "ih", action = "store",
                        default = -1, type = float)
    
    parser.add_argument("-oih", "--omega_i_plot_limit_high", dest = "oih", action = "store",
                        default = -0.4, type = float)

    parser.add_argument("-ft", "--ftol", dest = "ftol", action = "store",
                        default = 1e-11, type = float)
    parser.add_argument("-xt", "--xtol", dest = "xtol", action = "store",
                        default = 1e-11, type = float)
    parser.add_argument("-gt", "--gtol", dest = "gtol", action = "store",
                        default = 1e-11, type = float)
    parser.add_argument("-mn", "--max_nfev", dest = "max_nfev", action = "store",
                        default = 200000, type = int)

    args = parser.parse_args()

    batch_runname = args.batch_runname
    runname = args.runname
    N_free = args.N_free
    guess_num = args.guess_num

    num = args.num

    load_pickle = args.load_pickle
    l = args.l
    m = args.m
    seed = args.seed
    weighted = args.weighted

    eff = args.eff
    eff_i = args.eff_i
    delay = args.delay

    tl = args.tl
    th = args.th 
    tn = args.tn

    A_rel = args.A_rel

    Al = args.Al
    Ah = args.Ah
    pl = args.pl
    ph = args.ph
    rl = args.rl
    rh = args.rh
    il = args.il
    ih = args.ih

    oih = args.oih

    ftol = args.ftol
    xtol = args.xtol
    gtol = args.gtol
    max_nfev = args.max_nfev

    if eff:
        runname_full = f"{batch_runname}_{eff_i:03d}_Nfree_{N_free}_{runname}"
        run_string_prefix = f"{batch_runname}_{eff_i:03d}_convergence_{runname}"

        inject_params = read_json_for_param_dict(eff_i, batch_runname)
        h = make_eff_ringdown_waveform_from_param(inject_params, delay = delay)
    else:
        SXS_num = batch_runname
        runname_full = f"{SXS_num}_lm_{l}{m}_Nfree_{N_free}_{runname}"
        run_string_prefix = f"{SXS_num}_lm_{l}{m}_convergence_{runname}"


        h, _, _, _, _ = get_waveform_SXS(SXS_num, l, m)

    if A_rel:
        A_val = np.abs(h.h[0])
    else:
        A_val = 1

    rng = np.random.RandomState(seed)
    A_guesses = A_val*10**(rng.uniform(Al, Ah, size = (guess_num, N_free)))
    phi_guesses = rng.uniform(pl, ph, size = (guess_num, N_free))
    omegar_guesses = rng.uniform(rl, rh, size = (guess_num, N_free))
    omegai_guesses = rng.uniform(il, ih, size = (guess_num, N_free))

    guesses_stack = np.empty((guess_num, 4 * N_free), dtype = A_guesses.dtype)
    guesses_stack[:,0::4] = A_guesses 
    guesses_stack[:,1::4] = phi_guesses 
    guesses_stack[:,2::4] = omegar_guesses
    guesses_stack[:,3::4] = omegai_guesses

    guess_list = [list(guess) for guess in guesses_stack]

    t0_arr = np.linspace(tl, th, num = tn)
    qnm_fixed_list = []
    N_fix = len(qnm_fixed_list)
    time_longest, _, _ = h.postmerger(t0_arr[0])
    jcf = CurveFit(flength=2 * len(time_longest))
    fit_kwargs = {
        'ftol': ftol, 'xtol': xtol, 'gtol': gtol,
        # 'loss': 'soft_l1', 
        # 'x_scale': 'jac',
                  }

    number_of_cores = int(os.environ['SLURM_CPUS_PER_TASK'])
    print(number_of_cores)

    if num == -1:
        guess_num_list = list(range(guess_num))
    else:
        guess_num_list = [num]
   
    if number_of_cores == 1:
        fit_results = []
        for i in guess_num_list:
            params0 = jnp.array([1, 1] * N_fix + guess_list[i])
            fitter = QNMFitVaryingStartingTime(h, t0_arr, N_free = N_free,
                                            qnm_fixed_list = qnm_fixed_list, jcf = jcf,
                                            load_pickle = load_pickle,
                                            params0 = params0, 
                                            run_string_prefix = f"{run_string_prefix}_{i}",
                                            max_nfev = max_nfev,
                                            weighted = weighted,
                                            fit_kwargs = fit_kwargs)
            fitter.do_fits()
            fit_results.append(fitter.result_full)
            jcf = fitter.jcf

    else:
        fit_results = []
        do_one_fit_i = partial(
                    do_one_fit,
                    N_fix, guess_list, batch_runname, l, m, 
                    t0_arr, N_free, qnm_fixed_list,
                    load_pickle, run_string_prefix, fit_kwargs,
                    None, False, eff, eff_i, delay)
        processes = []
        for i in range(guess_num):
            process = Process(target = do_one_fit_i, args = (i, ))
            processes.append(process)
        for i in range(guess_num):
            processes[i].start()
        for i in range(guess_num):
            processes[i].join()
        # with Pool(number_of_cores) as pool:
        #     fit_results = pool.map(do_one_fit_i, list(range(guess_num)))

    if num == -1:
        return
    
    return
    
    mismatch_all = []
    for j, result_full in enumerate(fit_results):
        mismatch = result_full.mismatch_arr
        mismatch_all.append(mismatch[0])
    mismatch_all = np.array(mismatch_all)
    suc_num = len(mismatch_all[mismatch_all < 1.001*np.amin(mismatch_all)])

    fig, ax = plt.subplots()
    ax.hist(mismatch_all)
    ax.set_title(f'Success: {suc_num}/{guess_num}', fontsize = 20)
    ax.set_xlabel('Mismatch', fontsize = 16)
    ax.set_ylabel('count', fontsize = 16)

    plt.savefig(PLOT_PATH + f"{runname_full}_mismatch_hist.pdf", bbox_inches = "tight")
    plt.savefig(PLOT_PATH + f"{runname_full}_mismatch_hist.png", dpi = 150, bbox_inches = "tight",
                facecolor='white', transparent=False)

    if tn > 0:
        fig, ax = plt.subplots()
        for j, result_full in enumerate(fit_results):
            mismatch = result_full.mismatch_arr
            ax.semilogy(t0_arr, mismatch)
        ax.set_xlabel(r'$(t - t_{\rm peak}) / M$', fontsize = 16)
        ax.set_ylabel('Mismatch', fontsize = 16)
        plt.savefig(PLOT_PATH + f"{runname_full}_mismatch_vs_t.pdf", bbox_inches = "tight")
        plt.savefig(PLOT_PATH + f"{runname_full}_mismatch_vs_t.png", dpi = 150)

    fig, axs = plt.subplots(1, 2, figsize = (12, 5))

    for j, result_full in enumerate(fit_results):
        omega_dict = result_full.omega_dict
        for i in range(N_free):
            axs[0].scatter(t0_arr,
                        omega_dict['real'][f'omega_r_free_{i}'], 
                        s = 5, c = f'C{j}', alpha = 0.3)
            axs[1].scatter(t0_arr, 
                        omega_dict['imag'][f'omega_i_free_{i}'], 
                        s = 5, c = f'C{j}', alpha = 0.3)
    axs[0].set_ylim(-1, 2)
    axs[1].set_ylim(0.05, oih)
    axs[0].set_xlabel(r"$(t_0 - t_{\rm peak})/M$", fontsize = 16)
    axs[1].set_xlabel(r"$(t_0 - t_{\rm peak})/M$", fontsize = 16)
    axs[0].set_ylabel(r"$\omega_r$", fontsize = 16)
    axs[1].set_ylabel(r"$\omega_i$", fontsize = 16)
    fig.suptitle(f"N = {N_free}", fontsize = 24)

    plt.savefig(PLOT_PATH_ROOT + f"convergence_{runname_full}.pdf", bbox_inches = "tight")
    plt.savefig(PLOT_PATH_ROOT + f"convergence_{runname_full}.png", dpi = 150, bbox_inches = "tight",
                facecolor='white', transparent=False)


if __name__ == "__main__":
    main()
