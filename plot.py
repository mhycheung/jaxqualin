import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, LogLocator, NullFormatter)
import numpy as np
from QuasinormalMode import *
from ModeSelection import *
from bisect import bisect_right

plt.rc('text', usetex=True)
plt.rc('font', family='qpl')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['ytick.minor.size'] = 2

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top']  = True
plt.rcParams['ytick.left'] = plt.rcParams['ytick.right']  = True
plt.rcParams['xtick.labeltop'] = plt.rcParams['ytick.labelright'] = False
mpl.rcParams['axes.unicode_minus'] = False

params = {'axes.labelsize': 18,
          'font.family': 'serif',
          'font.size': 9,
          'legend.fontsize': 12,
          'xtick.labelsize': 11,
          'ytick.labelsize': 11,
          'savefig.dpi' : 200,
          'lines.markersize' : 6, 
          'axes.formatter.limits' : (-3,3)}

mpl.rcParams.update(params)

def plot_omega_free(results_full, ax = None, plot_indxs = [], t0_min = None, t0_max = None):
    omega_dict = results_full.omega_dict
    t0_arr = results_full.t0_arr
    if t0_min != None:
        t0_min_indx = bisect_right(t0_arr, t0_min)
    else:
        t0_min_indx = 0
    if t0_max != None:
        t0_max_indx = bisect_right(t0_arr, t0_max)
    else:
        t0_max_indx = len(t0_arr)
    
    if ax == None:
        fig, ax = plt.subplots()
    omega_r_dict = omega_dict["real"]
    omega_i_dict = omega_dict["imag"]
    omega_r_list = list(omega_r_dict.values())
    omega_i_list = list(omega_i_dict.values())
    length = len(omega_r_dict)
    for i in range(length):
        if len(plot_indxs) == 0 or i in plot_indxs:
            ax.plot(omega_r_list[i][t0_min_indx:t0_max_indx], omega_i_list[i][t0_min_indx:t0_max_indx], alpha = 0.3)
            ax.scatter(omega_r_list[i][t0_min_indx:t0_max_indx], omega_i_list[i][t0_min_indx:t0_max_indx], alpha = 0.5, s = 1)
    ax.invert_yaxis()
    
def plot_predicted_qnms(ax, predicted_qnm_list, label_offset = (0, 0.025), facecolor = "none", edgecolor = "gray"):
    ax.axvline(0, color = 'k', ls = '--')
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for mode in predicted_qnm_list:
        if xmin < mode.omegar < xmax and ymax < mode.omegai < ymin: # remember that y-axis is flipped
            ax.scatter(mode.omegar, mode.omegai, marker = 'o', 
                       facecolor = facecolor, edgecolor = edgecolor)
            transform = ax.transData.transform((mode.omegar, mode.omegai))
            mode_ax_coord = ax.transAxes.inverted().transform(transform)
            label_ax_coord = mode_ax_coord + label_offset
            ax.text(*label_ax_coord, mode.tex_string(), color = edgecolor, transform = ax.transAxes,
                    horizontalalignment = "center")
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax)
    ax.axhspan(0, 1e2, color = "gray", alpha = 0.5)
        
    ax.set_xlabel("$\omega_r$")
    ax.set_ylabel("$\omega_i$")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
def plot_amplitudes(results_full, fixed_modes = None, ax = None):
    if ax == None:
        fig, ax = plt.subplots()
    A_fix_dict = results_full.A_fix_dict
    A_free_dict = results_full.A_free_dict
    t0_arr = results_full.t0_arr   
    if fixed_modes != None:
        fixed_modes_string = qnms_to_tex_string(fixed_modes)
    for i, A in enumerate(list(A_fix_dict.values())):
        if fixed_modes == None:
            ax.semilogy(t0_arr, np.abs(A), lw = 2)
        else:
            ax.semilogy(t0_arr, np.abs(A), lw = 2, label = fixed_modes_string[i])
    for A in list(A_free_dict.values()):
        ax.semilogy(t0_arr, np.abs(A), lw = 1)
    if fixed_modes != None:
        ax.legend()
        
    ax.set_xlim(t0_arr[0], t0_arr[-1])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(r"$(t_0 - t_{\rm peak})/M$")
    ax.set_ylabel(r"$A$")
        
def plot_phases(results_full, fixed_modes = None, ax = None):
    if ax == None:
        fig, ax = plt.subplots()
    phi_fix_dict = results_full.phi_fix_dict
    phi_free_dict = results_full.phi_free_dict
    t0_arr = results_full.t0_arr
    if fixed_modes != None:
        fixed_modes_string = qnms_to_tex_string(fixed_modes)
    for i, phi in enumerate(list(phi_fix_dict.values())):
        t_breaks, phi_breaks = phase_break_for_plot(t0_arr, phi)
        for j, (t_break, phi_break) in enumerate(zip(t_breaks, phi_breaks)):
            if j == 0 and fixed_modes != None:
                ax.plot(t_break, phi_break, lw = 2, c = f"C{i}", label = fixed_modes_string[i])
            else:
                ax.plot(t_break, phi_break, lw = 2, c = f"C{i}")
    for i, phi in enumerate(list(phi_free_dict.values())):
        t_breaks, phi_breaks = phase_break_for_plot(t0_arr, phi)
        for t_break, phi_break in zip(t_breaks, phi_breaks):
            ax.plot(t_break, phi_break, lw = 1, c = f"C{i}")
    ax.set_ylim(0, 2*np.pi)
    if fixed_modes != None:
        ax.legend()
    ax.set_xlim(t0_arr[0], t0_arr[-1])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(r"$(t_0 - t_{\rm peak})/M$")
    ax.set_ylabel(r"$\phi$")
        

def plot_mismatch(results_full, ax = None):
    if ax == None:
        fig, ax = plt.subplots()
    t0_arr = results_full.t0_arr
    mismatch_arr = results_full.mismatch_arr
    ax.semilogy(t0_arr, mismatch_arr, c = 'k') 
    
    ax.set_xlim(t0_arr[0], t0_arr[-1])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(r"$(t_0 - t_{\rm peak})/M$")
    ax.set_ylabel(r"$\mathcal{M}$")


def plot_mode_distance(result_full, fixed_modes, omega_r_tol, omega_i_tol, ax = None):
    t0_arr = result_full.t0_arr
    if ax == None:
        fig, ax = plt.subplots()
    for mode in fixed_modes:
        delta = closest_free_mode_distance(result_full, mode,
                               r_scale = omega_r_tol,
                               i_scale = omega_i_tol)
        ax.semilogy(t0_arr, delta, lw = 2, label = mode.tex_string())
    ax.legend()
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    ax.axhspan(1, 1e20, color = "gray", alpha = 0.5)
    ax.set_xlim(t0_arr[0], t0_arr[-1])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(r"$(t_0 - t_{\rm peak})/M$")
    ax.set_ylabel(r"$\tilde{\delta} \omega$")
        
def plot_lm_row(mode_searcher_vary_N, predicted_qnm_list = [], indx = None, axs = None, lm = None):
    if indx == None:
        indx = mode_searcher_vary_N.best_run_indx
    if not hasattr(axs, "__iter__"):
        fig, axs = plt.subplots(1,5,figsize = (35,5),
                                gridspec_kw={'width_ratios': [2, 1, 1, 1, 1]})
    mode_searcher = mode_searcher_vary_N.mode_searchers[indx]
    plot_omega_free(mode_searcher.full_fit.result_full,
                    ax = axs[0])
    plot_predicted_qnms(axs[0], predicted_qnm_list)
    plot_mode_distance(mode_searcher.full_fit.result_full, mode_searcher.found_modes,
                       mode_searcher.mode_selector.omega_r_tol, mode_searcher.mode_selector.omega_i_tol,
                       ax = axs[1])
    plot_amplitudes(mode_searcher_vary_N.fixed_fitters[indx].result_full,
                    fixed_modes=mode_searcher.found_modes, ax = axs[2])
    plot_phases(mode_searcher_vary_N.fixed_fitters[indx].result_full,
                    fixed_modes=mode_searcher.found_modes, ax = axs[3])
    plot_mismatch(mode_searcher_vary_N.fixed_fitters[indx].result_full, ax = axs[4])
    if lm != None:
        axs[0].text(0.95, 0.05, r"$\ell m = {}{}$".format(*lm), ha = "right",
                va = "bottom", transform = axs[0].transAxes)

    
def plot_relevant_mode_search_full(mode_search_complete, predicted_qnm_list = [], indxs = None):
    
    varying_N_searcher_list = mode_search_complete.relevant_lm_mode_searcher_varying_N
    relevant_lm_list = mode_search_complete.relevant_lm_list
    n_rows = len(varying_N_searcher_list)
    if indxs == None:
        indxs = [None]*n_rows
        
    fig, ax_mat = plt.subplots(n_rows, 5, figsize = (35, 5*n_rows),
                               gridspec_kw={'width_ratios': [2, 1, 1, 1, 1]})
    
    for i, ax_row in enumerate(ax_mat):
        plot_lm_row(varying_N_searcher_list[i].mode_searcher_vary_N, 
                    predicted_qnm_list = predicted_qnm_list,
                    indx = indxs[i], axs = ax_row, lm = relevant_lm_list[i])
        
    fig.tight_layout()
    plt.savefig(f"./plots/{mode_search_complete.SXSnum}.pdf")
    
def phase_break_for_plot(times, phis_in):
    phis = phis_in %(2*np.pi)
    timeslist = []
    phislist = []
    j = 0
    for i in range(len(phis)-1):
        if (phis[i] < 1 and phis[i+1] > 2*np.pi - 1):
            if i - j > 0:
                timeslist.append(times[i-j:i+2])
                phiseg = np.copy(phis[i-j:i+2])
                phiseg[0] += nextadjust
            else:
                timeslist.append(times[i-j:i+2])
                phiseg = np.copy(phis[i-j:i+2])
            phiseg[-1] -= 2*np.pi
            phislist.append(phiseg)
            nextadjust = 2*np.pi
            j = 0
        elif (phis[i+1] < 1 and phis[i] > 2*np.pi - 1):
            if i - j > 0:
                timeslist.append(times[i-j:i+2])
                phiseg = np.copy(phis[i-j:i+2])
                phiseg[0] += nextadjust
            else:
                timeslist.append(times[i-j:i+2])
                phiseg = np.copy(phis[i-j:i+2])
            phiseg[-1] += 2*np.pi
            phislist.append(phiseg)
            nextadjust = -2*np.pi
            j = 0
        if i == len(phis)-2:
            if i - j > 0:
                timeslist.append(times[i-j:i+2])
                phiseg = np.copy(phis[i-j:i+2])
                phiseg[0] += nextadjust
            else:
                timeslist.append(times[i-j:i+2])
                phiseg = np.copy(phis[i-j:i+2])
            phislist.append(phiseg)
        j += 1
    return timeslist, phislist

