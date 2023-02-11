import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import (
    MultipleLocator, AutoMinorLocator, LogLocator, NullFormatter)
import numpy as np
from QuasinormalMode import *
from ModeSelection import *
from bisect import bisect_right

from scipy.odr import Model, ODR, RealData

import os
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PLOT_SAVE_PATH = os.path.join(ROOT_PATH, "plots/")

plt.rc('text', usetex=False)
plt.rc('font', family='qpl')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['ytick.minor.size'] = 2

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.left'] = plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.labeltop'] = plt.rcParams['ytick.labelright'] = False
mpl.rcParams['axes.unicode_minus'] = False

params = {'axes.labelsize': 18,
          'font.family': 'serif',
          'font.size': 9,
          'legend.fontsize': 12,
          'xtick.labelsize': 11,
          'ytick.labelsize': 11,
          'savefig.dpi': 200,
          'lines.markersize': 6,
          'axes.formatter.limits': (-3, 3)}

mpl.rcParams.update(params)


def plot_omega_free(
        results_full,
        ax=None,
        plot_indxs=[],
        t0_min=None,
        t0_max=None,
        indicate_start = False,
        color = None,
        line_alpha = 0.3,
        scatter_alpha = 0.5):
    omega_dict = results_full.omega_dict
    t0_arr = results_full.t0_arr
    if t0_min is not None:
        t0_min_indx = bisect_right(t0_arr, t0_min)
    else:
        t0_min_indx = 0
    if t0_max is not None:
        t0_max_indx = bisect_right(t0_arr, t0_max)
    else:
        t0_max_indx = len(t0_arr)

    if ax is None:
        fig, ax = plt.subplots()
    omega_r_dict = omega_dict["real"]
    omega_i_dict = omega_dict["imag"]
    omega_r_list = list(omega_r_dict.values())
    omega_i_list = list(omega_i_dict.values())
    length = len(omega_r_dict)
    for i in range(length):
        if len(plot_indxs) == 0 or i in plot_indxs:
            if indicate_start:
                ax.scatter(omega_r_list[i][t0_min_indx],
                       omega_i_list[i][t0_min_indx], alpha=1, s=15, c = color)
            ax.plot(omega_r_list[i][t0_min_indx:t0_max_indx],
                    omega_i_list[i][t0_min_indx:t0_max_indx], alpha=line_alpha, c = color)
            ax.scatter(omega_r_list[i][t0_min_indx:t0_max_indx],
                       omega_i_list[i][t0_min_indx:t0_max_indx], alpha=scatter_alpha, s=1, c = color)
    ax.invert_yaxis()


def plot_predicted_qnms(
        ax,
        predicted_qnm_list,
        fix_indx = [],
        label_offset=(
            0,
            0.025),
        change_lim = True,
        facecolor="none",
        edgecolor="gray",
        cut_at_0 = False):
    ax.axvline(0, color='k', ls='--')
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlim(max(xmin, -2), min(xmax, 2))
    if change_lim:
        if cut_at_0:
            ax.set_ylim(0, max(ymax, -0.7))
        else:
            ax.set_ylim(0.05, max(ymax, -0.7))
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for i, mode in enumerate(predicted_qnm_list):
        if xmin < mode.omegar < xmax and ymax < mode.omegai < ymin:  # remember that y-axis is flipped
            if i in fix_indx:
                ax.scatter(mode.omegar, mode.omegai, marker='o',
                       facecolor='k', edgecolor='k')
            else:
                ax.scatter(mode.omegar, mode.omegai, marker='o',
                           facecolor=facecolor, edgecolor=edgecolor)
            transform = ax.transData.transform((mode.omegar, mode.omegai))
            mode_ax_coord = ax.transAxes.inverted().transform(transform)
            label_ax_coord = mode_ax_coord + label_offset
            ax.text(
                *label_ax_coord,
                mode.tex_string(),
                color=edgecolor,
                transform=ax.transAxes,
                horizontalalignment="center")

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin, xmax)
    ax.axhspan(0, 1e2, color="gray", alpha=0.5)

    ax.set_xlabel("$\\omega_r$")
    ax.set_ylabel("$\\omega_i$")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def plot_M(
        results_full,
        Mf = None,
        ax=None,
        t0_min=None,
        t0_max=None):
    Ma_dict = results_full.Ma_dict
    t0_arr = results_full.t0_arr
    if t0_min is not None:
        t0_min_indx = bisect_right(t0_arr, t0_min)
    else:
        t0_min_indx = 0
    if t0_max is not None:
        t0_max_indx = bisect_right(t0_arr, t0_max)
    else:
        t0_max_indx = len(t0_arr)

    if ax is None:
        fig, ax = plt.subplots()
    M = Ma_dict["M"]
    ax.plot(t0_arr[t0_min_indx:t0_max_indx], M[t0_min_indx:t0_max_indx], alpha = 0.3)
    if Mf is not None:
        ax.axhline(Mf, c = 'k', alpha = 0.5)
    ax.set_xlabel(r"$(t_0 - t_{\rm peak})/M$")
    ax.set_ylabel(r"$M$")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
def plot_M_a(
        results_full,
        Mf = None,
        af = None,
        ax=None,
        t0_min=None,
        t0_max=None,
        color = None,
        indicate_start = False):
    Ma_dict = results_full.Ma_dict
    t0_arr = results_full.t0_arr
    if t0_min is not None:
        t0_min_indx = bisect_right(t0_arr, t0_min)
    else:
        t0_min_indx = 0
    if t0_max is not None:
        t0_max_indx = bisect_right(t0_arr, t0_max)
    else:
        t0_max_indx = len(t0_arr)

    if ax is None:
        fig, ax = plt.subplots()
    M = Ma_dict["M"]
    a = Ma_dict["a"]
    if indicate_start:
        ax.scatter(M[t0_min_indx], a[t0_min_indx], alpha=1, s=15, c = color)
    ax.scatter(M[t0_min_indx:t0_max_indx], a[t0_min_indx:t0_max_indx], alpha=0.5, s=1, c = color)
    ax.plot(M[t0_min_indx:t0_max_indx], a[t0_min_indx:t0_max_indx], alpha = 0.2, c = color)
    if (Mf is not None) and (af is not None):
        ax.axvline(Mf, c = 'k', alpha = 0.5)
        ax.axhline(af, c = 'k', alpha = 0.5)
    ax.set_xlabel(r"$M$")
    ax.set_ylabel(r"$a$")
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def plot_amplitudes(results_full, fixed_modes=None, ax=None, alpha = 1, ls = "-", use_label = True):
    colori = 0
    if ax is None:
        fig, ax = plt.subplots()
    A_fix_dict = results_full.A_fix_dict
    A_free_dict = results_full.A_free_dict
    t0_arr = results_full.t0_arr
    if fixed_modes is not None:
        fixed_mode_string_tex_list = qnms_to_tex_string(fixed_modes)
        fixed_mode_string_list = qnms_to_string(fixed_modes)
        for i, fixed_mode_string in enumerate(fixed_mode_string_list):
                if use_label:
                    label = fixed_mode_string_tex_list[i]
                else:
                    label = None
                ax.semilogy(t0_arr, np.abs(A_fix_dict[f"A_{fixed_mode_string}"]), 
                            lw=2, label=label, c = f"C{colori}",
                            alpha = alpha, ls = ls)
                colori += 1
    for A in list(A_free_dict.values()):
        ax.semilogy(t0_arr, np.abs(A), lw=1, c = f"C{colori}", alpha = alpha, ls = ls)
        colori += 1
    if fixed_modes is not None and use_label:
        ax.legend()

    ax.set_xlim(t0_arr[0], t0_arr[-1])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(r"$(t_0 - t_{\rm peak})/M$")
    ax.set_ylabel(r"$A$")
    
def plot_amplitudes_unadj(results_full, fixed_modes=None, ax=None, alpha = 1, ls = "-", use_label = True):
    colori = 0
    if ax is None:
        fig, ax = plt.subplots()
    A_fix_dict = results_full.A_fix_dict
    A_free_dict = results_full.A_free_dict
    t0_arr = results_full.t0_arr
    if fixed_modes is not None:
        fixed_mode_string_tex_list = qnms_to_tex_string(fixed_modes)
        fixed_mode_string_list = qnms_to_string(fixed_modes)
        for i, fixed_mode_string in enumerate(fixed_mode_string_list):
                if use_label:
                    label = fixed_mode_string_tex_list[i]
                else:
                    label = None
                ax.semilogy(t0_arr, np.exp(fixed_modes[i].omegai*t0_arr)*np.abs(A_fix_dict[f"A_{fixed_mode_string}"]), 
                            lw=2, label=label, c = f"C{colori}",
                            alpha = alpha, ls = ls)
                colori += 1
    for A in list(A_free_dict.values()):
        ax.semilogy(t0_arr, np.abs(A), lw=1, c = f"C{colori}", alpha = alpha, ls = ls)
        colori += 1
    if fixed_modes is not None and use_label:
        ax.legend()

    ax.set_xlim(t0_arr[0], t0_arr[-1])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(r"$(t_0 - t_{\rm peak})/M$")
    ax.set_ylabel(r"$A$")


def plot_phases(results_full, fixed_modes=None, ax=None, alpha = 1, ls = "-", use_label = True, shift_phase = True):
    colori = 0
    if ax is None:
        fig, ax = plt.subplots()
    phi_fix_dict = results_full.phi_fix_dict
    phi_free_dict = results_full.phi_free_dict
    A_fix_dict = results_full.A_fix_dict
    A_free_dict = results_full.A_free_dict
    t0_arr = results_full.t0_arr
    if fixed_modes is not None:
        fixed_mode_string_tex_list = qnms_to_tex_string(fixed_modes)
        fixed_mode_string_list = qnms_to_string(fixed_modes)
        for i, fixed_mode_string in enumerate(fixed_mode_string_list):
            phase_shift = np.where(A_fix_dict[f"A_{fixed_mode_string}"] > 0, 0, np.pi)
            t_breaks, phi_breaks = phase_break_for_plot(t0_arr, phi_fix_dict[f"phi_{fixed_mode_string}"] + phase_shift)
            for j, (t_break, phi_break) in enumerate(zip(t_breaks, phi_breaks)):
                if use_label:
                    label = fixed_mode_string_tex_list[i]
                else:
                    label = None
                if j == 0:
                    ax.plot(t_break, phi_break, lw=2,
                            c=f"C{i}", label = label, alpha = alpha, ls = ls)
                else:
                    ax.plot(t_break, phi_break, lw=2, c=f"C{i}", alpha = alpha, ls = ls)
            colori += 1
    for i, phi in enumerate(list(phi_free_dict.values())):
        t_breaks, phi_breaks = phase_break_for_plot(t0_arr, phi)
        for t_break, phi_break in zip(t_breaks, phi_breaks):
            ax.plot(t_break, phi_break, lw=1, c=f"C{colori + i}", ls = ls)
    ax.set_ylim(0, 2 * np.pi)
    if fixed_modes is not None and use_label:
        ax.legend()
    ax.set_xlim(t0_arr[0], t0_arr[-1])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(r"$(t_0 - t_{\rm peak})/M$")
    ax.set_ylabel(r"$\phi$")


def plot_mismatch(results_full, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    t0_arr = results_full.t0_arr
    mismatch_arr = results_full.mismatch_arr
    ax.semilogy(t0_arr, mismatch_arr, c='k')

    ax.set_xlim(t0_arr[0], t0_arr[-1])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(r"$(t_0 - t_{\rm peak})/M$")
    ax.set_ylabel(r"$\mathcal{M}$")


def plot_mode_distance(
        result_full,
        fixed_modes,
        omega_r_tol,
        omega_i_tol,
        ax=None):
    t0_arr = result_full.t0_arr
    if ax is None:
        fig, ax = plt.subplots()
    for mode in fixed_modes:
        delta = closest_free_mode_distance(result_full, mode,
                                           r_scale=omega_r_tol,
                                           i_scale=omega_i_tol)
        ax.semilogy(t0_arr, delta, lw=2, label=mode.tex_string())
    ax.legend()

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax)
    ax.axhspan(1, 1e20, color="gray", alpha=0.5)
    ax.set_xlim(t0_arr[0], t0_arr[-1])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel(r"$(t_0 - t_{\rm peak})/M$")
    ax.set_ylabel(r"$\tilde{\delta} \omega$")


def plot_lm_row(
        mode_searcher_vary_N,
        predicted_qnm_list=[],
        indx=None,
        axs=None,
        lm=None):
    if indx is None:
        indx = mode_searcher_vary_N.best_run_indx
    if not hasattr(axs, "__iter__"):
        fig, axs = plt.subplots(1, 7, figsize=(45, 5),
                                gridspec_kw={'width_ratios': [2, 1, 1, 1, 1, 1, 1]})
    mode_searcher = mode_searcher_vary_N.mode_searchers[indx]
    plot_omega_free(mode_searcher.full_fit.result_full,
                    ax=axs[0])
    plot_predicted_qnms(axs[0], predicted_qnm_list)
    plot_mode_distance(
        mode_searcher.full_fit.result_full,
        mode_searcher.found_modes,
        mode_searcher.mode_selector.omega_r_tol,
        mode_searcher.mode_selector.omega_i_tol,
        ax=axs[1])
    plot_amplitudes(mode_searcher.full_fit.result_full,
                    ax=axs[2])
    plot_phases(mode_searcher.full_fit.result_full,
                ax=axs[3])
    plot_amplitudes(mode_searcher_vary_N.fixed_fitters[indx].result_full,
                    fixed_modes=mode_searcher.found_modes, ax=axs[4])
    plot_phases(mode_searcher_vary_N.fixed_fitters[indx].result_full,
                fixed_modes=mode_searcher.found_modes, ax=axs[5])
    plot_mismatch(
        mode_searcher_vary_N.fixed_fitters[indx].result_full, ax=axs[6])
    if lm is not None:
        axs[0].text(0.95, 0.05, r"$\ell m = {}{}$".format(*lm), ha="right",
                    va="bottom", transform=axs[0].transAxes)


def plot_relevant_mode_search_full(
        mode_search_complete,
        predicted_qnm_list=[],
        indxs=None,
        postfix_string = "default"):

    varying_N_searcher_list = mode_search_complete.relevant_lm_mode_searcher_varying_N
    relevant_lm_list = mode_search_complete.relevant_lm_list
    n_rows = len(varying_N_searcher_list)
    if indxs is None:
        indxs = [None] * n_rows

    fig, ax_mat = plt.subplots(n_rows, 7, figsize=(45, 5 * n_rows),
                               gridspec_kw={'width_ratios': [2, 1, 1, 1, 1, 1, 1]})

    for i, ax_row in enumerate(ax_mat):
        if predicted_qnm_list == []:
            predicted_qnm_list_lm = varying_N_searcher_list[i].mode_searcher_vary_N.mode_searchers[0].potential_modes_full
        else:
            predicted_qnm_list_lm = predicted_qnm_list
        plot_lm_row(varying_N_searcher_list[i].mode_searcher_vary_N,
                    predicted_qnm_list=predicted_qnm_list_lm,
                    indx=indxs[i], axs=ax_row, lm=relevant_lm_list[i])

    fig.tight_layout()
    save_file_path = os.path.join(PLOT_SAVE_PATH, f"{mode_search_complete.SXSnum}_{postfix_string}.pdf")
    
    plt.savefig(save_file_path)


def phase_break_for_plot(times, phis_in):
    phis = phis_in % (2 * np.pi)
    timeslist = []
    phislist = []
    j = 0
    for i in range(len(phis) - 1):
        if (phis[i] < 1 and phis[i + 1] > 2 * np.pi - 1):
            if i - j > 0:
                timeslist.append(times[i - j:i + 2])
                phiseg = np.copy(phis[i - j:i + 2])
                phiseg[0] += nextadjust
            else:
                timeslist.append(times[i - j:i + 2])
                phiseg = np.copy(phis[i - j:i + 2])
            phiseg[-1] -= 2 * np.pi
            phislist.append(phiseg)
            nextadjust = 2 * np.pi
            j = 0
        elif (phis[i + 1] < 1 and phis[i] > 2 * np.pi - 1):
            if i - j > 0:
                timeslist.append(times[i - j:i + 2])
                phiseg = np.copy(phis[i - j:i + 2])
                phiseg[0] += nextadjust
            else:
                timeslist.append(times[i - j:i + 2])
                phiseg = np.copy(phis[i - j:i + 2])
            phiseg[-1] += 2 * np.pi
            phislist.append(phiseg)
            nextadjust = -2 * np.pi
            j = 0
        if i == len(phis) - 2:
            if i - j > 0:
                timeslist.append(times[i - j:i + 2])
                phiseg = np.copy(phis[i - j:i + 2])
                phiseg[0] += nextadjust
            else:
                timeslist.append(times[i - j:i + 2])
                phiseg = np.copy(phis[i - j:i + 2])
            phislist.append(phiseg)
        j += 1
    return timeslist, phislist


def mode_plot_3D(df, l, m, mode_string_pro, mode_string_retro):
    df_mode = df.loc[((df["l"] == l) & (df["m"] == m) & (df["mode_string"] == mode_string_pro) & (df["retro"] == False)) | 
              ((df["l"] == l) & (df["m"] == m) & (df["mode_string"] == mode_string_retro)& (df["retro"] == True))]
    xyz = df[["SXS_num", "chi_1_z", "chi_2_z", "q"]].drop_duplicates()
    df_missing = xyz[(~xyz["SXS_num"].isin(df_mode["SXS_num"]))]
    
    fig, axs = plt.subplots(2,2, figsize = (9, 7))
    
    axs[1,1].scatter(df_mode["chi_1_z"], df_mode["chi_2_z"], c=np.log10(df_mode["A_med"]))
    axs[1,1].scatter(df_missing["chi_1_z"], df_missing["chi_2_z"], c = "gray", alpha = 0.1)
    axs[1,1].set_xlabel(r"$\chi_1$")
    axs[1,1].set_ylabel(r"$\chi_2$")
    axs[1,0].scatter(df_mode["chi_1_z"], df_mode["q"], c=np.log10(df_mode["A_med"]))
    axs[1,0].scatter(df_missing["chi_1_z"], df_missing["q"], c = "gray", alpha = 0.1)
    axs[1,0].set_xlabel(r"$\chi_1$")
    axs[1,0].set_ylabel(r"$q$")
    axs[0,1].scatter(df_mode["chi_2_z"], df_mode["q"], c=np.log10(df_mode["A_med"]))
    axs[0,1].scatter(df_missing["chi_2_z"], df_missing["q"], c = "gray", alpha = 0.1)
    axs[0,1].set_xlabel(r"$\chi_2$")
    axs[0,1].set_ylabel(r"$q$")
    
    axs[0,0].remove()
    ax = fig.add_subplot(2,2,1,projection='3d')
    sc = ax.scatter3D(df_mode["chi_1_z"], df_mode["chi_2_z"], df_mode["q"], c = np.log10(df_mode["A_med"]))
    ax.scatter3D(df_missing["chi_1_z"], df_missing["chi_2_z"], df_missing["q"], c = "gray", alpha = 0.1)
    sc = ax.scatter3D(df_mode["chi_1_z"], df_mode["chi_2_z"], df_mode["q"], c = np.log10(df_mode["A_med"]))
    for i in range(len(sc.get_facecolors())):
        ax.plot([df_mode["chi_1_z"].to_numpy()[i], df_mode["chi_1_z"].to_numpy()[i]], 
                [df_mode["chi_2_z"].to_numpy()[i], df_mode["chi_2_z"].to_numpy()[i]], 
                [1, df_mode["q"].to_numpy()[i]], c = sc.get_facecolors()[i].tolist(), alpha = 0.5, lw = 1.5)
    ax.scatter3D(df_missing["chi_1_z"], df_missing["chi_2_z"], df_missing["q"], c = "gray", alpha = 0.2)
    for i in range(len(df_missing["chi_1_z"].to_numpy())):
        ax.plot([df_missing["chi_1_z"].to_numpy()[i], df_missing["chi_1_z"].to_numpy()[i]], 
                [df_missing["chi_2_z"].to_numpy()[i], df_missing["chi_2_z"].to_numpy()[i]], 
                [1, df_missing["q"].to_numpy()[i]], c = "gray", alpha = 0.2, lw = 1.5)
    ax.set_xlabel(r"$\chi_1$", fontsize = 12, labelpad = -4)
    ax.set_ylabel(r"$\chi_2$", fontsize = 12, labelpad = -4)
    ax.set_zlabel(r"$q$", fontsize = 12, labelpad = -4)
    ax.tick_params(axis = 'x', labelsize = 9, pad = -0.75)
    ax.tick_params(axis = 'y', labelsize = 9, pad = -0.75)
    ax.tick_params(axis = 'z', labelsize = 9, pad = -0.75)
    
    # fig.tight_layout()
    fig.subplots_adjust(right=0.9,wspace=0.25, hspace=0.3)
    ax.set_position([0,0.45,0.55,0.55])
    cb_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
    cbar = fig.colorbar(sc, cax=cb_ax)
    cb_ax.set_ylabel(r"$\log_{10} A$")

def linfunc(p, x):
    m, c= p
    return m * x + c 

def linfunc1(p, x):
    c = p
    return x + c 

def linfunc2(p, x):
    c = p
    return 2*x + c 
    
def plot_mode_vs_mode_amplitude(df, l1, m1, mode_string_pro_1, mode_string_retro_1,
                                l2, m2, mode_string_pro_2, mode_string_retro_2, fit_type = "agnostic"):
    df_1 = df.loc[((df["l"] == l1) & (df["m"] == m1) & (df["mode_string"] == mode_string_pro_1) & (df["retro"] == False)) | 
              ((df["l"] == l1) & (df["m"] == m1) & (df["mode_string"] == mode_string_retro_1)& (df["retro"] == True))]
    df_2 = df.loc[((df["l"] == l2) & (df["m"] == m2) & (df["mode_string"] == mode_string_pro_2) & (df["retro"] == False)) | 
              ((df["l"] == l2) & (df["m"] == m2) & (df["mode_string"] == mode_string_retro_2)& (df["retro"] == True))]
    df_merged = df_1.merge(df_2, on = "SXS_num", how = "inner", suffixes = ("_1", "_2"))
    xerr_low = df_merged["A_med_2"]-df_merged["A_low_2"]
    xerr_hi = df_merged["A_hi_2"]-df_merged["A_med_2"]
    yerr_low = df_merged["A_med_1"]-df_merged["A_low_1"]
    yerr_hi = df_merged["A_hi_1"]-df_merged["A_med_1"]
    xs = df_merged["A_med_2"]*df_merged["M_rem_1"]
    ys = df_merged["A_med_1"]*df_merged["M_rem_1"]
    
    if fit_type == "linear":
        fitfunc = linfunc1
        beta0 = [0.]
    if fit_type == "quadratic":
        fitfunc = linfunc2
        beta0 = [0.]
    if fit_type == "agnostic":
        fitfunc = linfunc
        beta0 = [1., 0.]
    lin_model = Model(fitfunc)
    errxlogs = (xerr_hi+xerr_low)/xs/np.log(10)
    errylogs = (yerr_hi+yerr_low)/ys/np.log(10)
    data = RealData(np.log10(xs), np.log10(ys), 
                    sx=errxlogs, sy=errylogs)
    odr = ODR(data, lin_model, beta0=beta0)
    out = odr.run()
    
    fig, ax = plt.subplots(figsize = (8,5))
    sc = ax.scatter(xs, ys, c = df_merged["chi_rem_1"], cmap = "cividis")
    plt.draw()
    for i in range(len(sc.get_facecolors())):
        ax.errorbar(xs[i], ys[i], xerr = ([xerr_low.to_numpy()[i]], [xerr_hi.to_numpy()[i]]),
                 yerr = ([yerr_low.to_numpy()[i]], [yerr_hi.to_numpy()[i]]), ecolor = sc.get_facecolors()[i].tolist(),
                     fmt = "None")
    cb = fig.colorbar(sc, ax = ax)
    cb.ax.set_ylabel(r"$\chi_{\rm rem}$")
    xsfit = np.linspace(*ax.get_xlim(), num = 100)
    ysfit = fitfunc(out.beta, np.log10(xsfit))
    # ax.loglog(xsfit, 10**ysfit, c = "k", ls = ":")
    xlabel_string = r"$A_{{{}}}$".format(mode_string_pro_2)
    ylabel_string = r"$A_{{{}}}$".format(mode_string_pro_1)
    ax.set_xlabel(xlabel_string.replace('x', r" \times "))
    ax.set_ylabel(ylabel_string.replace('x', r" \times "))
    return fig, ax
    
def plot_mode_vs_mode_phase(df, l1, m1, mode_string_pro_1, mode_string_retro_1,
                                l2, m2, mode_string_pro_2, mode_string_retro_2, fit_type = "quadratic"):
    if fit_type == "quadratic":
        fit_fac = 2
    elif fit_type == "linear":
        fit_fac = 1
    else:
        raise ValueError
    df_1 = df.loc[((df["l"] == l1) & (df["m"] == m1) & (df["mode_string"] == mode_string_pro_1) & (df["retro"] == False)) | 
              ((df["l"] == l1) & (df["m"] == m1) & (df["mode_string"] == mode_string_retro_1)& (df["retro"] == True))]
    df_2 = df.loc[((df["l"] == l2) & (df["m"] == m2) & (df["mode_string"] == mode_string_pro_2) & (df["retro"] == False)) | 
              ((df["l"] == l2) & (df["m"] == m2) & (df["mode_string"] == mode_string_retro_2)& (df["retro"] == True))]
    df_merged = df_1.merge(df_2, on = "SXS_num", how = "inner", suffixes = ("_1", "_2"))    
    xerr_low = df_merged["phi_med_2"]-df_merged["phi_low_2"]
    xerr_hi = df_merged["phi_hi_2"]-df_merged["phi_med_2"]
    yerr_low = df_merged["phi_med_1"]-df_merged["phi_low_1"]
    yerr_hi = df_merged["phi_hi_1"]-df_merged["phi_med_1"]
    xs = df_merged["phi_med_2"]
    ys = df_merged["phi_med_1"]
    fig, ax = plt.subplots(figsize = (8,5))
    sc = ax.scatter(fit_fac*xs%(2*np.pi), ys%(2*np.pi), c = df_merged["chi_rem_1"], cmap = "cividis")
    plt.draw()
    for i in range(len(sc.get_facecolors())):
        ax.errorbar(fit_fac*xs[i]%(2*np.pi), ys[i]%(2*np.pi), xerr = ([xerr_low.to_numpy()[i]], [xerr_hi.to_numpy()[i]]),
                 yerr = ([yerr_low.to_numpy()[i]], [yerr_hi.to_numpy()[i]]), ecolor = sc.get_facecolors()[i].tolist(),
                     fmt = "None")
    cb = fig.colorbar(sc, ax = ax)
    cb.ax.set_ylabel(r"$\chi_{\rm rem}$")
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(0, 2*np.pi)
    xsfit = np.linspace(0, 2*np.pi, num = 100)
    ysfit = xsfit
    # ax.plot(xsfit, ysfit, c = "k", ls = ":")
    xlabel_string = r"${}\phi_{{{}}}$".format(fit_fac, mode_string_pro_2)
    ylabel_string = r"$\phi_{{{}}}$".format(mode_string_pro_1)
    ax.set_xlabel(xlabel_string.replace('x', r" \times "))
    ax.set_ylabel(ylabel_string.replace('x', r" \times "))
    return fig, ax
