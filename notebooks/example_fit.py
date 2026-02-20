import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Example: fitting a waveform in the SXS catalog
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Import modules
    """)
    return


@app.cell
def _():
    from jaxqualin.qnmode import mode, mode_list
    from jaxqualin.waveforms import get_waveform_SXS
    from jaxqualin.fit import QNMFitVaryingStartingTime
    from jaxqualin.plot import plot_amplitudes, plot_phases, plot_omega_free, plot_predicted_qnms

    import numpy as np
    import matplotlib.pyplot as plt

    return (
        QNMFitVaryingStartingTime,
        get_waveform_SXS,
        mode_list,
        np,
        plot_amplitudes,
        plot_omega_free,
        plot_phases,
        plot_predicted_qnms,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Getting the SXS waveform

    The waveform will be downloaded with the `sxs` package, if not already.
    """)
    return


@app.cell
def _(get_waveform_SXS):
    SXSnum = "0305"
    l = 2
    m = 2

    h, Mf, af, Lev = get_waveform_SXS(SXSnum, l, m)
    return Mf, SXSnum, af, h, l, m


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Case 1: Fit model with free QNMs (unfixed frequencies)

    Here we will fit with three free QNMs ($N_{f} = 3$) and see if the resulting frequencies approach any of the QNMs we expect to find.
    By default, the fit results are saved with `pickle` into `./.jaxqualin_cache/fits/`.
    If `load_pickle` is `True`, the fitter will load the pickled result that matches `run_string_prefix` and the list of modes used.
    """)
    return


@app.cell
def _(QNMFitVaryingStartingTime, SXSnum, h, l, m, np):
    _t0_arr = np.linspace(0, 50, num=501)  # array of starting times to fit for
    qnm_fixed_list = []  # t0 = 0 is the peak of the strain
    _run_string_prefix = f'SXS{SXSnum}_lm_{l}.{m}'  # list of QNMs with fixed frequencies in the fit model
    _N_free = 3  # prefix of pickle file for saving the results
    # fitter object
    fitter = QNMFitVaryingStartingTime(h, _t0_arr, N_free=_N_free, qnm_fixed_list=qnm_fixed_list, load_pickle=False, run_string_prefix=_run_string_prefix)  # number of free modes to use
    return (fitter,)


@app.cell
def _(fitter, mo):
    with mo.status.spinner("Running free-QNM fits..."):
        fitter.do_fits()
    mo.md("**Free-QNM fits complete.**")
    return


@app.cell
def _(fitter):
    # fitter results object
    result = fitter.result_full
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting the results

    Different colored points trace out the frequency evolution of a free QNM from early to late $t_0$.
    """)
    return


@app.cell
def _(Mf, af, mode_list, plot_omega_free, plot_predicted_qnms, plt, result):
    _fig, _ax = plt.subplots()
    predicted_qnms = mode_list(['2.2.0', '2.2.1', '3.2.0'], Mf, af)
    # mode locations to visualize on the plot
    plot_omega_free(result, _ax)
    plot_predicted_qnms(_ax, predicted_qnms)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Case 2: Fixed QNMs (fixed frequencies)

    Here we assume that the $2{,}2{,}0, 2{,}2{,}1$ and $3{,}3{,}0$ modes are present, so we use a fit model including these three modes and fix their frequencies according to GR with the help of the `qnm` package.
    We will use $N_f = 0$, meaning that we do not include additional free QNMs on top of the three fixed modes.
    """)
    return


@app.cell
def _(Mf, QNMFitVaryingStartingTime, SXSnum, af, h, l, m, mode_list, np):
    _t0_arr = np.linspace(0, 50, num=501)
    qnm_fixed_list_1 = mode_list(['2.2.0', '2.2.1', '3.2.0'], Mf, af)
    _run_string_prefix = f'SXS{SXSnum}_lm_{l}.{m}'
    _N_free = 0
    fitter_1 = QNMFitVaryingStartingTime(h, _t0_arr, N_free=_N_free, qnm_fixed_list=qnm_fixed_list_1, load_pickle=False, run_string_prefix=_run_string_prefix)
    return fitter_1, qnm_fixed_list_1


@app.cell
def _(fitter_1, mo):
    with mo.status.spinner("Running fixed-QNM fits..."):
        fitter_1.do_fits()
    mo.md("**Fixed-QNM fits complete.**")
    return


@app.cell
def _(fitter_1):
    result_1 = fitter_1.result_full
    return (result_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We plot the amplitude and phase of the fixed modes, as a function of $t_0$
    """)
    return


@app.cell
def _(plot_amplitudes, plot_phases, plt, qnm_fixed_list_1, result_1):
    _fig, _axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_amplitudes(result_1, fixed_modes=qnm_fixed_list_1, ax=_axs[0])
    plot_phases(result_1, fixed_modes=qnm_fixed_list_1, ax=_axs[1], legend=False)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Flatness diagnostics on fixed-frequency fit

    We now use the same stability tolerance as the mode-search workflow (`fluc_tol = 0.2`).
    The bolded segments are the flattest windows of width $\Delta T = 10 M$, and the circle markers show the earliest window start that satisfies the flatness tolerance.
    """)
    return


@app.cell
def _(plot_amplitudes, plot_phases, plt, qnm_fixed_list_1, result_1):
    flatness_summary = result_1.summarize_fixed_mode_flatness(delta_t=10.0, fluc_tol=0.2)
    bold_dict, t_flat_start_dict = result_1.fixed_mode_flatness_plot_overlays(delta_t=10.0, fluc_tol=0.2)

    _fig, _axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_amplitudes(
        result_1,
        fixed_modes=qnm_fixed_list_1,
        ax=_axs[0],
        use_label=False,
        bold_dict=bold_dict,
        alpha=0.3,
        t_flat_start_dict=t_flat_start_dict,
        flat_start_s=50,
        flat_start_marker="o",
    )
    _axs[0].legend(fontsize=9)
    plot_phases(
        result_1,
        fixed_modes=qnm_fixed_list_1,
        ax=_axs[1],
        legend=False,
        bold_dict=bold_dict,
        alpha=0.3,
        t_flat_start_dict=t_flat_start_dict,
        flat_start_s=50,
        flat_start_marker="o",
    )

    _fig
    return (flatness_summary,)


@app.cell
def _(flatness_summary, mo, np):
    _lines = [
        "Per-mode flattest-window results:",
    ]
    for _mode_string_summary, info in flatness_summary.items():
        _earliest = info["earliest_flat_start_time"]
        if np.isnan(_earliest):
            _earliest_txt = "nan (no qualifying window)"
        else:
            _earliest_txt = f"{_earliest:.2f}"
        _lines.append(
            f"- `{_mode_string_summary}`: flattest window [{info['flattest_start_time']:.2f}, {info['flattest_end_time']:.2f}] M, "
            f"A={info['flattest_amplitude_median']:.4g} (+{info['flattest_amplitude_plus']:.3g}/-{info['flattest_amplitude_minus']:.3g}), "
            f"phi={info['flattest_phase_median']:.4g} (+{info['flattest_phase_plus']:.3g}/-{info['flattest_phase_minus']:.3g}), "
            f"earliest flat start={_earliest_txt} M"
        )
    mo.md("\n".join(_lines))
    return


if __name__ == "__main__":
    app.run()
