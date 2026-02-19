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
    # Example: Mode search in an SXS waveform

    In this example we will be using the procedure listed out in the methods paper to search for QNMs within a waveform.
    The two-stage procedure starts by identifing potential modes via a frequency-agnostic fits, then checks their stability with frequency-fixed fits.
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
    from jaxqualin.selection import ModeSearchAllFreeVaryingNSXS
    from jaxqualin.utils import load_pickle_file
    from jaxqualin.plot import plot_mode_searcher_results

    import numpy as np
    import matplotlib.pyplot as plt

    return (
        ModeSearchAllFreeVaryingNSXS,
        load_pickle_file,
        np,
        plot_mode_searcher_results,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Settings for mode-searcher object

    These are the default settings used in the methods paper (cf. Table I).
    """)
    return


@app.cell
def _():
    settings = dict(alpha_r = 0.05, alpha_i = 0.05,
                    tau_agnostic = 10, p_agnostic = 0.95, 
                    beta_A = 1.0, beta_phi = 1.5, A_tol = 1e-3,
                    tau_stable = 10, p_stable = 0.95)
    return (settings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prepare mode-searcher object

    `N_list` is a list of $N_f$ free QNMs to include in the frequency agnostic fit.
    A frequency agnoistic fit will be performed for each $N_f$ in `N_list`, and a list of potential modes will be compiled from the best run among them.
    """)
    return


@app.cell
def _(ModeSearchAllFreeVaryingNSXS, np, settings):
    mode_searcher_load_pickle = False # whether or not to load the mode-searcher from a cached run

    SXS_num = '0305'
    l, m = 2, 2
    N_list = [3, 4]
    t0_arr = np.linspace(0, 50, 501)

    # mode-searcher object
    mode_search_sxs = ModeSearchAllFreeVaryingNSXS(SXS_num, l, m, N_list = N_list, initial_num = 10, 
                                                   random_initial = True, load_pickle = False,
                                                   t0_arr = t0_arr, postfix_string = 'example',
                                                   **settings)
    return mode_search_sxs, mode_searcher_load_pickle


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Run mode-searcher

    If `mode_searcher_load_pickle` is `True` and a pickle file from a previous run with the same `postfix_string` exists, the mode-searcher object will load the results from it.
    """)
    return


@app.cell
def _(load_pickle_file, mo, mode_search_sxs, mode_searcher_load_pickle):
    if mode_search_sxs.pickle_exists() and mode_searcher_load_pickle:
        _file_path = mode_search_sxs.file_path
        mode_search_sxs_result = load_pickle_file(_file_path)
        _status_msg = "Loaded mode search results from pickle."
    else:
        with mo.status.spinner("Running mode search..."):
            mode_search_sxs.do_mode_search_varying_N()
        mode_search_sxs_result = mode_search_sxs
        _status_msg = "Mode search complete."
    mo.md(f"**{_status_msg}**")
    return (mode_search_sxs_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting the results

    The left panel shows the result of the best instance of the frequency-agnostic fits among `N_list`.
    The center and right panels are the amplitudes and phases obtained for all the modes within the potential mode list that passed the stability test.
    The bolded line segments are the regions where the amplitude and phase are the flattest, and the circle marks the time at which the mode has begun to stabilize.
    Please consult the methods paper for the details.
    """)
    return


@app.cell
def _(mode_search_sxs_result, plot_mode_searcher_results, plt):
    plot_mode_searcher_results(mode_search_sxs_result.mode_searcher_vary_N)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summary of per-mode results
    """)
    return


@app.cell
def _(mo, mode_search_sxs_result, np):
    _summary = mode_search_sxs_result.summarize_final_modes()
    _lines = [f"Final modes present: {', '.join(_summary.keys())}", ""]
    for _mode_string_ms, _info in _summary.items():
        _earliest = _info["earliest_flat_start_time"]
        if np.isnan(_earliest):
            _earliest_txt = "nan (no qualifying window)"
        else:
            _earliest_txt = f"{_earliest:.2f}"
        _lines.append(
            f"- `{_mode_string_ms}`: flattest window [{_info['flattest_start_time']:.2f}, {_info['flattest_end_time']:.2f}] M, "
            f"A={_info['flattest_amplitude_median']:.4g} (+{_info['flattest_amplitude_plus']:.3g}/-{_info['flattest_amplitude_minus']:.3g}), "
            f"phi={_info['flattest_phase_median']:.4g} (+{_info['flattest_phase_plus']:.3g}/-{_info['flattest_phase_minus']:.3g}), "
            f"earliest flat start={_earliest_txt} M"
        )
    mo.md("\n".join(_lines))
    return


if __name__ == "__main__":
    app.run()
