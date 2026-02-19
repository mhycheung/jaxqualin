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
    # Example: Fitting your own waveform
    """)
    return


@app.cell
def _():
    from jaxqualin.waveforms import delayed_QNM, waveform
    from jaxqualin.qnmode import mode, mode_list
    from jaxqualin.fit import QNMFitVaryingStartingTime
    from jaxqualin.selection import ModeSearchAllFreeVaryingN
    from jaxqualin.plot import (plot_amplitudes, plot_phases, 
                                plot_omega_free, plot_predicted_qnms,
                                plot_mode_searcher_results)

    import numpy as np
    import matplotlib.pyplot as plt

    return (
        ModeSearchAllFreeVaryingN,
        QNMFitVaryingStartingTime,
        delayed_QNM,
        mode_list,
        np,
        plot_amplitudes,
        plot_mode_searcher_results,
        plot_omega_free,
        plot_phases,
        plot_predicted_qnms,
        plt,
        waveform,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Make waveform

    As example, we will make a waveform that contains three QNMs that are distorted close to the merger peak.
    You can replace this with whatever time domain waveform you would like to fit.
    """)
    return


@app.cell
def _(delayed_QNM, mode_list, np):
    Mf = 1
    af = 0.7
    modes = mode_list(['2.2.0', '2.2.1', '3.2.0'], Mf, af)
    A_phi_dict = {'2.2.0': dict(A=1.0, phi=0.0), '2.2.1': dict(A=3.0, phi=np.pi / 2), '3.2.0': dict(A=0.01, phi=np.pi)}
    t_arr = np.linspace(0, 120, 1000)
    h_arr = np.zeros(t_arr.shape, dtype=np.complex128)
    for i, _mode in enumerate(modes):
        if i == 0:
            A_delay = 0
            A_sig = 10
            phi_sig = 5
        else:
            A_delay = 5
            A_sig = 2
            phi_sig = 2
        h_arr = h_arr + delayed_QNM(_mode, t_arr, A_phi_dict[_mode.string()]['A'], A_phi_dict[_mode.string()]['phi'], A_delay=A_delay, A_sig=A_sig, phi_sig=phi_sig)
    return Mf, af, h_arr, t_arr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Make waveform object

    You can pass in whatever time domain waveform.
    If you know that the peak strain is located at t = 0, then pass in `t_peak = 0`, or else the waveform object detects the peak on its own.
    If `t_peak` is not passed, the first `remove_num = 500` data points are removed by default because BBH merger simulations often contain junk radiation in the beginning.
    You can set `remove_num = 0` if you do not want to remove any data points.
    """)
    return


@app.cell
def _(h_arr, t_arr, waveform):
    h = waveform(t_arr, h_arr, t_peak = 0)
    return (h,)


@app.cell
def _(h, np, plt):
    _fig, _ax = plt.subplots()
    _ax.semilogy(h.time, np.abs(h.hr))
    _ax.set_xlabel('$t$')
    _ax.set_ylabel('$|h_r|$')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Free QNMs (unfixed frequencies)
    """)
    return


@app.cell
def _(QNMFitVaryingStartingTime, h, np):
    t0_arr = np.linspace(0, 50, num=101)  # array of starting times to fit for
    qnm_fixed_list = []  # t0 = 0 is the peak of the strain
    _run_string_prefix = f'custom_example_lm_2.2'  # list of QNMs with fixed frequencies in the fit model
    _N_free = 3  # prefix of pickle file for saving the results
    # fitter object
    fitter = QNMFitVaryingStartingTime(h, t0_arr, N_free=_N_free, qnm_fixed_list=qnm_fixed_list, load_pickle=False, run_string_prefix=_run_string_prefix)  # number of free modes to use
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
    ### Fixed QNMs (fixed frequencies)
    """)
    return


@app.cell
def _(Mf, QNMFitVaryingStartingTime, af, h, mode_list, np):
    t0_arr_1 = np.linspace(0, 50, num=101)
    qnm_fixed_list_1 = mode_list(['2.2.0', '2.2.1', '3.2.0'], Mf, af)
    _run_string_prefix = f'custom_example_lm_2.2'
    _N_free = 0
    fitter_1 = QNMFitVaryingStartingTime(h, t0_arr_1, N_free=_N_free, qnm_fixed_list=qnm_fixed_list_1, load_pickle=False, run_string_prefix=_run_string_prefix)
    return fitter_1, qnm_fixed_list_1, t0_arr_1


@app.cell
def _(fitter_1, mo):
    with mo.status.spinner("Running fixed-QNM fits..."):
        fitter_1.do_fits()
    result_1 = fitter_1.result_full
    mo.md("**Fixed-QNM fits complete.**")
    return (result_1,)


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
    ### Mode search

    With `BBH_potential_modes = True` (default), the mode searcher uses a list of modes that are expected to be seen in the waveform in a binary-black-hole merger, which depends on the waveform harmonic in question. For a custom waveform, the harmonic can be specified by `h.set_lm(l, m)`. The modesearcher will include all the overtones in that harmonic, spheroidal mixing modes with the same `m` but another `l`, and quadratic modes with `m_1 + m_2 = m`. mode mixing from other harmonics (e.g. due to not working in the superrest BMS frame) can be included by specifying `relevant_lm_list`. Consult the source code of the `potential_modes` function in `jaxqualin.qnmode` for more details.

    Additional custom modes can be included with the `potential_modes_custom` keyword argument. If `BBH_potential_modes = False` then these will be the only potential modes that the mode searcher will try to find.
    """)
    return


@app.cell
def _(Mf, ModeSearchAllFreeVaryingN, af, h, mode_list, t0_arr_1):
    h.set_lm(2, 2)
    relevant_lm_list = [(2, 2)]
    potential_modes_custom = qnm_fixed_list_2 = mode_list(['-2.2.0x2.2.0'], Mf, af)
    _run_string_prefix = f'custom_example_lm_2.2'
    mode_searcher = ModeSearchAllFreeVaryingN(h, Mf, af, relevant_lm_list=relevant_lm_list, N_list=[3], t0_arr=t0_arr_1, run_string_prefix=_run_string_prefix, BBH_potential_modes=True, potential_modes_custom=potential_modes_custom)
    return (mode_searcher,)


@app.cell
def _(mo, mode_searcher):
    with mo.status.spinner("Running mode searches..."):
        mode_searcher.do_mode_searches()
    mo.md("**Mode searches complete.**")
    return


@app.cell
def _(mode_searcher, plot_mode_searcher_results, plt):
    plot_mode_searcher_results(mode_searcher)
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
