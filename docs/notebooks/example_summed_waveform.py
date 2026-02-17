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
    # Example: Fitting a waveform with summed multipoles
    """)
    return


@app.cell
def _():
    from jaxqualin.waveforms import get_SXS_waveform_summed
    from jaxqualin.qnmode import mode_list
    from jaxqualin.fit import QNMFitVaryingStartingTime
    from jaxqualin.plot import (plot_amplitudes, plot_phases, 
                                plot_omega_free, plot_predicted_qnms)

    import numpy as np
    import matplotlib.pyplot as plt

    return (
        QNMFitVaryingStartingTime,
        get_SXS_waveform_summed,
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
    ### Make waveform

    We will use a waveform that is a superposition of multipoles (up to `l_max = 4`) of the `SXS:BBH:0305` simulation.
    The observer is at the angular coordinate $(\iota, \psi)$.
    """)
    return


@app.cell
def _(get_SXS_waveform_summed, np):
    SXSnum = '0305'
    iota = np.pi/3 # Cotesta's angle
    psi = np.pi/2
    h, Mf, af = get_SXS_waveform_summed(SXSnum, iota, psi, l_max=4, res=0, N_ext=2)
    return Mf, SXSnum, af, h, iota, psi


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
def _(QNMFitVaryingStartingTime, SXSnum, h, iota, np, psi):
    _t0_arr = np.linspace(0, 50, num=101)  # array of starting times to fit for
    qnm_fixed_list = []  # t0 = 0 is the peak of the strain
    _run_string_prefix = f'SXS{SXSnum}_lm_2.2_iota_{iota:.7f}_psi_{psi:.7f}'  # list of QNMs with fixed frequencies in the fit model
    _N_free = 6  # prefix of pickle file for saving the results
    # fitter object
    fitter = QNMFitVaryingStartingTime(h, _t0_arr, N_free=_N_free, qnm_fixed_list=qnm_fixed_list, load_pickle=True, run_string_prefix=_run_string_prefix)  # number of free modes to use
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
    #### Plotting the results
    We will see some of the mirror modes (modes with negative $m$) because we summed all of the multipoles
    """)
    return


@app.cell
def _(Mf, af, mode_list, plot_omega_free, plot_predicted_qnms, plt, result):
    _fig, _ax = plt.subplots()
    predicted_qnms = mode_list(['2.2.0', '2.2.1', '3.2.0', '3.3.0', '2.-2.0', '3.-3.0', '4.-4.0', '4.4.0'], Mf, af)
    # mode locations to visualize on the plot
    plot_omega_free(result, _ax)
    plot_predicted_qnms(_ax, predicted_qnms)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fixed QNMs (fixed frequencies)
    Let us assume that the $2{,}2{,}0, 3{,}3{,}0, 4{,}4{,}0$ modes and their mirror counter-parts exist in the waveform
    """)
    return


@app.cell
def _(Mf, QNMFitVaryingStartingTime, SXSnum, af, h, iota, mode_list, np, psi):
    _t0_arr = np.linspace(0, 50, num=101)
    qnm_fixed_list_1 = mode_list(['2.2.0', '2.-2.0', '3.3.0', '3.-3.0', '4.4.0', '4.-4.0'], Mf, af)
    _run_string_prefix = f'SXS{SXSnum}_lm_2.2_iota_{iota:.7f}_psi_{psi:.7f}'
    _N_free = 0
    fitter_1 = QNMFitVaryingStartingTime(h, _t0_arr, N_free=_N_free, qnm_fixed_list=qnm_fixed_list_1, load_pickle=True, run_string_prefix=_run_string_prefix)
    return fitter_1, qnm_fixed_list_1


@app.cell
def _(fitter_1, mo):
    with mo.status.spinner("Running fixed-QNM fits..."):
        fitter_1.do_fits()
    result_1 = fitter_1.result_full
    mo.md("**Fixed-QNM fits complete.**")
    return (result_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plotting the results
    Because our waveform is non-precessing, by PT symmetry, the amplitudes of the mirror modes can be predicted from those of the prograde modes, i.e.
    $$
    \tilde{A}_{\ell{,}-m{,}n} = \dfrac{S_{\ell{,}-m{,}n}(\iota, \psi)}{S_{\ell{,}m{,}n}^*(\iota, \psi)}\tilde{A}^*_{\ell{,}m{,}n} ,
    $$
    Where $\tilde{A}$ denotes the complex amplitude, and $S_{\ell{,}m{,}n}$ are the spin weighted ($s = -2$ in this case) spheroidal harmonics.
    We compute the predicted mirror mode amplitudes and plot them as dashed lines with colors corresponding to their prograde mode counter-parts.
    """)
    return


@app.cell
def _(
    af,
    iota,
    plot_amplitudes,
    plot_phases,
    plt,
    psi,
    qnm_fixed_list_1,
    result_1,
):
    _fig, _axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_amplitudes(result_1, fixed_modes=qnm_fixed_list_1, ax=_axs[0], plot_mirror_pred=True, iota=iota, psi=psi, af=af)
    plot_phases(result_1, fixed_modes=qnm_fixed_list_1, ax=_axs[1], legend=False, plot_mirror_pred=True, iota=iota, psi=psi, af=af)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Including a pair of prograde and mirror modes as the same mode
    Given the relationship between the complex amplitudes of the prograde and mirror modes for non-precessing mergers, we can include both modes as one mode into our fit model.
    This can be done with `include_mirror = True`.
    """)
    return


@app.cell
def _(Mf, QNMFitVaryingStartingTime, SXSnum, af, h, iota, mode_list, np, psi):
    _t0_arr = np.linspace(0, 50, num=101)
    qnm_fixed_list_2 = mode_list(['2.2.0', '3.3.0', '4.4.0'], Mf, af)
    _run_string_prefix = f'SXS{SXSnum}_lm_2.2_iota_{iota:.7f}_psi_{psi:.7f}_incl_mirror'
    _N_free = 0
    fitter_2 = QNMFitVaryingStartingTime(h, _t0_arr, N_free=_N_free, qnm_fixed_list=qnm_fixed_list_2, load_pickle=False, run_string_prefix=_run_string_prefix, include_mirror=True, iota=iota, psi=psi)
    return fitter_2, qnm_fixed_list_2


@app.cell
def _(fitter_2, mo):
    with mo.status.spinner("Running mirror-mode fits..."):
        fitter_2.do_fits()
    result_2 = fitter_2.result_full
    mo.md("**Mirror-mode fits complete.**")
    return (result_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Plotting the results
    Only the amplitude of the prograde modes are shown, but the mirror modes have been included in the fit, with amplitudes fixed by PT symmetry as explained above.
    """)
    return


@app.cell
def _(plot_amplitudes, plot_phases, plt, qnm_fixed_list_2, result_2):
    _fig, _axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_amplitudes(result_2, fixed_modes=qnm_fixed_list_2, ax=_axs[0], plot_mirror_pred=False)
    plot_phases(result_2, fixed_modes=qnm_fixed_list_2, ax=_axs[1], legend=False, plot_mirror_pred=False)
    _fig
    return


if __name__ == "__main__":
    app.run()
