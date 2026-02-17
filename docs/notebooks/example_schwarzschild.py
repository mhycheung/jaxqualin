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
    # Example: Fitting a Schwarzschild (non-spinning) waveform

    In this example, we construct a toy waveform from Schwarzschild quasi-normal modes (QNMs) and fit it using `jaxqualin`.

    A Schwarzschild black hole has zero spin ($a = 0$), so the gravitational waveform is purely real:

    $$h(t) = \sum_j A_j \, e^{\omega_{i,j}\, t} \cos(\omega_{r,j}\, t + \phi_j)$$

    We use `delayed_QNM` to add realistic distortions near the merger peak, then demonstrate three fitting approaches:

    1. **Free-frequency fit** across varying $t_0$ to identify modes
    2. **Fixed-frequency fit** across varying $t_0$ with `Schwarzschild=True`
    3. **Variable-mass fit** across varying $t_0$ with `Schwarzschild=True` to recover $M$
    """)
    return


@app.cell
def _():
    from jaxqualin.waveforms import delayed_QNM, waveform
    from jaxqualin.qnmode import mode_list, long_str_to_qnms_free
    from jaxqualin.fit import QNMFitVaryingStartingTime
    from jaxqualin.plot import (plot_amplitudes, plot_phases,
                                plot_omega_free, plot_predicted_qnms)

    import numpy as np
    import matplotlib.pyplot as plt

    return (
        QNMFitVaryingStartingTime,
        delayed_QNM,
        long_str_to_qnms_free,
        mode_list,
        np,
        plot_amplitudes,
        plot_omega_free,
        plot_phases,
        plot_predicted_qnms,
        plt,
        waveform,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build a toy Schwarzschild waveform

    We construct a waveform with two Schwarzschild QNMs ($a = 0$): the $(2,2,0)$ fundamental mode and the $(3,2,0)$ mode.
    The `delayed_QNM` function adds distortions near $t = 0$ to simulate realistic merger behavior.
    Since the black hole is non-spinning, we take the real part of the waveform.
    """)
    return


@app.cell
def _(delayed_QNM, mode_list, np, waveform):
    Mf_true = 1.0
    af_true = 0.0
    schw_modes = mode_list(['2.2.0', '3.2.0'], Mf_true, af_true)

    schw_A_phi_dict = {
        '2.2.0': dict(A=1.0, phi=0.0),
        '3.2.0': dict(A=0.5, phi=np.pi / 3),
    }

    schw_t_arr = np.linspace(0, 120, 1000)
    _h_arr = np.zeros(schw_t_arr.shape, dtype=np.complex128)
    for _i, _mode in enumerate(schw_modes):
        if _i == 0:
            _A_delay, _A_sig, _phi_sig = 0, 10, 5
        else:
            _A_delay, _A_sig, _phi_sig = 5, 2, 2
        _h_arr = _h_arr + delayed_QNM(
            _mode, schw_t_arr,
            schw_A_phi_dict[_mode.string()]['A'],
            schw_A_phi_dict[_mode.string()]['phi'],
            A_delay=_A_delay, A_sig=_A_sig, phi_sig=_phi_sig,
        )

    # Take the real part for a Schwarzschild waveform
    h_schw = waveform(schw_t_arr, np.real(_h_arr) + 0j * np.real(_h_arr), t_peak=0)
    return Mf_true, af_true, h_schw, schw_modes


@app.cell
def _(h_schw, np, plt):
    _fig, _ax = plt.subplots()
    _ax.semilogy(h_schw.time, np.abs(h_schw.hr))
    _ax.set_xlabel(r'$t$')
    _ax.set_ylabel(r'$|h_r|$')
    _ax.set_title('Schwarzschild toy waveform (real part)')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Free-frequency fit (varying $t_0$)

    First we fit with free QNMs ($N_f = 2$) whose frequencies are not fixed, to see if the resulting frequencies converge to the expected Schwarzschild QNMs.
    Different colored points trace out the frequency evolution of each free QNM from early to late $t_0$.
    """)
    return


@app.cell(hide_code=True)
def _(QNMFitVaryingStartingTime, h_schw, mo, np):
    schw_free_t0_arr = np.linspace(0, 50, num=101)

    fitter_schw_free = QNMFitVaryingStartingTime(
        h_schw, schw_free_t0_arr, N_free=2,
        qnm_fixed_list=[],
        Schwarzschild=True,
        load_pickle=False,
        run_string_prefix='schwarzschild_free_example',
        save_results=False,
    )
    with mo.status.spinner("Running free-frequency fits..."):
        fitter_schw_free.do_fits()
    result_schw_free = fitter_schw_free.result_full
    return (result_schw_free,)


@app.cell
def _(
    Mf_true,
    af_true,
    mode_list,
    plot_omega_free,
    plot_predicted_qnms,
    plt,
    result_schw_free,
):
    _fig, _ax = plt.subplots()
    _predicted_qnms = mode_list(['2.2.0', '3.2.0'], Mf_true, af_true)
    plot_omega_free(result_schw_free, _ax)
    plot_predicted_qnms(_ax, _predicted_qnms)
    _ax.set_title('Free-frequency fit: QNM frequency evolution')
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Fixed-frequency fit (varying $t_0$)

    Now we assume the $(2,2,0)$ and $(3,2,0)$ modes are present and fix their frequencies according to GR.
    We fit across a range of starting times $t_0$ and plot how the recovered amplitudes and phases behave.
    For a clean QNM signal, these should be approximately constant across $t_0$.
    """)
    return


@app.cell
def _(QNMFitVaryingStartingTime, h_schw, mo, np, schw_modes):
    schw_fixed_t0_arr = np.linspace(0, 50, num=101)

    fitter_schw_fixed_vst = QNMFitVaryingStartingTime(
        h_schw, schw_fixed_t0_arr, N_free=0,
        qnm_fixed_list=schw_modes,
        Schwarzschild=True,
        load_pickle=False,
        run_string_prefix='schwarzschild_fixed_example',
        save_results=False,
    )
    with mo.status.spinner("Running fixed-frequency fits..."):
        fitter_schw_fixed_vst.do_fits()
    result_schw_fixed_vst = fitter_schw_fixed_vst.result_full
    return (result_schw_fixed_vst,)


@app.cell
def _(plot_amplitudes, plot_phases, plt, result_schw_fixed_vst, schw_modes):
    _fig, _axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_amplitudes(result_schw_fixed_vst, fixed_modes=schw_modes, ax=_axs[0])
    plot_phases(result_schw_fixed_vst, fixed_modes=schw_modes, ax=_axs[1], legend=False)
    _fig.suptitle('Schwarzschild fixed-frequency fit across $t_0$')
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Variable-mass fit (varying $t_0$)

    We use the VarMa fitter across starting times with `Schwarzschild=True` to see how the recovered mass $M$ evolves with $t_0$.
    At later starting times (further from the merger), the distortions are smaller and $M$ should converge to the true value.
    """)
    return


@app.cell
def _(QNMFitVaryingStartingTime, h_schw, long_str_to_qnms_free, mo, np):
    schw_varma_t0_arr = np.linspace(0, 50, num=51)
    schw_varma_qnm_free = long_str_to_qnms_free('2.2.0_3.2.0')

    fitter_schw_varma_vst = QNMFitVaryingStartingTime(
        h_schw, schw_varma_t0_arr, N_free=0,
        qnm_fixed_list=[],
        qnm_free_list=schw_varma_qnm_free,
        var_M_a=True,
        Schwarzschild=True,
        load_pickle=False,
        run_string_prefix='schwarzschild_varma_example',
        save_results=False,
    )
    with mo.status.spinner("Running VarMa varying $t_0$ fits..."):
        fitter_schw_varma_vst.do_fits()
    result_schw_varma_vst = fitter_schw_varma_vst.result_full
    return result_schw_varma_vst, schw_varma_t0_arr


@app.cell
def _(Mf_true, np, plt, result_schw_varma_vst, schw_varma_t0_arr):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _M_arr = np.array(result_schw_varma_vst.Ma_dict['M'])
    _ax.plot(schw_varma_t0_arr, _M_arr, 'b-', lw=2, label='Recovered $M$')
    _ax.axhline(Mf_true, color='r', ls='--', label=f'True $M = {Mf_true}$')
    _ax.set_xlabel(r'$t_0$')
    _ax.set_ylabel(r'$M$')
    _ax.set_title('Recovered mass across starting times (Schwarzschild)')
    _ax.legend()
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
