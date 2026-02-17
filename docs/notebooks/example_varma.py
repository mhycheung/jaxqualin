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
    # Example: Variable mass and spin (VarMa) fitting

    In this example, we construct a toy waveform from Kerr quasi-normal modes (QNMs) and fit it using the VarMa fitter in `jaxqualin`, which simultaneously recovers the black hole mass $M$ and spin $a$ along with mode amplitudes and phases.

    A Kerr black hole has non-zero spin ($a \neq 0$), so the gravitational waveform is complex:

    $$h(t) = \sum_j A_j \, e^{(\omega_{i,j} - i\omega_{r,j})\, t - i\phi_j}$$

    We use `delayed_QNM` to add realistic distortions near the merger peak, then demonstrate:

    1. **Single starting time VarMa fit** with `QNMFitVarMa` to recover $M$ and $a$
    2. **Varying starting time VarMa fit** with `QNMFitVaryingStartingTime` showing $M(t_0)$ and $a(t_0)$ evolution
    """)
    return


@app.cell
def _():
    from jaxqualin.waveforms import delayed_QNM, waveform
    from jaxqualin.qnmode import mode_list, long_str_to_qnms_free
    from jaxqualin.fit import QNMFitVarMa, QNMFitVaryingStartingTime
    from jaxqualin.plot import plot_amplitudes, plot_phases

    import numpy as np
    import matplotlib.pyplot as plt

    return (
        QNMFitVarMa,
        QNMFitVaryingStartingTime,
        delayed_QNM,
        long_str_to_qnms_free,
        mode_list,
        np,
        plot_amplitudes,
        plot_phases,
        plt,
        waveform,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build a toy Kerr waveform

    We construct a waveform with three Kerr QNMs ($a = 0.7$): the $(2,2,0)$ fundamental, the $(2,2,1)$ first overtone, and the $(3,2,0)$ mode.
    The `delayed_QNM` function adds distortions near $t = 0$ to simulate realistic merger behavior.
    """)
    return


@app.cell
def _(delayed_QNM, mode_list, np, waveform):
    Mf_true = 1.0
    af_true = 0.7
    kerr_modes = mode_list(['2.2.0', '2.2.1', '3.2.0'], Mf_true, af_true)

    kerr_A_phi_dict = {
        '2.2.0': dict(A=1.0, phi=0.0),
        '2.2.1': dict(A=3.0, phi=np.pi / 2),
        '3.2.0': dict(A=1e-2, phi=np.pi),
    }

    kerr_t_arr = np.linspace(0, 120, 1000)
    _h_arr = np.zeros(kerr_t_arr.shape, dtype=np.complex128)
    for _i, _mode in enumerate(kerr_modes):
        if _i == 0:
            _A_delay, _A_sig, _phi_sig = 0, 10, 5
        else:
            _A_delay, _A_sig, _phi_sig = 5, 2, 2
        _h_arr = _h_arr + delayed_QNM(
            _mode, kerr_t_arr,
            kerr_A_phi_dict[_mode.string()]['A'],
            kerr_A_phi_dict[_mode.string()]['phi'],
            A_delay=_A_delay, A_sig=_A_sig, phi_sig=_phi_sig,
        )

    h_kerr = waveform(kerr_t_arr, _h_arr, t_peak=0)
    return Mf_true, af_true, h_kerr, kerr_modes, kerr_t_arr


@app.cell
def _(h_kerr, np, plt):
    _fig, _axs = plt.subplots(1, 2, figsize=(12, 4))
    _axs[0].semilogy(h_kerr.time, np.abs(h_kerr.hr))
    _axs[0].set_xlabel(r'$t$')
    _axs[0].set_ylabel(r'$|h_r|$')
    _axs[0].set_title('Real part')
    _axs[1].semilogy(h_kerr.time, np.abs(h_kerr.hi))
    _axs[1].set_xlabel(r'$t$')
    _axs[1].set_ylabel(r'$|h_i|$')
    _axs[1].set_title('Imaginary part')
    _fig.suptitle('Kerr toy waveform')
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Case 1: Single starting time VarMa fit

    We use `QNMFitVarMa` with `Schwarzschild=False` (the default) to fit the complex waveform at a single starting time $t_0 = 5$.
    The fitter recovers the mode amplitudes, phases, the black hole mass $M$, and spin $a$.

    The parameter vector `popt` has the structure: $[A_0, \phi_0, A_1, \phi_1, \ldots, M, a]$.
    """)
    return


@app.cell
def _(QNMFitVarMa, Mf_true, af_true, h_kerr, long_str_to_qnms_free, mo, np):
    kerr_qnm_free = long_str_to_qnms_free('2.2.0_2.2.1_3.2.0')

    fitter_kerr_varma = QNMFitVarMa(
        h_kerr, t0=5.0, qnm_free_list=kerr_qnm_free,
        Schwarzschild=False,
        guess_free=[1.0, 0.5],
        guess_M_a=[Mf_true * 0.9, af_true * 0.9],
    )
    fitter_kerr_varma.do_fit()

    _popt_kerr_varma = np.array(fitter_kerr_varma.popt)
    mo.md(f"""
    **VarMa fit results (single $t_0 = 5$):**

    - Mismatch: `{fitter_kerr_varma.mismatch:.2e}`
    - Recovered $M$: `{_popt_kerr_varma[-2]:.6f}` (true: `{Mf_true}`)
    - Recovered $a$: `{_popt_kerr_varma[-1]:.6f}` (true: `{af_true}`)
    """)
    return (fitter_kerr_varma,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Case 2: Varying starting time VarMa fit

    We now fit across a range of starting times $t_0$ to see how the recovered mass $M$ and spin $a$ evolve.
    At later starting times (further from the merger), the distortions are smaller and the recovered parameters should converge to the true values.
    """)
    return


@app.cell
def _(
    QNMFitVaryingStartingTime,
    h_kerr,
    long_str_to_qnms_free,
    mo,
    np,
):
    kerr_varma_t0_arr = np.linspace(0, 50, num=51)
    kerr_varma_qnm_free = long_str_to_qnms_free('2.2.0_2.2.1_3.2.0')

    fitter_kerr_varma_vst = QNMFitVaryingStartingTime(
        h_kerr, kerr_varma_t0_arr, N_free=0,
        qnm_fixed_list=[],
        qnm_free_list=kerr_varma_qnm_free,
        var_M_a=True,
        Schwarzschild=False,
        load_pickle=False,
        run_string_prefix='kerr_varma_example',
        save_results=False,
    )
    with mo.status.spinner("Running VarMa varying $t_0$ fits..."):
        fitter_kerr_varma_vst.do_fits()
    result_kerr_varma_vst = fitter_kerr_varma_vst.result_full
    return kerr_varma_t0_arr, result_kerr_varma_vst


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Recovered mass and spin across starting times

    The dashed red lines indicate the true values $M = 1.0$ and $a = 0.7$.
    At early starting times, the distortions from `delayed_QNM` cause deviations from the true values.
    At later starting times, the fit converges as the waveform approaches a pure QNM sum.
    """)
    return


@app.cell
def _(
    Mf_true,
    af_true,
    kerr_varma_t0_arr,
    np,
    plt,
    result_kerr_varma_vst,
):
    _fig, _axs = plt.subplots(1, 2, figsize=(12, 4))

    _M_arr = np.array(result_kerr_varma_vst.Ma_dict['M'])
    _axs[0].plot(kerr_varma_t0_arr, _M_arr, 'b-', lw=2, label='Recovered $M$')
    _axs[0].axhline(Mf_true, color='r', ls='--', label=f'True $M = {Mf_true}$')
    _axs[0].set_xlabel(r'$t_0$')
    _axs[0].set_ylabel(r'$M$')
    _axs[0].set_title('Recovered mass')
    _axs[0].legend()

    _a_arr = np.array(result_kerr_varma_vst.Ma_dict['a'])
    _axs[1].plot(kerr_varma_t0_arr, _a_arr, 'b-', lw=2, label='Recovered $a$')
    _axs[1].axhline(af_true, color='r', ls='--', label=f'True $a = {af_true}$')
    _axs[1].set_xlabel(r'$t_0$')
    _axs[1].set_ylabel(r'$a$')
    _axs[1].set_title('Recovered spin')
    _axs[1].legend()

    _fig.suptitle('VarMa fit: recovered $M$ and $a$ across starting times')
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Recovered amplitudes and phases

    We can also look at how the mode amplitudes and phases evolve across starting times.
    """)
    return


@app.cell
def _(plot_amplitudes, plot_phases, plt, result_kerr_varma_vst):
    _fig, _axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_amplitudes(result_kerr_varma_vst, ax=_axs[0])
    plot_phases(result_kerr_varma_vst, ax=_axs[1], legend=False)
    _fig.suptitle('VarMa fit: amplitudes and phases across $t_0$')
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
