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
    # Example: Custom QNM modes and parametric models

    This notebook demonstrates two features in `jaxqualin`:

    1. **Custom fixed-omega modes** -- fit a waveform with user-specified complex frequencies.
    2. **Fully custom model** -- define a QNM model from scratch with arbitrary parameters.
    """)
    return


@app.cell
def _():
    from jaxqualin.waveforms import delayed_QNM, waveform
    from jaxqualin.qnmode import (
        mode_list, custom_mode, custom_mode_list,
        KerrModel, QNMModel, model_mode_free, model_mode,
    )
    from jaxqualin.fit import (
        QNMFit, QNMFitModel, QNMFitVaryingStartingTime,
    )
    from jaxqualin.plot import plot_amplitudes, plot_phases

    import numpy as np
    import matplotlib.pyplot as plt

    return (
        QNMFitVaryingStartingTime,
        QNMModel,
        custom_mode,
        delayed_QNM,
        model_mode_free,
        np,
        plot_amplitudes,
        plot_phases,
        plt,
        waveform,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: Custom fixed-frequency QNMs

    If you want to use non-Kerr frequencies, you can create a `custom_mode`
    objects with arbitrary $\omega$ and optional labels, then use them directly in `QNMFit` or
    `QNMFitVaryingStartingTime`.

    Here we use *truly arbitrary* frequencies that are **not** derived from Kerr:
    - `"fundamental"`:  $\omega = 0.50 - 0.08i$
    - `"overtone"`:  $\omega = 0.30 - 0.12i$
    """)
    return


@app.cell
def _(custom_mode, delayed_QNM, np, waveform):
    omega_1a = 0.50 - 0.08j
    omega_1b = 0.30 - 0.12j

    custom_modes_1 = [
        custom_mode(omega_1a, label="fundamental"),
        custom_mode(omega_1b, label="overtone"),
    ]

    A_phi_1 = [(1.0, 0.0), (3.0, np.pi / 2)]

    t_arr_1 = np.linspace(0, 120, 1000)
    _h_arr = np.zeros(t_arr_1.shape, dtype=np.complex128)
    for _i, (_mode, (_A, _phi)) in enumerate(zip(custom_modes_1, A_phi_1)):
        if _i == 0:
            _A_delay, _A_sig, _phi_sig = 0, 10, 5
        else:
            _A_delay, _A_sig, _phi_sig = 5, 2, 2
        _h_arr = _h_arr + delayed_QNM(
            _mode, t_arr_1, _A, _phi,
            A_delay=_A_delay, A_sig=_A_sig, phi_sig=_phi_sig,
        )
    h_1 = waveform(t_arr_1, _h_arr, t_peak=0)
    return custom_modes_1, h_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fit with custom fixed modes

    Since the frequencies are already fixed, we use `QNMFitVaryingStartingTime` with `N_free=0`
    and pass the custom modes as `qnm_fixed_list`.
    """)
    return


@app.cell
def _(
    QNMFitVaryingStartingTime,
    custom_modes_1,
    h_1,
    mo,
    np,
    plot_amplitudes,
    plot_phases,
    plt,
):
    t0_arr_1 = np.linspace(0, 50, num=51)

    fitter_1 = QNMFitVaryingStartingTime(
        h_1, t0_arr_1, N_free=0,
        qnm_fixed_list=custom_modes_1,
        load_pickle=False,
        run_string_prefix='custom_fixed_example',
        save_results=False,
    )
    with mo.status.spinner("Fitting with custom fixed modes..."):
        fitter_1.do_fits()

    result_1 = fitter_1.result_full

    _fig, _axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_amplitudes(result_1, fixed_modes=custom_modes_1, ax=_axs[0])
    plot_phases(result_1, fixed_modes=custom_modes_1, ax=_axs[1], legend=False)
    _fig.suptitle('Part 1: Custom fixed-omega modes')
    _fig.tight_layout()
    _fig
    return (result_1,)


@app.cell
def _(mo, np, result_1):
    _keys = list(result_1.A_dict.keys())
    _vals = [f"`{k}`: A = {np.array(v)[-1]:.4f}" for k, v in result_1.A_dict.items()]
    mo.md(f"""
    **Result dict keys (labels appear correctly):**

    {chr(10).join('- ' + v for v in _vals)}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2: Fitting a custom model

    You can also define a QNM model from scratch, where $\omega$ is a function of
    completely arbitrary parameters (not necessarily $M$ and $a$).

    As a toy example, consider a model where each mode has the frequency:

    $$\omega = \alpha + i\,\beta$$

    where $\alpha$ (real frequency) and $\beta$ (damping rate) are the free parameters
    to be fitted. We use `delayed_QNM` to build a synthetic waveform and
    `QNMFitVaryingStartingTime` to fit at multiple starting times, then plot the
    recovered parameters as a function of $t_0$.
    """)
    return


@app.cell
def _(QNMModel):
    class SimpleFreqModel(QNMModel):
        param_names = ["alpha", "beta"]

        def compute_omega(self, lmnx, alpha, beta, **kwargs):
            return alpha + 1j * beta

        def param_bounds(self):
            return {"alpha": (0, 10), "beta": (-5, 0)}

    return (SimpleFreqModel,)


@app.cell(hide_code=True)
def _(SimpleFreqModel, custom_mode, delayed_QNM, np, waveform):
    alpha_true = 0.5
    beta_true = -0.08

    sfm_cm = custom_mode(alpha_true + 1j * beta_true, label="toy_mode")

    _A_true, _phi_true = 2.0, 0.3
    t_arr_3 = np.linspace(0, 120, 1000)
    _h_arr = delayed_QNM(
        sfm_cm, t_arr_3, _A_true, _phi_true,
        A_delay=3, A_sig=5, phi_sig=3,
    )
    h_3 = waveform(t_arr_3, _h_arr, t_peak=0)

    sfm = SimpleFreqModel()
    return alpha_true, beta_true, h_3, sfm


@app.cell
def _(QNMFitVaryingStartingTime, h_3, mo, model_mode_free, np, sfm):
    sfm_modes = [model_mode_free([[2, 2, 0]], model=sfm)]

    t0_arr_3 = np.linspace(0, 50, num=51)

    fitter_3 = QNMFitVaryingStartingTime(
        h_3, t0_arr_3, N_free=0,
        qnm_fixed_list=[],
        qnm_free_list=sfm_modes,
        var_M_a=True,
        load_pickle=False,
        run_string_prefix='custom_simple_freq',
        save_results=False,
        model=sfm,
        model_params_guess={"alpha": 0.4, "beta": -0.1},
    )
    with mo.status.spinner("Fitting with fully custom model..."):
        fitter_3.do_fits()

    result_3 = fitter_3.result_full
    return result_3, t0_arr_3


@app.cell
def _(alpha_true, beta_true, np, plt, result_3, t0_arr_3):
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 4))

    _alpha_arr = np.array(result_3.model_params_dict['alpha'])
    _axs[0].plot(t0_arr_3, _alpha_arr, 'b-', lw=2)
    _axs[0].axhline(alpha_true, color='r', ls='--', label=rf'True $\alpha = {alpha_true}$')
    _axs[0].set_xlabel(r'$t_0$')
    _axs[0].set_ylabel(r'$\alpha$')
    _axs[0].set_title(r'Recovered $\alpha$')
    _axs[0].legend()

    _beta_arr = np.array(result_3.model_params_dict['beta'])
    _axs[1].plot(t0_arr_3, _beta_arr, 'b-', lw=2)
    _axs[1].axhline(beta_true, color='r', ls='--', label=rf'True $\beta = {beta_true}$')
    _axs[1].set_xlabel(r'$t_0$')
    _axs[1].set_ylabel(r'$\beta$')
    _axs[1].set_title(r'Recovered $\beta$')
    _axs[1].legend()

    _mismatch = np.array(result_3.mismatch_arr)
    _axs[2].semilogy(t0_arr_3, _mismatch, 'k-', lw=2)
    _axs[2].set_xlabel(r'$t_0$')
    _axs[2].set_ylabel('Mismatch')
    _axs[2].set_title('Fit mismatch')

    _fig.suptitle('Part 3: Fully custom model — recovered parameters vs $t_0$')
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
