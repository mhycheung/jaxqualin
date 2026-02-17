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

    This notebook demonstrates three new features in `jaxqualin`:

    1. **Custom fixed-omega modes** -- fit a waveform with user-specified complex frequencies.
    2. **Augmented Kerr model** -- extend the Kerr model with additional parameters.
    3. **Fully custom model** -- define a QNM model from scratch with arbitrary parameters.

    All synthetic waveforms use `delayed_QNM` for realistic merger distortion.
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
        QNMFit, QNMFitVarMa, QNMFitVaryingStartingTime,
    )
    from jaxqualin.plot import plot_amplitudes, plot_phases

    import numpy as np
    import matplotlib.pyplot as plt

    return (
        KerrModel,
        QNMFit,
        QNMFitVarMa,
        QNMFitVaryingStartingTime,
        QNMModel,
        custom_mode,
        custom_mode_list,
        delayed_QNM,
        model_mode,
        model_mode_free,
        mode_list,
        np,
        plot_amplitudes,
        plot_phases,
        plt,
        waveform,
    )


# ===================================================================
# Part 1: Custom fixed-omega modes
# ===================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: Custom fixed-omega modes

    Sometimes you already know the complex frequencies of the modes you want to fit, either from
    a different code, from a table, or from your own physical model. You can create `custom_mode`
    objects with arbitrary $\omega$ and optional labels, then use them directly in `QNMFit` or
    `QNMFitVaryingStartingTime`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build a synthetic waveform

    We create a waveform using Kerr QNM frequencies (which we will pretend to know already),
    distorted near the merger with `delayed_QNM`.
    """)
    return


@app.cell
def _(delayed_QNM, mode_list, np, waveform):
    Mf_1 = 1.0
    af_1 = 0.7
    kerr_modes_1 = mode_list(['2.2.0', '2.2.1'], Mf_1, af_1)

    A_phi_dict_1 = {
        '2.2.0': dict(A=1.0, phi=0.0),
        '2.2.1': dict(A=3.0, phi=np.pi / 2),
    }

    t_arr_1 = np.linspace(0, 120, 1000)
    _h_arr = np.zeros(t_arr_1.shape, dtype=np.complex128)
    for _i, _mode in enumerate(kerr_modes_1):
        if _i == 0:
            _A_delay, _A_sig, _phi_sig = 0, 10, 5
        else:
            _A_delay, _A_sig, _phi_sig = 5, 2, 2
        _h_arr = _h_arr + delayed_QNM(
            _mode, t_arr_1,
            A_phi_dict_1[_mode.string()]['A'],
            A_phi_dict_1[_mode.string()]['phi'],
            A_delay=_A_delay, A_sig=_A_sig, phi_sig=_phi_sig,
        )
    h_1 = waveform(t_arr_1, _h_arr, t_peak=0)
    return A_phi_dict_1, Mf_1, af_1, h_1, kerr_modes_1, t_arr_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Create custom modes with known frequencies

    We extract the frequencies from the Kerr modes and pass them to `custom_mode` objects.
    Labels can be standard QNM strings like `"2.2.0"`, custom names, or left as `None`
    (auto-labeled `mode_0`, `mode_1`, ...).
    """)
    return


@app.cell
def _(custom_mode, kerr_modes_1):
    custom_modes_1 = [
        custom_mode(
            complex(m.omegar) + 1j * complex(m.omegai),
            label=m.string(),
        )
        for m in kerr_modes_1
    ]
    return (custom_modes_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fit with custom fixed modes

    Since the frequencies are already fixed, we use `QNMFitVaryingStartingTime` with `N_free=0`
    and pass the custom modes as `qnm_fixed_list`.
    """)
    return


@app.cell
def _(QNMFitVaryingStartingTime, custom_modes_1, h_1, mo, np, plot_amplitudes, plot_phases, plt):
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
    plot_amplitudes(result_1, ax=_axs[0])
    plot_phases(result_1, ax=_axs[1], legend=False)
    _fig.suptitle('Part 1: Custom fixed-omega modes')
    _fig.tight_layout()
    _fig
    return result_1, t0_arr_1


@app.cell
def _(mo, np, result_1):
    _keys = list(result_1.A_dict.keys())
    _vals = [f"`{k}`: A = {np.array(v)[-1]:.4f}" for k, v in result_1.A_dict.items()]
    mo.md(f"""
    **Result dict keys (labels appear correctly):**

    {chr(10).join('- ' + v for v in _vals)}
    """)
    return


# ===================================================================
# Part 2: Augmented Kerr model
# ===================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2: Augmented Kerr model

    Suppose you want to test whether the data is consistent with Kerr, allowing for a small
    frequency deviation $\delta$. You can subclass `KerrModel` and add an extra parameter:

    $$\omega_{\rm total} = \omega_{\rm Kerr}(M, a) + \delta$$

    The `QNMFitVarMa` fitter will optimize $M$, $a$, **and** $\delta$ simultaneously.
    """)
    return


@app.cell
def _(KerrModel, QNMModel):
    class KerrPlusDelta(KerrModel):
        param_names = ["M", "a", "delta"]

        def compute_omega(self, lmnx, M, a, delta, **kwargs):
            omega_kerr = super().compute_omega(lmnx, M, a)
            return omega_kerr + delta

        def param_bounds(self):
            b = super().param_bounds()
            b["delta"] = (-1.0, 1.0)
            return b

    return (KerrPlusDelta,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build waveform with a frequency shift

    We create a waveform where each mode's frequency is shifted by $\delta = 0.02$ from the Kerr value.
    """)
    return


@app.cell
def _(KerrPlusDelta, custom_mode, delayed_QNM, np, waveform):
    Mf_2 = 1.0
    af_2 = 0.7
    delta_true = 0.02

    kpd_gen = KerrPlusDelta()
    lmnx_list_2 = [[[2, 2, 0]], [[2, 2, 1]]]
    shifted_omegas = [
        kpd_gen.compute_omega(lmx, M=Mf_2, a=af_2, delta=delta_true)
        for lmx in lmnx_list_2
    ]
    shifted_cm = [
        custom_mode(omega, label=f"{lmx[0][0]}.{lmx[0][1]}.{lmx[0][2]}")
        for omega, lmx in zip(shifted_omegas, lmnx_list_2)
    ]

    A_phi_2 = [(1.0, 0.0), (3.0, np.pi / 2)]

    t_arr_2 = np.linspace(0, 120, 1000)
    _h_arr = np.zeros(t_arr_2.shape, dtype=np.complex128)
    for _i, (_mode, (_A, _phi)) in enumerate(zip(shifted_cm, A_phi_2)):
        if _i == 0:
            _A_delay, _A_sig, _phi_sig = 0, 10, 5
        else:
            _A_delay, _A_sig, _phi_sig = 5, 2, 2
        _h_arr = _h_arr + delayed_QNM(
            _mode, t_arr_2, _A, _phi,
            A_delay=_A_delay, A_sig=_A_sig, phi_sig=_phi_sig,
        )

    h_2 = waveform(t_arr_2, _h_arr, t_peak=0)
    return Mf_2, af_2, delta_true, h_2, t_arr_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fit with augmented Kerr model

    We create `model_mode_free` objects using the `KerrPlusDelta` model and fit with
    `QNMFitVaryingStartingTime`, which internally uses `QNMFitVarMa` with the custom model.
    """)
    return


@app.cell
def _(
    KerrPlusDelta,
    Mf_2,
    QNMFitVaryingStartingTime,
    af_2,
    h_2,
    mo,
    model_mode_free,
    np,
    plt,
):
    kpd_model = KerrPlusDelta()
    kpd_modes = [
        model_mode_free([[2, 2, 0]], model=kpd_model),
        model_mode_free([[2, 2, 1]], model=kpd_model),
    ]

    t0_arr_2 = np.linspace(0, 50, num=51)

    fitter_2 = QNMFitVaryingStartingTime(
        h_2, t0_arr_2, N_free=0,
        qnm_fixed_list=[],
        qnm_free_list=kpd_modes,
        var_M_a=True,
        load_pickle=False,
        run_string_prefix='kerr_plus_delta_example',
        save_results=False,
        model=kpd_model,
        model_params_guess={"M": Mf_2 * 0.9, "a": af_2 * 0.9, "delta": 0.0},
    )
    with mo.status.spinner("Fitting with augmented Kerr model..."):
        fitter_2.do_fits()

    result_2 = fitter_2.result_full
    return fitter_2, kpd_model, kpd_modes, result_2, t0_arr_2


@app.cell
def _(Mf_2, af_2, delta_true, np, plt, result_2, t0_arr_2):
    _fig, _axs = plt.subplots(1, 3, figsize=(15, 4))

    _M_arr = np.array(result_2.Ma_dict['M'])
    _axs[0].plot(t0_arr_2, _M_arr, 'b-', lw=2)
    _axs[0].axhline(Mf_2, color='r', ls='--', label=f'True $M = {Mf_2}$')
    _axs[0].set_xlabel(r'$t_0$')
    _axs[0].set_ylabel(r'$M$')
    _axs[0].set_title('Recovered mass')
    _axs[0].legend()

    _a_arr = np.array(result_2.Ma_dict['a'])
    _axs[1].plot(t0_arr_2, _a_arr, 'b-', lw=2)
    _axs[1].axhline(af_2, color='r', ls='--', label=f'True $a = {af_2}$')
    _axs[1].set_xlabel(r'$t_0$')
    _axs[1].set_ylabel(r'$a$')
    _axs[1].set_title('Recovered spin')
    _axs[1].legend()

    _delta_arr = np.array(result_2.Ma_dict['delta'])
    _axs[2].plot(t0_arr_2, _delta_arr, 'b-', lw=2)
    _axs[2].axhline(delta_true, color='r', ls='--', label=rf'True $\delta = {delta_true}$')
    _axs[2].set_xlabel(r'$t_0$')
    _axs[2].set_ylabel(r'$\delta$')
    _axs[2].set_title(r'Recovered $\delta$')
    _axs[2].legend()

    _fig.suptitle('Part 2: Augmented Kerr model — recovered parameters')
    _fig.tight_layout()
    _fig
    return


# ===================================================================
# Part 3: Fully custom model
# ===================================================================

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3: Fully custom (non-Kerr) model

    You can also define a QNM model from scratch, where $\omega$ is a function of
    completely arbitrary parameters (not necessarily $M$ and $a$).

    As a toy example, consider a model where each mode has the frequency:

    $$\omega = \alpha + i\,\beta$$

    where $\alpha$ (real frequency) and $\beta$ (damping rate) are the free parameters
    to be fitted.
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
def _(mo):
    mo.md(r"""
    ### Build waveform and fit

    We create a single-mode waveform with known $\alpha$ and $\beta$, then fit using
    `QNMFitVarMa` with the `SimpleFreqModel`.
    """)
    return


@app.cell
def _(QNMFitVarMa, SimpleFreqModel, mo, model_mode_free, np, waveform):
    alpha_true = 0.5
    beta_true = -0.08
    _omega_3 = alpha_true + 1j * beta_true

    _A_true, _phi_true = 2.0, 0.3
    _t_arr = np.linspace(0, 100, 2000)
    _h_arr = _A_true * np.exp(-1j * (_omega_3 * _t_arr + _phi_true))
    h_3 = waveform(_t_arr, _h_arr, t_peak=0)

    sfm = SimpleFreqModel()
    sfm_modes = [model_mode_free([[2, 2, 0]], model=sfm)]

    fitter_3 = QNMFitVarMa(
        h_3, t0=0.0, qnm_free_list=sfm_modes,
        model=sfm,
        model_params_guess={"alpha": 0.4, "beta": -0.1},
    )
    fitter_3.do_fit()
    _popt_3 = np.array(fitter_3.popt)

    mo.md(f"""
    **Fully custom model fit results:**

    | Parameter | True | Recovered |
    |-----------|------|-----------|
    | $\\alpha$ | {alpha_true} | {_popt_3[2]:.6f} |
    | $\\beta$  | {beta_true} | {_popt_3[3]:.6f} |
    | Mismatch  | -- | {fitter_3.mismatch:.2e} |
    """)
    return alpha_true, beta_true, fitter_3, h_3, sfm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    **Summary**

    - `custom_mode` / `custom_mode_list`: for when you already know $\omega$ exactly.
    - `KerrModel` (and subclasses like `KerrPlusDelta`): augment the standard Kerr model with extra parameters.
    - `QNMModel` subclass: fully custom parametric models with any parameters.
    - All integrate seamlessly with `QNMFit`, `QNMFitVarMa`, and `QNMFitVaryingStartingTime`.
    """)
    return


if __name__ == "__main__":
    app.run()
