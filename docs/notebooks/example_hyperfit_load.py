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
    # Load Hyperfit functions and interpolator

    (Please update to the latest version of `jaxqualin` to avoid errors!)

    In this example we will call the hyperfit functions (listed in the `jaxqualin` paper) and corresponding interpolators to estimate the amplitudes and phases of different modes, as a function of the binary black hole simulation parameters.
    The hyperfit polynomial terms and interpolation data might be updated in future versions of the paper.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Import functions
    """)
    return


@app.cell
def _():
    from jaxqualin.data import (download_hyperfit_data, 
                                download_interpolate_data,
                                make_hyper_fit_functions,
                                make_interpolators)

    return (
        download_hyperfit_data,
        download_interpolate_data,
        make_hyper_fit_functions,
        make_interpolators,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Download data

    By default, the download functions compares the local version of the data with the one hosted on this webpage.
    If the online one is newer, it will be downloaded and the local version will be overriden.
    Use `overwrite = 'force'` to force overwrite, and `never` to avoid overwriting.
    """)
    return


@app.cell
def _(download_hyperfit_data, download_interpolate_data):
    download_hyperfit_data()
    download_interpolate_data()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Make hyperfit functions and interpolators

    Now we convert the downloaded data into functions and interpolators that we can easily call.
    """)
    return


@app.cell
def _(make_hyper_fit_functions, make_interpolators):
    hyperfit_functions = make_hyper_fit_functions()
    hyper_interpolators = make_interpolators()
    return hyper_interpolators, hyperfit_functions


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estimating the amplitude and phase

    We can estimate the amplitude and phase for a set of progenitor parameters $(\eta, \chi_+, \chi_-)$.
    All is well if the hyperfit and interpolation returns similar results.
    """)
    return


@app.cell
def _(hyper_interpolators, hyperfit_functions):
    mode_name = '2.2.1'
    eta, chi_p, chi_m = (0.2, 0.1, 0.4)
    _A_fit = hyperfit_functions[mode_name]['A'](eta, chi_p, chi_m)
    _A_interp = hyper_interpolators[mode_name]['A'](eta, chi_p, chi_m)
    _phi_fit = hyperfit_functions[mode_name]['phi'](eta, chi_p, chi_m)
    _phi_interp = hyper_interpolators[mode_name]['phi'](eta, chi_p, chi_m)
    print(f'A_fit: {_A_fit:.5f}, A_interp: {_A_interp:.5f}')
    print(f'phi_fit: {_phi_fit:.5f}, phi_interp: {_phi_interp:.5f}')
    return (mode_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The interpolator returns `nan` if the requested point is outside of the convex hull of the interpolation data.
    This can be used to check whether the hyperfit is extrapolating at the point, because the fit is trained with the same data used to construct the interpolator.
    If the point is not covered by the convex hull, it could be because no simulations in the SXS catalog cover that region of the parameter space, or the mode amplitude is too weak (such that the mode extraction procedure missed the mode).
    """)
    return


@app.cell
def _(hyper_interpolators, hyperfit_functions, mode_name):
    eta_1, chi_p_1, chi_m_1 = (0.1, 0.9, -0.9)
    _A_fit = hyperfit_functions[mode_name]['A'](eta_1, chi_p_1, chi_m_1)
    _A_interp = hyper_interpolators[mode_name]['A'](eta_1, chi_p_1, chi_m_1)
    _phi_fit = hyperfit_functions[mode_name]['phi'](eta_1, chi_p_1, chi_m_1)
    _phi_interp = hyper_interpolators[mode_name]['phi'](eta_1, chi_p_1, chi_m_1)
    print(f'A_fit: {_A_fit:.5f}, A_interp: {_A_interp:.5f}')
    print(f'phi_fit: {_phi_fit:.5f}, phi_interp: {_phi_interp:.5f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When the hyperfit and interpolation returns significantly different results, care should be taken.
    The mode amplitude might be too small for the hyperfit to work accurately, and the results in these regions of parameter space should at most be used as an order of magnitude estimation.
    In fact, the hyperfit amplitude could even be negative, if the mode amplitude is too low.
    """)
    return


@app.cell
def _(hyper_interpolators, hyperfit_functions):
    mode_name_1 = '-2.2.0'
    eta_2, chi_p_2, chi_m_2 = (0.2, -0.1, 0.4)
    _A_fit = hyperfit_functions[mode_name_1]['A'](eta_2, chi_p_2, chi_m_2)
    _A_interp = hyper_interpolators[mode_name_1]['A'](eta_2, chi_p_2, chi_m_2)
    _phi_fit = hyperfit_functions[mode_name_1]['phi'](eta_2, chi_p_2, chi_m_2)
    _phi_interp = hyper_interpolators[mode_name_1]['phi'](eta_2, chi_p_2, chi_m_2)
    print(f'A_fit: {_A_fit:.5e}, A_interp: {_A_interp:.5e}')
    print(f'phi_fit: {_phi_fit:.5f}, phi_interp: {_phi_interp:.5f}')
    return chi_m_2, chi_p_2, eta_2, mode_name_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can also interpolate the fluctuation of the mode amplitude and phase when they were extracted with our procedure
    """)
    return


@app.cell
def _(chi_m_2, chi_p_2, eta_2, hyper_interpolators, mode_name_1):
    dA_interp = hyper_interpolators[mode_name_1]['dA'](eta_2, chi_p_2, chi_m_2)
    dphi_interp = hyper_interpolators[mode_name_1]['dphi'](eta_2, chi_p_2, chi_m_2)
    print(f'dA_interp: {dA_interp:.5e}, dphi_interp: {dphi_interp:.5f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Changing parameterization

    By specifying `PN = False`, we can use the $(q, \chi_1, \chi_2)$ parameterization instead of $(\eta, \chi_+, \chi_-)$
    """)
    return


@app.cell
def _(make_hyper_fit_functions, make_interpolators):
    hyperfit_functions_1 = make_hyper_fit_functions(PN=False)
    hyper_interpolators_1 = make_interpolators(PN=False)
    return hyper_interpolators_1, hyperfit_functions_1


@app.cell
def _(hyper_interpolators_1, hyperfit_functions_1):
    mode_name_2 = '2.2.1'
    q, chi_1, chi_2 = (2.3, 0.2, -0.2)
    _A_fit = hyperfit_functions_1[mode_name_2]['A'](q, chi_1, chi_2)
    _A_interp = hyper_interpolators_1[mode_name_2]['A'](q, chi_1, chi_2)
    _phi_fit = hyperfit_functions_1[mode_name_2]['phi'](q, chi_1, chi_2)
    _phi_interp = hyper_interpolators_1[mode_name_2]['phi'](q, chi_1, chi_2)
    print(f'A_fit: {_A_fit:.5f}, A_interp: {_A_interp:.5f}')
    print(f'phi_fit: {_phi_fit:.5f}, phi_interp: {_phi_interp:.5f}')
    return


if __name__ == "__main__":
    app.run()
