from jaxqualin.data import (download_hyperfit_data, 
                            download_interpolate_data,
                            make_hyper_fit_functions,
                            make_interpolators)
import numpy as np
import pytest

@pytest.fixture
def hyperfit_data():
    download_hyperfit_data()
    download_interpolate_data()
    hyperfit_functions = make_hyper_fit_functions()
    hyper_interpolators = make_interpolators()
    return hyperfit_functions, hyper_interpolators

def test_hyperfit(hyperfit_data):

    hyperfit_functions, hyper_interpolators = hyperfit_data

    mode_name = '2.2.1'

    eta, chi_p, chi_m = 0.2, 0.1, 0.4
    A_fit = hyperfit_functions[mode_name]['A'](eta, chi_p, chi_m)
    A_interp = hyper_interpolators[mode_name]['A'](eta, chi_p, chi_m)
    phi_fit = hyperfit_functions[mode_name]['phi'](eta, chi_p, chi_m)
    phi_interp = hyper_interpolators[mode_name]['phi'](eta, chi_p, chi_m)
    dA_interp = hyper_interpolators[mode_name]['dA'](eta, chi_p, chi_m)
    dphi_interp = hyper_interpolators[mode_name]['dphi'](eta, chi_p, chi_m)
    assert not np.isnan(A_interp)
    assert not np.isnan(phi_interp)
    assert not np.isnan(dA_interp)
    assert not np.isnan(dphi_interp)
    assert np.isclose(A_fit, A_interp, rtol = 0.5)
    assert np.isclose(phi_fit, phi_interp, rtol = 0.5)