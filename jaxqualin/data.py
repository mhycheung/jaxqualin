from urllib.request import urlretrieve, urlopen
import time
import os
import json

from scipy.interpolate import LinearNDInterpolator
import numpy as np

from typing import List, Tuple, Union, Optional, Dict, Any, Callable

_full_data_url = 'https://mhycheung.github.io/jaxqualin/data/SXS_results_latest.csv'
_hyperfit_url = 'https://mhycheung.github.io/jaxqualin/data/hyperfit_functions_latest.json'
_interpolate_url = 'https://mhycheung.github.io/jaxqualin/data/interpolate_data_latest.json'

DATA_SAVE_PATH = os.path.join(os.getcwd(), ".jaxqualin_cache/data")
_default_full_mode_data_path = os.path.join(
    DATA_SAVE_PATH, 'SXS_results_latest.csv')
_default_hyperfit_data_path = os.path.join(
    DATA_SAVE_PATH, 'hyperfit_functions_latest.json')
_default_interpolate_data_path = os.path.join(
    DATA_SAVE_PATH, 'interpolate_data_latest.json')


def last_modified_time(url: str) -> time.struct_time:
    """
    Returns the last modified time of a url
    
    Parameters:
        url: The url to check the last modified time of

    Returns:
        The last modified time of the url as a time.struct_time object
    
    """
    response = urlopen(url)
    last_modified = response.headers['Last-Modified']
    return time.strptime(last_modified, "%a, %d %b %Y %H:%M:%S %Z")


def download_file(filepath: str, url: str, overwrite: str='update') -> None:
    """
    Downloads a file from a url to a filepath

    Parameters:
        filepath: The filepath to save the downloaded file to
        url: The url to download the file from
        overwrite: Whether to overwrite the file if it already exists. Can
            be one of 'force', 'update', or 'never'.
    """

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    download = False
    if overwrite == 'force' or not os.path.exists(filepath):
        download = True
    elif overwrite == 'update':
        if last_modified_time(url) > time.gmtime(os.path.getmtime(filepath)):
            print('{} is more recent than {}, updating'.format(url, filepath))
            download = True
        else:
            print(
                '{} is not more recent than {}, file will not be downloaded.'.format(
                    url, filepath))
    elif overwrite == 'never':
        pass
    else:
        raise ValueError(
            "overwrite must be one of 'force', 'update', or 'never'")

    if download:
        print("Downloading data file from {} to {}".format(url, filepath))
        urlretrieve(url, filepath)


def download_full_mode_data(
        filepath=_default_full_mode_data_path,
        url=_full_data_url,
        overwrite='update'):
    download_file(filepath, url, overwrite=overwrite)


def download_hyperfit_data(
        filepath=_default_hyperfit_data_path,
        url=_hyperfit_url,
        overwrite='update'):
    download_file(filepath, url, overwrite=overwrite)


def download_interpolate_data(
        filepath=_default_interpolate_data_path,
        url=_interpolate_url,
        overwrite='update'):
    download_file(filepath, url, overwrite=overwrite)


def hyperfit_list_to_func(hyperfit_list, m, var='A', PN=True):
    if PN:
        def hyperfit_func(eta, chi_p, chi_m):
            val = 0
            for term_list in hyperfit_list:
                val += term_list[3] * eta**term_list[0] * \
                    chi_p**term_list[1] * chi_m**term_list[2]
            if var == 'A':
                if m % 2 == 0:
                    adj = eta
                else:
                    delta = np.sqrt(1 - 4 * eta)
                    adj = eta * delta
            elif var == 'phi':
                adj = 1
            else:
                raise ValueError("var must be 'A' or 'phi'")
            return adj * val
    else:
        def hyperfit_func(q, chi_1, chi_2):
            eta = q / (1 + q)**2
            chi_p = (q * chi_1 + chi_2) / (1 + q)
            chi_m = (q * chi_1 - chi_2) / (1 + q)
            val = 0
            for term_list in hyperfit_list:
                val += term_list[3] * eta**term_list[0] * \
                    chi_p**term_list[1] * chi_m**term_list[2]
            if var == 'A':
                if m % 2 == 0:
                    adj = eta
                else:
                    delta = np.sqrt(1 - 4 * eta)
                    adj = eta * delta
            elif var == 'phi':
                adj = 1
            else:
                raise ValueError("var must be 'A' or 'phi'")
            return adj * val
    return hyperfit_func


def make_hyper_fit_functions(filepath: str=_default_hyperfit_data_path, PN: bool=True) -> Dict[str, Dict[str, Callable[[float, float, float], float]]]:
    """
    Make a dictionary of hyperfit functions from a hyperfit data file.
    
    Parameters:
        filepath: The filepath of the hyperfit data file.
        PN: Whether to use Post Newtonian variables (`eta`, `chi_p`,
            `chi_m`) or the natural parameterization of the BBH simulations
            (`q`, `chi_1`, `chi_2`).

    Returns:
        A dictionary of hyperfit functions. The keys are the mode names, and
            the values are dictionaries with keys `A` and `phi` for the
            amplitude and phase hyperfit functions, respectively.
    """
    with open(filepath, 'r') as f:
        hyperfit_data = json.load(f)
    hyperfit_func_dict = {}
    for mode in hyperfit_data['A']:
        hyperfit_func_dict[mode] = {}
    for var in hyperfit_data:
        for mode in hyperfit_data[var]:
            l, m, n = map(int, mode.split('.'))
            hyperfit_list = hyperfit_data[var][mode]
            hyperfit_func_dict[mode][var] = hyperfit_list_to_func(
                hyperfit_list, m, var=var, PN=PN)
    return hyperfit_func_dict


def make_interpolators(filepath: str=_default_interpolate_data_path, PN: bool=True) -> Dict[str, Dict[str, LinearNDInterpolator]]:
    """
    Make a dictionary of interpolators from a data file containing the extracted
    amplitude and phases of different modes from BBH simulations.

    Parameters:
        filepath: The filepath of the data file.
        PN: Whether to use Post Newtonian variables (`eta`, `chi_p`,
            `chi_m`) or the natural parameterization of the BBH simulations
            (`q`, `chi_1`, `chi_2`).

    Returns:
        A dictionary of interpolators. The keys are the mode names, and the
            values are dictionaries with keys `A`, `dA`, `phi`, and `dphi` for
            the amplitude, amplitude fluctuation, phase, and phase fluctuation
            interpolators, respectively.
    """
    with open(filepath, 'r') as f:
        interpolate_data = json.load(f)
    interpolate_func_dict = {}
    for mode in interpolate_data:
        interpolate_func_dict[mode] = make_hyper_interpolator(
            interpolate_data[mode], PN=PN)
    return interpolate_func_dict


def make_hyper_interpolator(data_dict, PN=True):

    A = data_dict['A']
    dA = data_dict['dA']
    phi = data_dict['phi']
    dphi = data_dict['dphi']

    if PN:
        eta = data_dict['eta']
        chi_p = data_dict['chi_p']
        chi_m = data_dict['chi_m']
        points_eta_chi_p_chi_m = np.array([eta, chi_p, chi_m]).T
        interpolator_A = LinearNDInterpolator(points_eta_chi_p_chi_m, A)
        interpolator_dA = LinearNDInterpolator(points_eta_chi_p_chi_m, dA)
        interpolator_phi = LinearNDInterpolator(points_eta_chi_p_chi_m, phi)
        interpolator_dphi = LinearNDInterpolator(points_eta_chi_p_chi_m, dphi)
    else:

        q = data_dict['q']
        chi_1 = data_dict['chi_1']
        chi_2 = data_dict['chi_2']
        points_q_chi_1_chi_2 = np.array([q, chi_1, chi_2]).T
        interpolator_A = LinearNDInterpolator(points_q_chi_1_chi_2, A)
        interpolator_dA = LinearNDInterpolator(points_q_chi_1_chi_2, dA)
        interpolator_phi = LinearNDInterpolator(points_q_chi_1_chi_2, phi)
        interpolator_dphi = LinearNDInterpolator(points_q_chi_1_chi_2, dphi)

    return {'A': interpolator_A,
            'dA': interpolator_dA,
            'phi': interpolator_phi,
            'dphi': interpolator_dphi}
