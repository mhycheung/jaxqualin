"""
jaxqualin — JAX-accelerated quasinormal mode analysis for gravitational-wave ringdowns.
"""

__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# QNM mode representations
# ---------------------------------------------------------------------------
from .qnmode import (
    mode_free,
    mode,
    str_to_mode,
    mode_list,
    qnms_to_string,
    potential_modes,
    make_mirror_ratio_list,
    custom_mode,
    custom_mode_list,
    QNMModel,
    KerrModel,
    model_mode_free,
    model_mode,
)

# ---------------------------------------------------------------------------
# Fitting infrastructure
# ---------------------------------------------------------------------------
from .fit import (
    FitConfig,
    InitialGuessConfig,
    QNMFitResult,
    QNMFit,
    QNMFitVarMa,
    QNMFitVaryingStartingTime,
    QNMFitVaryingStartingTimeResult,
    FIT_SAVE_PATH,
)

# ---------------------------------------------------------------------------
# Waveform handling
# ---------------------------------------------------------------------------
from .waveforms import (
    waveform,
    get_waveform_SXS,
    get_M_a_SXS,
    get_SXS_waveform_dict,
    get_relevant_lm_waveforms_SXS,
)

# ---------------------------------------------------------------------------
# Automated mode selection / searching
# ---------------------------------------------------------------------------
from .selection import (
    ModeSearchAllFreeVaryingN,
    ModeSearchAllFreeVaryingNSXS,
    ModeSearchAllFreeVaryingNSXSAllRelevant,
)

# ---------------------------------------------------------------------------
# Data download and interpolation utilities
# ---------------------------------------------------------------------------
from .data import (
    download_hyperfit_data,
    download_interpolate_data,
    make_hyper_fit_functions,
    make_interpolators,
)

# ---------------------------------------------------------------------------
# Plotting is intentionally NOT re-exported here; use:
#   from jaxqualin.plot import plot_amplitudes, plot_phases, ...
# ---------------------------------------------------------------------------

__all__ = [
    # version
    "__version__",
    # qnmode
    "mode_free",
    "mode",
    "str_to_mode",
    "mode_list",
    "qnms_to_string",
    "potential_modes",
    "make_mirror_ratio_list",
    "custom_mode",
    "custom_mode_list",
    "QNMModel",
    "KerrModel",
    "model_mode_free",
    "model_mode",
    # fit
    "FitConfig",
    "InitialGuessConfig",
    "QNMFitResult",
    "QNMFit",
    "QNMFitVarMa",
    "QNMFitVaryingStartingTime",
    "QNMFitVaryingStartingTimeResult",
    "FIT_SAVE_PATH",
    # waveforms
    "waveform",
    "get_waveform_SXS",
    "get_M_a_SXS",
    "get_SXS_waveform_dict",
    "get_relevant_lm_waveforms_SXS",
    # selection
    "ModeSearchAllFreeVaryingN",
    "ModeSearchAllFreeVaryingNSXS",
    "ModeSearchAllFreeVaryingNSXSAllRelevant",
    # data
    "download_hyperfit_data",
    "download_interpolate_data",
    "make_hyper_fit_functions",
    "make_interpolators",
]
