_ext_coeffs = ["deoxyhemoglobin", "get_extinction_coeffs", "oxyhemoglobin", "soy_oil", "water"]
_nlayer = [
    "model_nlayer_fd",
    "model_nlayer_fd_fft",
    "model_nlayer_g1",
    "model_nlayer_g1_fft",
    "model_nlayer_g2",
    "model_nlayer_ss",
    "model_nlayer_ss_fft",
]
_model = ["model_fd", "model_g1", "model_g2", "model_ss", "model_td"]
_plotting = ["joint_hist", "soft_dark_style"]
_stat = ["CentralMoments", "StandardMoments", "WeightedCentralMoments", "weighted_quantile"]
_util = ["fma", "gen_coeffs", "gen_impedance", "integrate", "qrng"]

__all__ = _ext_coeffs + _nlayer + _model + _plotting + _stat + _util


def __getattr__(name):
    from warnings import warn

    if name in _ext_coeffs:
        import libnirs.extinction_coeffs

        submod = libnirs.extinction_coeffs
    elif name in _nlayer:
        import libnirs.layered_model

        submod = libnirs.layered_model
    elif name in _model:
        import libnirs.model

        submod = libnirs.model
    elif name in _plotting:
        import libnirs.plotting

        submod = libnirs.plotting
    elif name in _stat:
        import libnirs.statistical

        submod = libnirs.statistical
    elif name in _util:
        import libnirs.utils

        submod = libnirs.utils
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    warn(
        f"Importing {name} directly from libnirs.__init__ is deprecated. Use `from {submod.__name__} import {name}` instead",
        FutureWarning,
    )
    return getattr(submod, name)
