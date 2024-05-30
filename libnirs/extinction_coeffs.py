import os

import numpy as np
from scipy.interpolate import interp1d

_extinctions = np.load(os.path.join(os.path.dirname(__file__), "extc.npz"))
_extinction_interps = {k: interp1d(*v) for k, v in _extinctions.items()}


def water(wavelength: "Quantity[ArrayLike, 'nm']") -> "Quantity[ndarray, 1 / 'mm']":
    """Get Extinction Coefficent [1/fraction/mm] of water at given wavelength(s) [nm]"""
    return _extinction_interps["water"](wavelength)


def oxyhemoglobin(wavelength: "Quantity[ArrayLike, 'nm']") -> "Quantity[ndarray, 1 / 'molar' / 'mm']":
    """Get Extinction Coefficent [1/Molar/mm] of oxyhemoglobin [HbO] at given wavelength(s) [nm]"""
    return _extinction_interps["HbO"](wavelength)


def deoxyhemoglobin(wavelength: "Quantity[ArrayLike, 'nm']") -> "Quantity[ndarray, 1 / 'molar' / 'mm']":
    """Get Extinction Coefficent [1/Molar/mm] of deoxyhemoglobin [HbR] at given wavelength(s) [nm]"""
    return _extinction_interps["HbR"](wavelength)


def soy_oil(wavelength: "Quantity[ArrayLike, 'nm']") -> "Quantity[ndarray, 1 / 'mm']":
    """Get Extinction Coefficent [1/fraction/mm] of soy-oil at given wavelength(s) [nm]"""
    return _extinction_interps["soyoil"](wavelength)


def get_extinction_coeffs(wavelengths, *species, axis=0):
    """Get Extinction Coefficents of species at various wavelengths in an array.
    Units:
        water := [1/fraction/mm]
        HbO := [1/Molar/mm]
        HbR := [1/Molar/mm]
        soyoil := [1/fraction/mm]
    """
    import warnings

    warnings.warn(
        "Function `get_extinction_coeffs` is now deprecated!", category=warnings.DeprecationWarning, stacklevel=2
    )
    if len(species) == 0:
        raise Exception("No species given.")
    elif len(species) == 1:
        return _extinction_interps[species[0]](wavelengths)
    else:
        return np.stack([_extinction_interps[k](wavelengths) for k in species], axis=axis)
