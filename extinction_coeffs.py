import os
import numpy as np
from scipy.interpolate import interp1d

_extinctions = np.load(os.path.join(os.path.dirname(__file__), 'extc.npz'))
_extinction_interps = {k: interp1d(*v) for k, v in _extinctions.items()}

def get_extinction_coeffs(wavelengths, *species):
    """Get Extinction Coefficents of species at various wavelengths in an array.
    Units:
        water := [1/fraction/mm]
        HbO := [1/Molar/mm]
        HbR := [1/Molar/mm]
        soyoil := [1/fraction/mm]
    """
    if len(species) == 0:
        raise Exception("No species given.")
    elif len(species) == 1:
        return _extinction_interps[species[0]](wavelengths)
    else:
        return np.stack(_extinction_interps[k](wavelengths) for k in species)
