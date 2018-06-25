import os
os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'
from .utils import jit, integrate, gen_coeffs
from .layered_model import model_nlayer_ss, model_nlayer_fd, model_nlayer_g2
import numpy as np
import numba as nb
import numba.cuda


@nb.cuda.jit
def cuda_model_nlayer_ss(rhos, muas, musps, depths, n, n_ext, int_limit, int_divs, eps, out):
    """Model Steady-State Reflectance in N Layers with Extrapolated Boundary Condition.
    Source: "Noninvasive determination of the optical properties of two-layered turbid media"
    parameters:
        rho := Source-Detector Seperation [length]
        mua := N Absorption Coefficents [1/length]
        musp := N Reduced Scattering Coefficents [1/length]
        depths := N-1 Layer Depths
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        int_limit := Integration Limit [length]
        int_divs := Number of subregions to integrate over []
        eps := Epsilion for numerical diffrention [length]
    """
    nlayer = muas.shape[1]
    i = nb.cuda.grid(1)
    if i > out.shape[0]:
        return
    rho = rhos[i]
    mua = muas[i]
    musp = musps[i]
    out[i] = model_nlayer_ss(rho, mua, musp, depths, n, n_ext, int_limit, int_divs, eps)


@nb.cuda.jit
def cuda_model_nlayer_fd(rhos, muas, musps, depths, freq, c, n, n_ext, int_limit, int_divs, eps, out):
    """Model Frequncy-Domain Reflectance in N Layers with Extrapolated Boundary Condition.
    Source: "Noninvasive determination of the optical properties of two-layered turbid media"
    parameters:
        rho := Source-Detector Seperation [length]
        mua := N Absorption Coefficents [1/length]
        musp := N Reduced Scattering Coefficents [1/length]
        depths := N-1 Layer Depths
        freq := Frequncy of Source [1/time]
        c := Speed of Light in vacuum [length/time]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        int_limit := Integration Limit [length]
        int_divs := Number of subregions to integrate over []
        eps := Epsilion for numerical diffrention [length]
    """
    nlayer = muas.shape[1]
    i = nb.cuda.grid(1)
    if i > out.shape[0]:
        return
    rho = rhos[i]
    mua = muas[i]
    musp = musps[i]
    out[i] = model_nlayer_fd(rho, mua, musp, depths, freq, c, n, n_ext, int_limit, int_divs, eps)


@nb.cuda.jit
def cuda_model_nlayer_g2(rhos, taus, muas, musps, depths, BFi, wavelengths, n, n_ext, beta, tau_0, int_limit, int_divs, eps, out):
    """Model g2 (autocorelation) for Diffuse Correlation Spectroscopy in 4 Layers with Extrapolated Boundary Condition.
    Source1: "Noninvasive determination of the optical properties of two-layered turbid media"
    Source2: "Diffuse optics for tissue monitoring and tomography"
    parameters:
        rho := Source-Detector Seperation [length]
        tau := Correlation Time [time]
        mua := N Absorption Coefficents [1/length]
        musp := N Reduced Scattering Coefficents [1/length]
        depths := N-1 Layer Depths
        BFi := N Blood Flow indices []
        wavelength := Measurement Wavelength [length]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        beta := Beta derived for Siegert relation []
        tau_0 := The first tau for normalization [time]
        int_limit := Integration Limit [length]
        int_divs := Number of subregions to integrate over []
        eps := Epsilion for numerical diffrention [length]
    """
    nlayer = muas.shape[1]
    i = nb.cuda.grid(1)
    if i > out.shape[0]:
        return
    rho = rhos[i]
    tau = taus[i]
    mua = muas[i]
    musp = musps[i]
    wavelength = wavelengths[i]
    out[i] = model_nlayer_g2(rho, tau, mua, musp, depths, BFi, wavelength, n, n_ext, beta, tau_0, int_limit, int_divs, eps, out)
