import os
os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'
from .utils import jit, integrate, gen_coeffs
from .model import model_ss, model_fd, model_td, model_g2
from .layered_model import model_nlayer_ss, model_nlayer_fd, model_nlayer_g2
import numpy as np
import numba as nb
import numba.cuda


@nb.cuda.jit
def cuda_model_ss(rhos, muas, musps, n, n_ext, out):
    """Model Steady-State Reflectance with Extrapolated Boundary Condition.
    Source: "Improved solutions of the steady-state and the time-resolved diffusion equations for reflectance from a semi-infinite turbid medium"
    parameters:
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        out := Output Array
    """
    i = nb.cuda.grid(1)
    if i > out.shape[0]:
        return
    rho = rhos[i]
    mua = muas[i]
    musp = musps[i]
    out[i] = model_ss(rho, mua, musp, n, n_ext)


@nb.cuda.jit
def cuda_model_fd(rhos, muas, musps, n, n_ext, freq, c, out):
    """Model Frequncy-Domain Reflectance with Extrapolated Boundary Condition.
    Source: "Improved solutions of the steady-state and the time-resolved diffusion equations for reflectance from a semi-infinite turbid medium"
    parameters:
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        freq := Frequncy of Source [1/time]
        c := Speed of Light in vacuum [length/time]
        out := Output Array
    """
    i = nb.cuda.grid(1)
    if i > out.shape[0]:
        return
    rho = rhos[i]
    mua = muas[i]
    musp = musps[i]
    out[i] = model_fd(rho, mua, musp, n, n_ext, freq, c)


@nb.cuda.jit
def cuda_model_td(ts, rhos, muas, musps, n, n_ext, c, out):
    """Model Time-Domain Reflectance with Extrapolated Boundary Condition.
    Source: "Improved solutions of the steady-state and the time-resolved diffusion equations for reflectance from a semi-infinite turbid medium"
    parameters:
        t := Time of Flight [time]
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        c := Speed of Light in vacuum [length/time]
        out := Output Array
    """
    i = nb.cuda.grid(1)
    if i > out.shape[0]:
        return
    t = ts[i]
    rho = rhos[i]
    mua = muas[i]
    musp = musps[i]
    out[i] = model_td(t, rho, mua, musp, n, n_ext, c)


@nb.cuda.jit
def cuda_model_g2(taus, bfi, beta, muas, musps, wavelengths, rhos, first_tau_delay, n, n_ext, out):
    """Model g2 (autocorelation) for Diffuse correlation spectroscopy.
    Source: "Diffuse optics for tissue monitoring and tomography"
    parameters:
        tau := Correlation time [time]
        bfi := Blood-Flow Index []
        beta := Beta derived for Siegert relation []
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        rho := Source-Detector Seperation [length]
        first_tau_delay := The first tau for normalization [time]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        out := Output Array
    """
    i = nb.cuda.grid(1)
    if i > out.shape[0]:
        return
    tau = taus[i]
    rho = rhos[i]
    tau = taus[i]
    mua = muas[i]
    musp = musps[i]
    wavelength = wavelengths[i]
    out[i] = model_g2(tau, bfi, beta, mua, musp, wavelength, rho, first_tau_delay, n, n_ext)



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
        out := Output Array
    """
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
        out := Output Array
    """
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
        out := Output Array
    """
    i = nb.cuda.grid(1)
    if i > out.shape[0]:
        return
    rho = rhos[i]
    tau = taus[i]
    mua = muas[i]
    musp = musps[i]
    wavelength = wavelengths[i]
    out[i] = model_nlayer_g2(rho, tau, mua, musp, depths, BFi, wavelength, n, n_ext, beta, tau_0, int_limit, int_divs, eps)
