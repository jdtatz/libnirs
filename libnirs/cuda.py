from .model import _model_ss, _model_ss_sig, _model_fd, _model_fd_sig, _model_td, _model_td_sig, _model_g1, _model_g1_sig, _model_g2, _model_g2_sig
from .layered_model import model_nlayer_ss, model_nlayer_fd, model_nlayer_g1, model_nlayer_g2
import numpy as np
from numba import vectorize
import numba.cuda as cuda

cuda_model_ss = vectorize(_model_ss_sig, target="cuda")(_model_ss)
cuda_model_fd = vectorize(_model_fd_sig, target="cuda")(_model_fd)
cuda_model_td = vectorize(_model_td_sig, target="cuda")(_model_td)
cuda_model_g1 = vectorize(_model_g1_sig, target="cuda")(_model_g1)
cuda_model_g2 = vectorize(_model_g2_sig, target="cuda")(_model_g2)


@cuda.jit
def cuda_model_nlayer_ss(rhos, muas, musps, depths, n, n_ext, int_limit, int_divs, out):
    """Model Steady-State Reflectance in N Layers with Partial-Current Boundary Condition.
    Source: "Noninvasive determination of the optical properties of two-layered turbid media"
    parameters:
        rhos := Source-Detector Seperation [length]
        muas := N Absorption Coefficents [1/length]
        musps := N Reduced Scattering Coefficents [1/length]
        depths := N-1 Layer Depths
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        int_limit := Integration Limit [length]
        int_divs := Number of subregions to integrate over []
        out := Output Array
    """
    i = cuda.grid(1)
    if i > out.shape[0]:
        return
    rho = rhos[i]
    mua = muas[i]
    musp = musps[i]
    out[i] = model_nlayer_ss(rho, mua, musp, depths, n, n_ext, int_limit, int_divs)


@cuda.jit
def cuda_model_nlayer_fd(rhos, muas, musps, depths, freq, c, n, n_ext, int_limit, int_divs, out):
    """Model Frequncy-Domain Reflectance in N Layers with Partial-Current Boundary Condition.
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
        out := Output Array
    """
    i = cuda.grid(1)
    if i > out.shape[0]:
        return
    rho = rhos[i]
    mua = muas[i]
    musp = musps[i]
    out[i] = model_nlayer_fd(rho, mua, musp, depths, freq, c, n, n_ext, int_limit, int_divs)


@cuda.jit
def cuda_model_nlayer_g1(rhos, taus, muas, musps, depths, BFi, wavelengths, n, n_ext, tau_0, int_limit, int_divs, out):
    """Model g1 (autocorelation) for Diffuse Correlation Spectroscopy in N Layers with Partial-Current Boundary Condition.
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
        tau_0 := The first tau for normalization [time]
        int_limit := Integration Limit [length]
        int_divs := Number of subregions to integrate over []
        out := Output Array
    """
    i = cuda.grid(1)
    if i > out.shape[0]:
        return
    rho = rhos[i]
    tau = taus[i]
    mua = muas[i]
    musp = musps[i]
    wavelength = wavelengths[i]
    out[i] = model_nlayer_g1(rho, tau, mua, musp, depths, BFi, wavelength, n, n_ext, tau_0, int_limit, int_divs)

@cuda.jit
def cuda_model_nlayer_g2(rhos, taus, muas, musps, depths, BFi, wavelengths, n, n_ext, beta, tau_0, int_limit, int_divs, out):
    """Model g2 (autocorelation) for Diffuse Correlation Spectroscopy in N Layers with Partial-Current Boundary Condition.
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
        out := Output Array
    """
    i = cuda.grid(1)
    if i > out.shape[0]:
        return
    rho = rhos[i]
    tau = taus[i]
    mua = muas[i]
    musp = musps[i]
    wavelength = wavelengths[i]
    out[i] = model_nlayer_g2(rho, tau, mua, musp, depths, BFi, wavelength, n, n_ext, beta, tau_0, int_limit, int_divs)
