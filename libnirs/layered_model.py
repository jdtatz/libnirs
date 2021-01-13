import numpy as np
from numpy import pi, exp, sqrt, sinh, cosh
from scipy import fft
from scipy.special import j0, gamma
from .utils import jit, integrate, gen_impedance
from .model import model_ss, model_fd, model_g2

"""
WARNING
The D and alpha values are not precomputed due to the fact that the
number of layers isn't known during numba compilation, and the dynamic
memory allocation needed for storing the precumputed values isn't allowed
for the cuda target.
"""

@jit
def _n_layer_refl(s, z0, zb, ls, muas, musps, alphas, alpha_args):
    n = len(muas)

    def ds(i):
        return 1 / (3 * (muas[i] + musps[i]))

    d2 = ds(n-1)
    d1 = ds(n-2)
    alpha2 = alphas(s, n-1, d2, *alpha_args)
    alpha1 = alphas(s, n-2, d1, *alpha_args)

    if n == 2:
        l1 = ls[0]
        signal = (cosh((-z0 + l1)*alpha1)*d1*alpha1 + sinh((-z0 + l1)*alpha1)*d2*alpha2) / \
            (d1*alpha1*(cosh(l1*alpha1) + zb*sinh(l1*alpha1)*alpha1) + d2*(sinh(l1*alpha1) + zb*cosh(l1*alpha1)*alpha1)*alpha2)
        return signal

    k_num = (alpha1 * d1 - alpha2 * d2) * exp(-ls[-1] * (alpha1 + alpha2))
    k_dem = (alpha1 * d1 + alpha2 * d2) * exp( ls[-1] * (alpha1 - alpha2))

    d1, d2 = ds(n-3), d1
    alpha1, alpha2 = alphas(s, n-3, d1, *alpha_args), alpha1
    for i in range(n - 3, 0, -1):
        adi = alpha1 * d1
        adn = alpha2 * d2
        kaa = (adi + adn) * exp(-ls[i] * (alpha1 - alpha2))
        kab = (adi - adn) * exp(-ls[i] * (alpha1 + alpha2))
        kba = (adi - adn) * exp( ls[i] * (alpha1 + alpha2))
        kbb = (adi + adn) * exp( ls[i] * (alpha1 - alpha2))
        k_num, k_dem = kaa * k_num + kab * k_dem, kba * k_num + kbb * k_dem

        d1, d2 = ds(i-1), d1
        alpha1, alpha2 = alphas(s, i - 1, d1, *alpha_args), alpha1
    l1 = ls[0]
    signal = (cosh((-z0 + l1)*alpha1)*(k_dem + exp(2*l1*alpha2)*k_num)*d1*alpha1 + (k_dem - exp(2*l1*alpha2)*k_num)*sinh((-z0 + l1)*alpha1)*d2*alpha2) / \
        ((k_dem + exp(2*l1*alpha2)*k_num)*d1*alpha1*(cosh(l1*alpha1) + zb*sinh(l1*alpha1)*alpha1) + (k_dem - exp(2*l1*alpha2)*k_num)*d2*(sinh(l1*alpha1) + zb*cosh(l1*alpha1)*alpha1)*alpha2)
    return signal


@jit
def _refl_integrator(s, rho, z0, zb, ls, muas, musps, alphas, alpha_args):
    """Inverse HankelTransform"""
    return s*j0(s*rho)*_n_layer_refl(s, z0, zb, ls, muas, musps, alphas, alpha_args)


def fourierBesselJv(omega):
    return gamma((1 - 1j * omega) / 2) / gamma((1 + 1j * omega) / 2) * 2**(-1j * omega)


def rfft_hankel(integrator, integrator_args, log_limit, npoints):
    """Inverse HankelTransform using Fast Fourier Transform"""
    sp = np.linspace(-log_limit, log_limit, fft.next_fast_len(npoints, True))
    dt = sp[1] - sp[0]
    s = np.exp(-sp)
    r = np.exp(sp)
    res = s * np.nan_to_num(integrator(s, *integrator_args))
    wq = 2 * pi * fft.rfftfreq(len(sp), dt)
    fres = fft.rfft(fft.ifftshift(res), norm="ortho") * dt
    fcres = fourierBesselJv(wq) * fres
    hres = fft.ifftshift(fft.irfft(fcres, norm="ortho")) * sp.size * (wq[1] - wq[0]) / (2*pi)
    return r, hres / r


def fft_hankel(integrator, integrator_args, log_limit, npoints):
    """Inverse HankelTransform using Fast Fourier Transform"""
    sp = np.linspace(-log_limit, log_limit, fft.next_fast_len(npoints, True))
    dt = sp[1] - sp[0]
    s = np.exp(-sp)
    r = np.exp(sp)
    res = s * np.nan_to_num(integrator(s, *integrator_args))
    wq = 2 * pi * fft.fftfreq(len(sp), dt)
    fres = fft.fft(fft.ifftshift(res), norm="ortho") * dt
    fcres = fourierBesselJv(wq) * fres
    hres = fft.ifftshift(fft.ifft(fcres, norm="ortho")) * sp.size * (wq[1] - wq[0]) / (2*pi)
    return r, hres / r


@jit
def _ss_alphas(s, i, d, muas):
    return sqrt(s**2 + muas[i] / d)


@jit
def model_nlayer_ss(rho, mua, musp, depths, n, n_ext=1, int_limit=10, int_divs=10):
    """Model Steady-State Reflectance in N Layers with Partial-Current Boundary Condition.
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
    """
    nlayer = len(mua)
    imp = gen_impedance(n/n_ext)
    D1 = 1 / (3 * (mua[0] + musp[0]))
    z0 = 3*D1
    assert depths[0] >= z0
    zb = 2*D1*imp
    alpha_args = (mua,)
    return integrate(_refl_integrator, 0, int_limit, int_divs, (rho, z0, zb, depths, mua, musp, _ss_alphas, alpha_args)) / (2*pi)


def model_nlayer_ss_fft(rho, mua, musp, depths, n, n_ext=1, log_limit=15, npoints=512):
    """Model Steady-State Reflectance in N Layers with Partial-Current Boundary Condition.
    Source: "Noninvasive determination of the optical properties of two-layered turbid media"
    parameters:
        rho := Source-Detector Seperation [length]
        mua := N Absorption Coefficents [1/length]
        musp := N Reduced Scattering Coefficents [1/length]
        depths := N-1 Layer Depths
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        log_limit := Log-Space Integration Limit for fft []
        npoints := Minimum number of points to evalulate for fft []
    """
    nlayer = len(mua)
    imp = gen_impedance(n / n_ext)
    D1 = 1 / (3 * (mua[0] + musp[0]))
    z0 = 3*D1
    assert depths[0] >= z0
    zb = 2*D1*imp
    alpha_args = (mua,)
    r, h = rfft_hankel(_n_layer_refl, (z0, zb, depths, mua, musp, _ss_alphas, alpha_args), log_limit, npoints)
    return np.exp(np.interp(rho, r, np.log(h))) / (2*pi)


@jit
def _fd_alphas(s, i, d, wave, muas):
    return sqrt(s**2 + (wave + muas[i]) / d)


@jit
def model_nlayer_fd(rho, mua, musp, depths, freq, c, n, n_ext=1, int_limit=10, int_divs=10):
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
    """
    nlayer = len(mua)
    imp = gen_impedance(n/n_ext)
    D1 = 1 / (3 * (mua[0] + musp[0]))
    z0 = 3*D1
    assert depths[0] >= z0
    zb = 2*D1*imp
    w = 2*pi*freq
    v = c / n
    wave = w/v*1j
    alpha_args = wave, mua
    return integrate(_refl_integrator, 0, int_limit, int_divs, (rho, z0, zb, depths, mua, musp, _fd_alphas, alpha_args)) / (2*pi)


def model_nlayer_fd_fft(rho, mua, musp, depths, freq, c, n, n_ext=1, log_limit=15, npoints=512):
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
        log_limit := Log-Space Integration Limit for fft []
        npoints := Minimum number of points to evalulate for fft []
    """
    nlayer = len(mua)
    imp = gen_impedance(n/n_ext)
    D1 = 1 / (3 * (mua[0] + musp[0]))
    z0 = 3*D1
    assert depths[0] >= z0
    zb = 2*D1*imp
    w = 2*pi*freq
    v = c / n
    wave = w/v*1j
    alpha_args = wave, mua
    r, h = fft_hankel(_n_layer_refl, (z0, zb, depths, mua, musp, _fd_alphas, alpha_args), log_limit, npoints)
    return np.exp(np.interp(rho, r, np.log(h))) / (2*pi)


@jit
def _g1_alphas(s, i, d, muas, musps, k0, BFi, tau):
    return sqrt(s**2 + (muas[i] + 2 * musps[i] * k0**2 * BFi[i] * tau) / d)


@jit
def model_nlayer_g1(rho, tau, mua, musp, depths, BFi, wavelength, n, n_ext=1, tau_0=0, int_limit=10, int_divs=10):
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
    """
    nlayer = len(mua)
    imp = gen_impedance(n/n_ext)
    D1 = 1 / (3 * (mua[0] + musp[0]))
    z0 = 3*D1
    assert depths[0] >= z0
    zb = 2*D1*imp
    k0 = 2 * pi * n / wavelength
    alpha_args = mua, musp, k0, BFi, tau
    alpha_norm_args = mua, musp, k0, BFi, tau_0
    refl = integrate(_refl_integrator, 0, int_limit, int_divs, (rho, z0, zb, depths, mua, musp, _g1_alphas, alpha_args))
    refl_norm = integrate(_refl_integrator, 0, int_limit, int_divs, (rho, z0, zb, depths, mua, musp, _g1_alphas, alpha_norm_args))
    g1 = refl / refl_norm
    return g1


def model_nlayer_g1_fft(rho, tau, mua, musp, depths, BFi, wavelength, n, n_ext=1, tau_0=0, log_limit=15, npoints=512):
    """Model g1 (autocorelation) for Diffuse Correlation Spectroscopy in N Layers with Partial-Current Boundary Condition.
    Source1: "Noninvasive determination of the optical properties of two-layered turbid media"
    Source2: "Diffuse optics for tissue monitoring and tomography"
    parameters:
        rho := Source-Detector Seperation [length]
        tau := Correlation Time [time]
        mua := N Absorption Coefficents [1/length]
        musp := N Reduced Scattering Coefficents [1/length]
        depths := N-1 Layer Depths [length]
        BFi := N Blood Flow indices []
        wavelength := Measurement Wavelength [length]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        tau_0 := The first tau for normalization [time]
        log_limit := Log-Space Integration Limit for fft []
        npoints := Minimum number of points to evalulate for fft []
    """
    nlayer = len(mua)
    imp = gen_impedance(n/n_ext)
    D1 = 1 / (3 * (mua[0] + musp[0]))
    z0 = 3*D1
    assert depths[0] >= z0
    zb = 2*D1*imp
    k0 = 2 * pi * n / wavelength
    alpha_args = mua, musp, k0, BFi, tau
    alpha_norm_args = mua, musp, k0, BFi, tau_0
    r, h = rfft_hankel(_n_layer_refl, (z0, zb, depths, mua, musp, _g1_alphas, alpha_args), log_limit, npoints)
    _r, h_norm = rfft_hankel(_n_layer_refl, (z0, zb, depths, mua, musp, _g1_alphas, alpha_norm_args), log_limit, npoints)
    g1 = np.interp(rho, r, (h / h_norm))
    return g1


@jit
def model_nlayer_g2(rho, tau, mua, musp, depths, BFi, wavelength, n, n_ext=1, beta=0.5, tau_0=0, int_limit=10, int_divs=10):
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
    """
    g1 = model_nlayer_g2(rho, tau, mua, musp, depths, BFi, wavelength, n, n_ext, tau_0, int_limit, int_divs)
    return 1 + beta * g1**2
