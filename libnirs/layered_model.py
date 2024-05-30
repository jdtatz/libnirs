from itertools import starmap

import numpy as np
import scipy.fft
from numba import guvectorize
from numpy import cosh, exp, pi, sinh, sqrt
from scipy.special import gamma, j0

from .model import model_fd, model_g2, model_ss
from .utils import gen_impedance, integrate, jit, map_tuples


@jit
def _n_layer_refl(s, z0, zb, ls, D, k2):
    """alpha = sqrt(s^2 + k^2)"""
    n = len(D)
    s2 = s * s

    d2 = D[n - 1]
    d1 = D[n - 2]
    alpha2 = sqrt(s2 + k2[n - 1])
    alpha1 = sqrt(s2 + k2[n - 2])

    if n == 2:
        l1 = ls[0]
        signal = (cosh((-z0 + l1) * alpha1) * d1 * alpha1 + sinh((-z0 + l1) * alpha1) * d2 * alpha2) / (
            d1 * alpha1 * (cosh(l1 * alpha1) + zb * sinh(l1 * alpha1) * alpha1)
            + d2 * (sinh(l1 * alpha1) + zb * cosh(l1 * alpha1) * alpha1) * alpha2
        )
        return signal

    k_num = (alpha1 * d1 - alpha2 * d2) * exp(-ls[-1] * (alpha1 + alpha2))
    k_dem = (alpha1 * d1 + alpha2 * d2) * exp(ls[-1] * (alpha1 - alpha2))

    d1, d2 = D[n - 3], d1
    alpha1, alpha2 = sqrt(s2 + k2[n - 3]), alpha1
    for i in range(n - 3, 0, -1):
        adi = alpha1 * d1
        adn = alpha2 * d2
        kaa = (adi + adn) * exp(-ls[i] * (alpha1 - alpha2))
        kab = (adi - adn) * exp(-ls[i] * (alpha1 + alpha2))
        kba = (adi - adn) * exp(ls[i] * (alpha1 + alpha2))
        kbb = (adi + adn) * exp(ls[i] * (alpha1 - alpha2))
        k_num, k_dem = kaa * k_num + kab * k_dem, kba * k_num + kbb * k_dem

        d1, d2 = D[i - 1], d1
        alpha1, alpha2 = sqrt(s2 + k2[i - 1]), alpha1
    l1 = ls[0]
    signal = (
        cosh((-z0 + l1) * alpha1) * (k_dem + exp(2 * l1 * alpha2) * k_num) * d1 * alpha1
        + (k_dem - exp(2 * l1 * alpha2) * k_num) * sinh((-z0 + l1) * alpha1) * d2 * alpha2
    ) / (
        (k_dem + exp(2 * l1 * alpha2) * k_num) * d1 * alpha1 * (cosh(l1 * alpha1) + zb * sinh(l1 * alpha1) * alpha1)
        + (k_dem - exp(2 * l1 * alpha2) * k_num) * d2 * (sinh(l1 * alpha1) + zb * cosh(l1 * alpha1) * alpha1) * alpha2
    )
    return signal


@guvectorize("(),(),(),(n),(n),(n)->()", nopython=True, target="cpu")
def _vectorize_n_layer_refl(s, z0, zb, ls, D, k2, result):
    result[()] = _n_layer_refl(s, z0, zb, ls, D, k2)


_cuda_vectorize_n_layer_refl_fn = None


def _get_cuda_vectorize_n_layer_refl():
    global _cuda_vectorize_n_layer_refl_fn

    if _cuda_vectorize_n_layer_refl_fn is None:
        _ty_sig = lambda ty: f"({ty}, {ty}, {ty}, {ty}[:], {ty}[:], {ty}[:], {ty}[:])"

        @guvectorize(
            [_ty_sig("f4"), _ty_sig("f8"), _ty_sig("c8"), _ty_sig("c16")],
            "(),(),(),(n),(n),(n)->()",
            nopython=True,
            target="cuda",
        )
        def _cuda_vectorize_n_layer_refl(s, z0, zb, ls, D, k2, result):
            result[()] = _n_layer_refl(s, z0, zb, ls, D, k2)

        _cuda_vectorize_n_layer_refl_fn = _cuda_vectorize_n_layer_refl
    return _cuda_vectorize_n_layer_refl_fn


@jit
def _D(mua, musp):
    return 1 / (3 * (mua + musp))


@jit
def _refl_integrator(s, rho, z0, zb, ls, D, k2):
    """Naive Inverse HankelTransform"""
    return s * j0(s * rho) * _n_layer_refl(s, z0, zb, ls, D, k2)


def _map_tuples(func, *tuples):
    return tuple(starmap(func, zip(*tuples)))


def _expand_dims(arr, axis=0):
    return np.expand_dims(arr, axis=axis) if np.shape(arr) != () else arr


def _expand_last_dim(arr):
    return _expand_dims(arr, axis=-1)


def _expand_first_ndim(arr, ndim):
    return _expand_dims(arr, axis=tuple(range(ndim)))


def fourierBesselJv(omega):
    return gamma((1 - 1j * omega) / 2) / gamma((1 + 1j * omega) / 2) * 2 ** (-1j * omega)


def fft_hankel(
    integrator, integrator_args, integrator_kwargs, log_limit, npoints, in_ndim=0, is_complex=True, fft=scipy.fft
):
    """Inverse HankelTransform using Fast Fourier Transform"""
    if hasattr(fft, "next_fast_len"):
        npoints = fft.next_fast_len(npoints, not is_complex)
    sp = np.linspace(-log_limit, log_limit, npoints)
    dt = sp[1] - sp[0]
    wq = 2 * pi * (fft.fftfreq if is_complex else fft.rfftfreq)(len(sp), dt)
    dw = wq[1] - wq[0]
    sp = _expand_first_ndim(sp, in_ndim)
    wq = _expand_first_ndim(wq, in_ndim)
    s = np.exp(-sp)
    r = np.exp(sp)
    # Weird tricks ahead to reduce memory usage
    res = np.nan_to_num(integrator(s, *integrator_args, **integrator_kwargs), copy=False)
    res *= s
    shifted = fft.ifftshift(res, axes=-1)
    del res
    fres = (fft.fft if is_complex else fft.rfft)(shifted, norm="ortho", axis=-1)
    del shifted
    fres *= dt
    fres *= fourierBesselJv(wq)
    shifted_hres = (fft.ifft if is_complex else fft.irfft)(fres, norm="ortho", axis=-1)
    del fres
    hres = fft.ifftshift(shifted_hres, axes=-1)
    del shifted_hres
    hres *= sp.size * dw / (2 * pi)
    hres /= r
    return sp, hres


def _refl_fft_hankel(z0, zb, depths, D, k2, log_limit, npoints, use_gpu=False):
    in_ndim = np.broadcast(*D, *k2, *depths).ndim
    move_tuple_axis = lambda tup: np.moveaxis(np.broadcast_arrays(*tup), 0, -1)
    depths = move_tuple_axis((*depths, np.inf))
    D = move_tuple_axis(D)
    k2 = move_tuple_axis(k2)
    is_complex = np.iscomplexobj(k2)
    contiguous_expand = lambda a: np.ascontiguousarray(_expand_dims(a, axis=in_ndim))
    args = (contiguous_expand(a) for a in (z0, zb, depths, D, k2))
    if use_gpu:
        import cupy

        refl = _get_cuda_vectorize_n_layer_refl()
        args = map(cupy.asarray, args)
        fft = cupy.fft
    else:
        refl = _vectorize_n_layer_refl
        fft = scipy.fft
    sp, hres = fft_hankel(refl, args, dict(axis=-1), log_limit, npoints, in_ndim, is_complex=is_complex, fft=fft)
    if use_gpu:
        import cupy

        sp = cupy.asnumpy(sp)
        hres = cupy.asnumpy(hres)
    return sp, hres


def _deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    import functools
    import warnings

    reason = "use fft version instead, naive hankel tranform integration has uncontrollable numerical error"

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter("always", DeprecationWarning)  # turn off filter
        warnings.warn(
            f"Call to deprecated function {func.__name__} ({reason}).", category=DeprecationWarning, stacklevel=2
        )
        warnings.simplefilter("default", DeprecationWarning)  # reset filter
        return func(*args, **kwargs)

    return new_func


@jit
def _ss_k2(mua, D):
    return mua / D


@_deprecated
def model_nlayer_ss(rho, mua, musp, depths, n, n_ext=1, int_limit=10, int_divs=10):
    """Model Steady-State Reflectance in N Layers with Partial-Current Boundary Condition.
    Source: "Noninvasive determination of the optical properties of two-layered turbid media"
    parameters:
        rho := Source-Detector Seperation [length]
        mua := N Absorption Coefficients [1/length]
        musp := N Reduced Scattering Coefficients [1/length]
        depths := N-1 Layer Depths
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        int_limit := Integration Limit [length]
        int_divs := Number of subregions to integrate over []
    """
    nlayer = len(mua)
    imp = gen_impedance(n / n_ext)
    D = map_tuples(_D, mua, musp)
    D1 = D[0]
    z0 = 3 * D1
    assert depths[0] >= z0
    zb = 2 * D1 * imp
    k2 = map_tuples(_ss_k2, mua, D)
    return integrate(_refl_integrator, 0, int_limit, int_divs, (rho, z0, zb, depths, D, k2)) / (2 * pi)


def model_nlayer_ss_fft(mua, musp, depths, n, n_ext=1, log_limit=15, npoints=512, use_gpu=False):
    """Model Steady-State Reflectance in N Layers with Partial-Current Boundary Condition.
    Source: "Noninvasive determination of the optical properties of two-layered turbid media"
    parameters:
        mua := N Absorption Coefficients [1/length]
        musp := N Reduced Scattering Coefficients [1/length]
        depths := N-1 Layer Depths
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        log_limit := Log-Space Integration Limit for fft []
        npoints := Minimum number of points to evaluate for fft []
    """
    imp = gen_impedance(n / n_ext)
    D = _map_tuples(_D, mua, musp)
    D1 = D[0]
    z0 = 3 * D1
    # assert np.all(depths[0] >= z0)
    zb = 2 * D1 * imp
    k2 = _map_tuples(_ss_k2, mua, D)
    sp, h = _refl_fft_hankel(z0, zb, depths, D, k2, log_limit, npoints, use_gpu)
    h /= 2 * pi
    return np.squeeze(sp), h


@jit
def _fd_k2(mua, D, wave):
    return (wave + mua) / D


@_deprecated
def model_nlayer_fd(rho, mua, musp, depths, freq, c, n, n_ext=1, int_limit=10, int_divs=10):
    """Model Frequency-Domain Reflectance in N Layers with Partial-Current Boundary Condition.
    Source: "Noninvasive determination of the optical properties of two-layered turbid media"
    parameters:
        rho := Source-Detector Seperation [length]
        mua := N Absorption Coefficents [1/length]
        musp := N Reduced Scattering Coefficents [1/length]
        depths := N-1 Layer Depths
        freq := Frequency of Source [1/time]
        c := Speed of Light in vacuum [length/time]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        int_limit := Integration Limit [length]
        int_divs := Number of subregions to integrate over []
    """
    nlayer = len(mua)
    imp = gen_impedance(n / n_ext)
    D = map_tuples(_D, mua, musp)
    D1 = D[0]
    z0 = 3 * D1
    assert depths[0] >= z0
    zb = 2 * D1 * imp
    w = 2 * pi * freq
    v = c / n
    wave = w / v * 1j
    k2 = map_tuples(lambda a, d: _fd_k2(a, d, wave), mua, D)
    return integrate(_refl_integrator, 0, int_limit, int_divs, (rho, z0, zb, depths, D, k2)) / (2 * pi)


def model_nlayer_fd_fft(mua, musp, depths, freq, c, n, n_ext=1, log_limit=15, npoints=512, use_gpu=False):
    """Model Frequency-Domain Reflectance in N Layers with Partial-Current Boundary Condition.
    Source: "Noninvasive determination of the optical properties of two-layered turbid media"
    parameters:
        mua := N Absorption Coefficents [1/length]
        musp := N Reduced Scattering Coefficents [1/length]
        depths := N-1 Layer Depths
        freq := Frequency of Source [1/time]
        c := Speed of Light in vacuum [length/time]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        log_limit := Log-Space Integration Limit for fft []
        npoints := Minimum number of points to evaluate for fft []
    """
    imp = gen_impedance(n / n_ext)
    D = _map_tuples(_D, mua, musp)
    D1 = D[0]
    z0 = 3 * D1
    # assert np.all(depths[0] >= z0)
    zb = 2 * D1 * imp
    w = 2 * pi * freq
    v = c / n
    wave = w / v * 1j
    k2 = _map_tuples(lambda a, d: _fd_k2(a, d, wave), mua, D)
    sp, h = _refl_fft_hankel(z0, zb, depths, D, k2, log_limit, npoints, use_gpu)
    h /= 2 * pi
    return np.squeeze(sp), h


@jit
def _g1_k2(mua, musp, D, BFi, k0, tau):
    return (mua + 2 * musp * k0**2 * BFi * tau) / D


@_deprecated
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
    imp = gen_impedance(n / n_ext)
    D = map_tuples(_D, mua, musp)
    D1 = D[0]
    z0 = 3 * D1
    assert depths[0] >= z0
    zb = 2 * D1 * imp
    k0 = 2 * pi * n / wavelength
    k2 = map_tuples(lambda a, sp, d, b: _g1_k2(a, sp, d, b, k0, tau), mua, musp, D, BFi)
    k2_norm = map_tuples(lambda a, sp, d, b: _g1_k2(a, sp, d, b, k0, tau_0), mua, musp, D, BFi)
    refl = integrate(_refl_integrator, 0, int_limit, int_divs, (rho, z0, zb, depths, D, k2))
    refl_norm = integrate(_refl_integrator, 0, int_limit, int_divs, (rho, z0, zb, depths, D, k2_norm))
    g1 = refl / refl_norm
    return g1


def model_nlayer_g1_fft(
    tau, mua, musp, depths, BFi, wavelength, n, n_ext=1, tau_0=0, log_limit=15, npoints=512, use_gpu=False
):
    """Model g1 (autocorelation) for Diffuse Correlation Spectroscopy in N Layers with Partial-Current Boundary Condition.
    Source1: "Noninvasive determination of the optical properties of two-layered turbid media"
    Source2: "Diffuse optics for tissue monitoring and tomography"
    parameters:
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
        npoints := Minimum number of points to evaluate for fft []
    """
    imp = gen_impedance(n / n_ext)
    D = _map_tuples(_D, mua, musp)
    D1 = D[0]
    z0 = 3 * D1
    # assert np.all(depths[0] >= z0)
    zb = 2 * D1 * imp
    k0 = 2 * pi * n / wavelength
    k2 = _map_tuples(lambda a, sp, d, b: _g1_k2(a, sp, d, b, k0, tau), mua, musp, D, BFi)
    k2_norm = _map_tuples(lambda a, sp, d, b: _g1_k2(a, sp, d, b, k0, tau_0), mua, musp, D, BFi)
    sp, h = _refl_fft_hankel(z0, zb, depths, D, k2, log_limit, npoints, use_gpu)
    _sp, h_norm = _refl_fft_hankel(z0, zb, depths, D, k2_norm, log_limit, npoints, use_gpu)
    g1 = h / h_norm
    return np.squeeze(sp), g1


@_deprecated
def model_nlayer_g2(
    rho, tau, mua, musp, depths, BFi, wavelength, n, n_ext=1, beta=0.5, tau_0=0, int_limit=10, int_divs=10
):
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
    g1 = model_nlayer_g1(rho, tau, mua, musp, depths, BFi, wavelength, n, n_ext, tau_0, int_limit, int_divs)
    return 1 + beta * g1**2


def model_nlayer_g2_fft(
    tau, mua, musp, depths, BFi, wavelength, n, n_ext=1, tau_0=0, log_limit=15, npoints=512, use_gpu=False
):
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
        log_limit := Log-Space Integration Limit for fft []
        npoints := Minimum number of points to evaluate for fft []
    """
    sp, g1 = model_nlayer_g1_fft(tau, mua, musp, depths, BFi, wavelength, n, n_ext, tau_0, log_limit, npoints, use_gpu)
    return sp, 1 + beta * g1**2
