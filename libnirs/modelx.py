"""
Homogenous Modeling of Reflectance
"""

from math import pi

import jax.numpy as jnp
from numpy import iscomplexobj
from scipy.special import erfcx as _sc_erfcx

from .fresnelx import ecbc_coeffs_exact, ecbc_coeffs_quad, impedence_exact, impedence_quad

try:
    from jax.scipy.special import erfcx as _jax_erfcx
except ImportError:
    from .special import erfcx as _jax_erfcx


USE_EXACT_COEFS = True


# TODO: replace entirely with scipy's erfcx once it fully supports the array api?
def erfcx(x, *, xp=jnp):
    if xp is jnp:
        return _jax_erfcx(x)
    else:
        return _sc_erfcx(x)


def cabs(x, *, xp):
    """Return `abs(x) if iscomplexobj(x) else x`."""
    return abs(x) if iscomplexobj(x) else x


def abs_square(x, *, xp):
    """Return `xp.square(abs(x))`."""
    if iscomplexobj(x):
        return xp.square(xp.real(x)) + xp.square(xp.imag(x))
    else:
        return xp.square(x)


def _ecbc_coeffs(n_media, n_ext, *, xp):
    if USE_EXACT_COEFS:
        return ecbc_coeffs_exact(n_media, n_ext, xp=xp)
    else:
        return ecbc_coeffs_quad(n_media, n_ext, xp=xp)


def _impedence(n_media, n_ext, *, xp):
    if USE_EXACT_COEFS:
        return impedence_exact(n_media, n_ext, xp=xp)
    else:
        return impedence_quad(n_media, n_ext, xp=xp)


def ecbc_reflectance(rho, k, mua, musp, n_media, n_ext, *, xp=jnp):
    """Time-Independent Reflectance with Extrapolated Boundary Condition
    parameters:
        rho := Source-Detector Seperation [length]
        k := pseudo-attenuation coefficient [1/length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
    Returns:
        Reflectance [1/length^2]
    """
    impedence, fresTj, fresTphi = _ecbc_coeffs(n_media, n_ext, xp=xp)
    mu_t = mua + musp  # linear transport coefficient
    ltr = 1 / mu_t  # transport mean free path
    D = 1 / (3 * mu_t)  # Diffusion Constant
    # z = 0
    z0 = ltr
    zb = 2 * D * impedence
    # r1 = xp.sqrt(rho ** 2 + (z - z0) ** 2)
    r1 = xp.hypot(rho, z0)
    # r2 = xp.sqrt(rho ** 2 + (z + z0 + 2 * zb) ** 2)
    r2 = xp.hypot(rho, z0 + 2 * zb)
    phi = (xp.exp(-k * r1) / r1 - xp.exp(-k * r2) / r2) / (4 * pi * D)
    j = (z0 * (1 + k * r1) * xp.exp(-k * r1) / r1**3 + (z0 + 2 * zb) * (1 + k * r2) * xp.exp(-k * r2) / r2**3) / (
        4 * pi
    )
    return fresTphi * phi + fresTj * j


def pcbc_td_reflectance(t, rho, k, mua, musp, n_media, n_ext, c, *, xp=jnp):
    """Time-Domain Reflectance with Partial-Current Boundary Condition
    parameters:
        t := Time of Flight [time]
        rho := Source-Detector Seperation [length]
        k := pseudo-attenuation coefficient [1/length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
        c := Speed of Light in vacuum [length/time]
    Returns:
        Time-Domain Reflectance [1/length^2/time]
    """
    impedence = _impedence(n_media, n_ext, xp=xp)
    v = c / n_media  # Speed of light in turbid medium
    mu_t = mua + musp  # linear transport coefficient
    ltr = 1 / mu_t  # transport mean free path
    D = 1 / (3 * mu_t)  # Diffusion Constant
    # z = 0
    z0 = ltr
    zb = 2 * D * impedence
    alpha = 4 * D * v * t
    return (
        D
        * v
        * (2 * xp.sqrt(pi) * zb - pi * xp.sqrt(alpha) * erfcx((alpha + 2 * z0 * zb) / (2 * xp.sqrt(alpha) * zb), xp=xp))
        * xp.exp(-k * v * t - (rho**2 + z0**2) / alpha)
        / (pi**2 * xp.sqrt(alpha) ** 3 * zb**2)
    )


def model_ss(rho, mua, musp, n_media, n_ext, *, xp=jnp):
    """Model Steady-State Reflectance with Extrapolated Boundary Condition.
    Source: "Kienle, A., & Patterson, M. S. (1997). Improved solutions of the steady-state and the time-resolved diffusion equations for reflectance from a semi-infinite turbid medium. Journal of the Optical Society of America A, 14(1), 246. doi:10.1364/josaa.14.000246 "
    parameters:
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
    Returns:
        Steady-State Reflectance [1/length^2]
    """
    D = 1 / (3 * (mua + musp))
    k = xp.sqrt(mua / D)
    return ecbc_reflectance(rho, k, mua, musp, n_media, n_ext, xp=xp)


def model_fd(rho, mua, musp, spatial_freq, n_media, n_ext, *, xp=jnp):
    """Model Frequncy-Domain Reflectance with Extrapolated Boundary Condition.
    Source: "Kienle, A., & Patterson, M. S. (1997). Improved solutions of the steady-state and the time-resolved diffusion equations for reflectance from a semi-infinite turbid medium. Journal of the Optical Society of America A, 14(1), 246. doi:10.1364/josaa.14.000246 "
    parameters:
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        spatial_freq := Spatial Frequncy of the Source in vacuum (`ξ = freq / c`) [1/length]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
    """
    D = 1 / (3 * (mua + musp))
    src_k = 2 * pi * n_media * spatial_freq
    k = xp.sqrt((mua + 1j * src_k) / D)
    # k_in = xp.sqrt(1 + (omega / (mua * v)) ** 2)
    # _k = xp.sqrt(mua / D / 2) * (xp.sqrt(k_in + 1) + 1j * xp.sqrt(k_in - 1))
    # assert _k == k
    return ecbc_reflectance(rho, k, mua, musp, n_media, n_ext, xp=xp)


def model_td(t, rho, mua, musp, n_media, n_ext, c, *, xp=jnp):
    """Model Time-Domain Reflectance with Partial-Current Boundary Condition.
    Source: "Kienle, A., & Patterson, M. S. (1997). Improved solutions of the steady-state and the time-resolved diffusion equations for reflectance from a semi-infinite turbid medium. Journal of the Optical Society of America A, 14(1), 246. doi:10.1364/josaa.14.000246 "
    parameters:
        t := Time of Flight [time]
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
        c := Speed of Light in vacuum [length/time]
    Returns:
        Time-Domain Reflectance [1/length^2/time]
    """
    k = mua
    return pcbc_td_reflectance(t, rho, k, mua, musp, n_media, n_ext, c, xp=xp)


def model_g1_unnorm(rho, mua, musp, wavelength, bfi, tau, n_media, n_ext, *, xp=jnp):
    """Model G1 (unnormalized electric field autocorrelation) for Diffuse correlation spectroscopy with Extrapolated Boundary Condition.
    Source: "Diffuse optics for tissue monitoring and tomography"
    parameters:
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        bfi := Blood-Flow Index [length^2/time]
        tau := Correlation time [time]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
    """
    D = 1 / (3 * (mua + musp))
    k0 = 2 * pi * n_media / wavelength
    k = xp.sqrt((mua + 2 * musp * k0**2 * bfi * tau) / D)
    return ecbc_reflectance(rho, k, mua, musp, n_media, n_ext, xp=xp)


def model_fd_g1_unnorm(rho, mua, musp, wavelength, bfi, tau, spatial_freq, n_media, n_ext, *, xp=jnp):
    """Model G1 (unnormalized electric field autocorrelation) for Frequency Domain Diffuse correlation spectroscopy with Extrapolated Boundary Condition.
    Source: "Frequency Domain Diffuse Correlation Spectroscopy: A New Method for Simultaneous Estimation of Static and Dynamic Tissue Optical Properties"
    parameters:
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        bfi := Blood-Flow Index [length^2/time]
        tau := Correlation time [time]
        spatial_freq := Spatial Frequncy of the Source in vacuum (`ξ = freq / c`) [1/length]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
    """
    D = 1 / (3 * (mua + musp))
    src_k = 2 * pi * n_media * spatial_freq
    k0 = 2 * pi * n_media / wavelength
    k = xp.sqrt((mua + 2 * musp * k0**2 * bfi * tau - 1j * src_k) / D)
    return abs(ecbc_reflectance(rho, k, mua, musp, n_media, n_ext, xp=xp))


def model_g1_norm(rho, mua, musp, wavelength, bfi, tau, n_media, n_ext, *, xp=jnp):
    """Model g1 (normalized electric field autocorrelation) for Diffuse correlation spectroscopy with Extrapolated Boundary Condition.
    Source: "Diffuse optics for tissue monitoring and tomography"
    parameters:
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        bfi := Blood-Flow Index [length^2/time]
        tau := Correlation time [time]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
    """
    D = 1 / (3 * (mua + musp))
    k0 = 2 * pi * n_media / wavelength
    k_tau = xp.sqrt((mua + 2 * musp * k0**2 * bfi * tau) / D)
    k_norm = xp.sqrt(mua / D)
    G1_norm = ecbc_reflectance(rho, k_norm, mua, musp, n_media, n_ext, xp=xp)
    return ecbc_reflectance(rho, k_tau, mua, musp, n_media, n_ext, xp=xp) / G1_norm


def g2_from_g1(g1, beta, *, xp=jnp):
    """Compute g2 (normalized intensity autocorrelation) using the Siegert relation
    parameters:
        g1 := normalized electric field autocorrelation []
        beta := Beta derived for Siegert relation []
    """
    return 1 + beta * abs_square(g1, xp=xp)


def model_g2(rho, mua, musp, wavelength, bfi, tau, beta, n_media, n_ext, *, xp=jnp):
    """Model g2 (normalized intensity autocorrelation) for Diffuse correlation spectroscopy with Extrapolated Boundary Condition.
    Source: "Diffuse optics for tissue monitoring and tomography"
    parameters:
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        bfi := Blood-Flow Index [length^2/time]
        tau := Correlation time [time]
        beta := Beta derived for Siegert relation []
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
    """
    g1 = model_g1_norm(rho, mua, musp, wavelength, bfi, tau, n_media, n_ext, xp=xp)
    return g2_from_g1(g1, beta, xp=xp)


def model_fd_g1_norm(rho, mua, musp, wavelength, bfi, tau, spatial_freq, n_media, n_ext, *, xp=jnp):
    """Model g1 (normalized electric field autocorrelation) for Frequency Domain Diffuse correlation spectroscopy with Extrapolated Boundary Condition.
    Source: "Frequency Domain Diffuse Correlation Spectroscopy: A New Method for Simultaneous Estimation of Static and Dynamic Tissue Optical Properties"
    parameters:
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        bfi := Blood-Flow Index [length^2/time]
        tau := Correlation time [time]
        spatial_freq := Spatial Frequncy of the Source in vacuum (`ξ = freq / c`) [1/length]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
    """
    D = 1 / (3 * (mua + musp))
    src_k = 2 * pi * n_media * spatial_freq
    k0 = 2 * pi * n_media / wavelength
    k_tau = xp.sqrt((mua + 2 * musp * k0**2 * bfi * tau - 1j * src_k) / D)
    k_norm = xp.sqrt(mua / D)
    G1_norm = ecbc_reflectance(rho, k_norm, mua, musp, n_media, n_ext, xp=xp)
    return abs(ecbc_reflectance(rho, k_tau, mua, musp, n_media, n_ext, xp=xp)) / G1_norm


def simplified_fd_g2_from_g1(g1_dc, g1_ac, beta, mod_depth, *, xp=jnp):
    """Compute Frequency Domain g2 (normalized intensity autocorrelation) using the extended Siegert relation
    Source: "Frequency Domain Diffuse Correlation Spectroscopy: A New Method for Simultaneous Estimation of Static and Dynamic Tissue Optical Properties"
    parameters:
        g1_dc := Steady State normalized electric field autocorrelation []
        g1_ac := Frequency Domain normalized electric field autocorrelation []
        beta := Beta derived for Siegert relation []
        mod_depth := Source Modulation Depth []
    """
    return 1 + beta * (1 - mod_depth) * g1_dc**2 + beta * mod_depth * abs_square(g1_ac, xp=xp)


def expanded_fd_g2_from_g1(g1_dc, g1_ac, beta, mod_depth, *, xp=jnp):
    """Compute Frequency Domain g2 (normalized intensity autocorrelation) using the expanded intensity autocorrelation form
    Source: "Frequency Domain Diffuse Optics Spectroscopies for Quantitative Measurement of Tissue Optical Properties"
    parameters:
        g1_dc := Steady State normalized electric field autocorrelation []
        g1_ac := Frequency Domain normalized electric field autocorrelation []
        beta := Beta derived for Siegert relation []
        mod_depth := Source Modulation Depth []
    """
    return 1 + beta * ((g1_dc + mod_depth * cabs(g1_ac, xp=xp)) / (1 + mod_depth)) ** 2


def model_fd_g2_simplified(
    rho,
    mua,
    musp,
    wavelength,
    bfi,
    tau,
    spatial_freq,
    beta,
    mod_depth,
    n_media,
    n_ext,
    *,
    xp=jnp,
):
    """Model g2 (normalized intensity autocorrelation) for Frequency Domain Diffuse correlation spectroscopy with Extrapolated Boundary Condition.
    Source: "Frequency Domain Diffuse Correlation Spectroscopy: A New Method for Simultaneous Estimation of Static and Dynamic Tissue Optical Properties"
    parameters:
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        bfi := Blood-Flow Index [length^2/time]
        tau := Correlation time [time]
        spatial_freq := Spatial Frequncy of the Source in vacuum (`ξ = freq / c`) [1/length]
        beta := Beta derived for Siegert relation []
        mod_depth := Source Modulation Depth []
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
    """
    # return 1 + beta * (1 - mod_depth) * model_g1(...)**2 + beta * mod_depth * abs_square(model_fd_g1(...))
    D = 1 / (3 * (mua + musp))
    src_k = 2 * pi * n_media * spatial_freq
    k0 = 2 * pi * n_media / wavelength
    k_ac = xp.sqrt((mua + 2 * musp * k0**2 * bfi * tau - 1j * src_k) / D)
    k_dc = xp.sqrt((mua + 2 * musp * k0**2 * bfi * tau) / D)
    k_norm = xp.sqrt(mua / D)
    G1_norm = ecbc_reflectance(rho, k_norm, mua, musp, n_media, n_ext, xp=xp)
    g1_ac = ecbc_reflectance(rho, k_ac, mua, musp, n_media, n_ext, xp=xp) / G1_norm
    g1_dc = ecbc_reflectance(rho, k_dc, mua, musp, n_media, n_ext, xp=xp) / G1_norm
    return simplified_fd_g2_from_g1(g1_dc, g1_ac, beta, mod_depth, xp=xp)


def model_fd_g2(
    rho,
    mua,
    musp,
    wavelength,
    bfi,
    tau,
    spatial_freq,
    beta,
    mod_depth,
    n_media,
    n_ext,
    *,
    xp=jnp,
):
    """Model g2 (normalized intensity autocorrelation) for Frequency Domain Diffuse correlation spectroscopy with Extrapolated Boundary Condition.
    Source: "Frequency Domain Diffuse Optics Spectroscopies for Quantitative Measurement of Tissue Optical Properties"
    parameters:
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        bfi := Blood-Flow Index [length^2/time]
        tau := Correlation time [time]
        spatial_freq := Spatial Frequncy of the Source in vacuum (`ξ = freq / c`) [1/length]
        beta := Beta derived for Siegert relation []
        mod_depth := Source Modulation Depth []
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
    """
    # return 1 + beta * ((model_g1(...) + mod_depth * abs(model_fd_g1(...))) / (1 + mod_depth))**2
    D = 1 / (3 * (mua + musp))
    src_k = 2 * pi * n_media * spatial_freq
    k0 = 2 * pi * n_media / wavelength
    k_ac = xp.sqrt((mua + 2 * musp * k0**2 * bfi * tau - 1j * src_k) / D)
    k_dc = xp.sqrt((mua + 2 * musp * k0**2 * bfi * tau) / D)
    k_norm = xp.sqrt(mua / D)
    G1_norm = ecbc_reflectance(rho, k_norm, mua, musp, n_media, n_ext, xp=xp)
    g1_ac = ecbc_reflectance(rho, k_ac, mua, musp, n_media, n_ext, xp=xp) / G1_norm
    g1_dc = ecbc_reflectance(rho, k_dc, mua, musp, n_media, n_ext, xp=xp) / G1_norm
    return expanded_fd_g2_from_g1(g1_dc, g1_ac, beta, mod_depth, xp=xp)
