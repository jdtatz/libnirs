"""
Homogenous Modeling of Reflectance
"""

from jax.numpy import exp, pi, sqrt
from jax.scipy.special import erfc

from .fresnelx import ecbc_coeffs_exact, ecbc_coeffs_quad, impedence_exact, impedence_quad

USE_EXACT_COEFS = True


def _ecbc_coeffs(n_media, n_ext):
    if USE_EXACT_COEFS:
        return ecbc_coeffs_exact(n_media, n_ext)
    else:
        return ecbc_coeffs_quad(n_media, n_ext)


def _impedence(n_media, n_ext):
    if USE_EXACT_COEFS:
        return impedence_exact(n_media, n_ext)
    else:
        return impedence_quad(n_media, n_ext)


def ecbc_reflectance(rho, k, mua, musp, n_media, n_ext):
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
    impedence, fresTj, fresTphi = _ecbc_coeffs(n_media, n_ext)
    D = 1 / (3 * (mua + musp))  # Diffusion Constant
    # z = 0
    z0 = 3 * D
    zb = 2 * D * impedence
    # r1 = sqrt(rho ** 2 + (z - z0) ** 2)
    r1 = sqrt(rho**2 + z0**2)
    # r2 = sqrt(rho ** 2 + (z + z0 + 2 * zb) ** 2)
    r2 = sqrt(rho**2 + (z0 + 2 * zb) ** 2)
    phi = (exp(-k * r1) / r1 - exp(-k * r2) / r2) / (4 * pi * D)
    j = (z0 * (1 + k * r1) * exp(-k * r1) / r1**3 + (z0 + 2 * zb) * (1 + k * r2) * exp(-k * r2) / r2**3) / (4 * pi)
    return fresTphi * phi + fresTj * j


def pcbc_td_reflectance(t, rho, k, mua, musp, n_media, n_ext, c):
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
    impedence = _impedence(n_media, n_ext)
    v = c / n_media  # Speed of light in turbid medium
    D = 1 / (3 * (mua + musp))  # Diffusion Constant
    # z = 0
    z0 = 3 * D
    zb = 2 * D * impedence
    alpha = 4 * D * v * t
    return (
        -D
        * v
        * (
            pi
            * sqrt(alpha)
            * exp((alpha + 2 * z0 * zb) ** 2 / (4 * alpha * zb**2))
            * erfc((alpha + 2 * z0 * zb) / (2 * sqrt(alpha) * zb))
            - 2 * sqrt(pi) * zb
        )
        * exp(-k * v * t - (rho**2 + z0**2) / alpha)
        / (pi**2 * sqrt(alpha) ** 3 * zb**2)
    )


def model_ss(rho, mua, musp, n_media, n_ext):
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
    k = sqrt(mua / D)
    return ecbc_reflectance(rho, k, mua, musp, n_media, n_ext)


def model_fd(rho, mua, musp, n_media, n_ext, freq, c):
    """Model Frequncy-Domain Reflectance with Extrapolated Boundary Condition.
    Source: "Kienle, A., & Patterson, M. S. (1997). Improved solutions of the steady-state and the time-resolved diffusion equations for reflectance from a semi-infinite turbid medium. Journal of the Optical Society of America A, 14(1), 246. doi:10.1364/josaa.14.000246 "
    parameters:
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
        freq := Frequncy of Source [1/time]
        c := Speed of Light in vacuum [length/time]
    """
    D = 1 / (3 * (mua + musp))
    v = c / n_media
    omega = 2 * pi * freq
    k = sqrt((mua + 1j * (omega / v)) / D)
    # k_in = sqrt(1 + (omega / (mua * v)) ** 2)
    # _k = sqrt(mua / D / 2) * (sqrt(k_in + 1) + 1j * sqrt(k_in - 1))
    # assert _k == k
    return ecbc_reflectance(rho, k, mua, musp, n_media, n_ext)


def model_td(t, rho, mua, musp, n_media, n_ext, c):
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
    return pcbc_td_reflectance(t, rho, k, mua, musp, n_media, n_ext, c)


def model_unnorm_G1(tau, bfi, mua, musp, wavelength, rho, n_media, n_ext):
    """Model G1 (unnormalized autocorelation) for Diffuse correlation spectroscopy with Extrapolated Boundary Condition.
    Source: "Diffuse optics for tissue monitoring and tomography"
    parameters:
        tau := Correlation time [time]
        bfi := Blood-Flow Index [1/length^2/time]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        rho := Source-Detector Seperation [length]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
    """
    D = 1 / (3 * (mua + musp))
    k0 = 2 * pi * n_media / wavelength
    k = sqrt((mua + 2 * musp * k0**2 * bfi * tau) / D)
    return ecbc_reflectance(rho, k, mua, musp, n_media, n_ext)


def model_unnorm_fd_G1(tau, bfi, mua, musp, wavelength, rho, n_media, n_ext, freq, c):
    """Model G1 (unnormalized autocorelation) for Frequency Domain Diffuse correlation spectroscopy with Extrapolated Boundary Condition.
    Source: "Frequency Domain Diffuse Correlation Spectroscopy: A New Method for Simultaneous Estimation of Static and Dynamic Tissue Optical Properties"
    parameters:
        tau := Correlation time [time]
        bfi := Blood-Flow Index [1/length^2/time]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        rho := Source-Detector Seperation [length]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
        freq := Frequncy of Source [1/time]
        c := Speed of Light in vacuum [length/time]
    """
    D = 1 / (3 * (mua + musp))
    v = c / n_media
    omega = 2 * pi * freq
    k0 = 2 * pi * n_media / wavelength
    k = sqrt((mua + 2 * musp * k0**2 * bfi * tau - 1j * (omega / v)) / D)
    return abs(ecbc_reflectance(rho, k, mua, musp, n_media, n_ext))


def model_g1(tau, bfi, mua, musp, wavelength, rho, n_media, n_ext):
    """Model g1 (autocorelation) for Diffuse correlation spectroscopy with Extrapolated Boundary Condition.
    Source: "Diffuse optics for tissue monitoring and tomography"
    parameters:
        tau := Correlation time [time]
        bfi := Blood-Flow Index [1/length^2/time]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        rho := Source-Detector Seperation [length]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
    """
    D = 1 / (3 * (mua + musp))
    k0 = 2 * pi * n_media / wavelength
    k_tau = sqrt((mua + 2 * musp * k0**2 * bfi * tau) / D)
    k_norm = sqrt(mua / D)
    G1_norm = ecbc_reflectance(rho, k_norm, mua, musp, n_media, n_ext)
    return ecbc_reflectance(rho, k_tau, mua, musp, n_media, n_ext) / G1_norm


def model_g2(tau, bfi, beta, mua, musp, wavelength, rho, n_media, n_ext):
    """Model g2 (autocorelation) for Diffuse correlation spectroscopy with Extrapolated Boundary Condition.
    Source: "Diffuse optics for tissue monitoring and tomography"
    parameters:
        tau := Correlation time [time]
        bfi := Blood-Flow Index [1/length^2/time]
        beta := Beta derived for Siegert relation []
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        rho := Source-Detector Seperation [length]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
    """
    g1 = model_g1(tau, bfi, mua, musp, wavelength, rho, n_media, n_ext)
    return 1 + beta * g1**2


def model_fd_g1(tau, bfi, mua, musp, wavelength, rho, n_media, n_ext, freq, c):
    """Model g1 (autocorelation) for Frequency Domain Diffuse correlation spectroscopy with Extrapolated Boundary Condition.
    Source: "Frequency Domain Diffuse Correlation Spectroscopy: A New Method for Simultaneous Estimation of Static and Dynamic Tissue Optical Properties"
    parameters:
        tau := Correlation time [time]
        bfi := Blood-Flow Index [1/length^2/time]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        rho := Source-Detector Seperation [length]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
        freq := Frequncy of Source [1/time]
        c := Speed of Light in vacuum [length/time]
    """
    D = 1 / (3 * (mua + musp))
    v = c / n_media
    omega = 2 * pi * freq
    k0 = 2 * pi * n_media / wavelength
    k_tau = sqrt((mua + 2 * musp * k0**2 * bfi * tau - 1j * (omega / v)) / D)
    k_norm = sqrt(mua / D)
    G1_norm = ecbc_reflectance(rho, k_norm, mua, musp, n_media, n_ext)
    return abs(ecbc_reflectance(rho, k_tau, mua, musp, n_media, n_ext)) / G1_norm


def model_fd_g2_simplified(
    tau,
    bfi,
    beta,
    src_mod_depth,
    mua,
    musp,
    wavelength,
    rho,
    n_media,
    n_ext,
    freq,
    c,
):
    """Model g2 (autocorelation) for Frequency Domain Diffuse correlation spectroscopy with Extrapolated Boundary Condition.
    Source: "Frequency Domain Diffuse Correlation Spectroscopy: A New Method for Simultaneous Estimation of Static and Dynamic Tissue Optical Properties"
    parameters:
        tau := Correlation time [time]
        bfi := Blood-Flow Index [1/length^2/time]
        beta := Beta derived for Siegert relation []
        src_mod_depth := Source Modulation Depth []
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        rho := Source-Detector Seperation [length]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
        freq := Frequncy of Source [1/time]
        c := Speed of Light in vacuum [length/time]
    """
    # return 1 + beta * (1 - src_mod_depth) * model_g1(...)**2 + beta * src_mod_depth * abs(model_fd_g1(...))**2
    D = 1 / (3 * (mua + musp))
    v = c / n_media
    omega = 2 * pi * freq
    k0 = 2 * pi * n_media / wavelength
    k_ac = sqrt((mua + 2 * musp * k0**2 * bfi * tau - 1j * (omega / v)) / D)
    k_dc = sqrt((mua + 2 * musp * k0**2 * bfi * tau) / D)
    k_norm = sqrt(mua / D)
    G1_norm = ecbc_reflectance(rho, k_norm, mua, musp, n_media, n_ext)
    g1_ac = ecbc_reflectance(rho, k_ac, mua, musp, n_media, n_ext) / G1_norm
    g1_dc = ecbc_reflectance(rho, k_dc, mua, musp, n_media, n_ext) / G1_norm
    g1_ac_sq = g1_ac.real * g1_ac.real + g1_ac.imag * g1_ac.imag
    return 1 + beta * (1 - src_mod_depth) * g1_dc**2 + beta * src_mod_depth * g1_ac_sq


def model_fd_g2(
    tau,
    bfi,
    beta,
    src_mod_depth,
    mua,
    musp,
    wavelength,
    rho,
    n_media,
    n_ext,
    freq,
    c,
):
    """Model g2 (autocorelation) for Frequency Domain Diffuse correlation spectroscopy with Extrapolated Boundary Condition.
    Source: "Frequency Domain Diffuse Optics Spectroscopies for Quantitative Measurement of Tissue Optical Properties"
    parameters:
        tau := Correlation time [time]
        bfi := Blood-Flow Index [1/length^2/time]
        beta := Beta derived for Siegert relation []
        src_mod_depth := Source Modulation Depth []
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        rho := Source-Detector Seperation [length]
        n_media := Media Index of Refraction []
        n_ext := External Index of Refraction []
        freq := Frequncy of Source [1/time]
        c := Speed of Light in vacuum [length/time]
    """
    # return 1 + beta * ((model_g1(...) + src_mod_depth * abs(model_fd_g1(...))) / (1 + src_mod_depth))**2
    D = 1 / (3 * (mua + musp))
    v = c / n_media
    omega = 2 * pi * freq
    k0 = 2 * pi * n_media / wavelength
    k_ac = sqrt((mua + 2 * musp * k0**2 * bfi * tau - 1j * (omega / v)) / D)
    k_dc = sqrt((mua + 2 * musp * k0**2 * bfi * tau) / D)
    k_norm = sqrt(mua / D)
    G1_norm = ecbc_reflectance(rho, k_norm, mua, musp, n_media, n_ext)
    g1_ac_c = ecbc_reflectance(rho, k_ac, mua, musp, n_media, n_ext) / G1_norm
    g1_dc = ecbc_reflectance(rho, k_dc, mua, musp, n_media, n_ext) / G1_norm
    # g1_ac = hypot(g1_ac_c.real, g1_ac_c.imag)
    g1_ac = abs(g1_ac_c)
    return 1 + beta * ((g1_dc + src_mod_depth * g1_ac) / (1 + src_mod_depth)) ** 2
