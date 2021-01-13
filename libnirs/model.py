from numpy import pi, exp, sqrt
from .utils import jit, gen_impedance, gen_coeffs


@jit
def _ecbc(k, rho, D, n, n_ext):
    """Time-Independent Reflectance with Extrapolated Boundary Condition"""
    imp, fresTj, fresTphi = gen_coeffs(n, n_ext)
    # z = 0
    z0 = 3 * D
    zb = 2 * D * imp
    # r1 = sqrt(rho ** 2 + (z - z0) ** 2)
    r1 = sqrt(rho ** 2 + z0 ** 2)
    # r2 = sqrt(rho ** 2 + (z + z0 + 2 * zb) ** 2)
    r2 = sqrt(rho ** 2 + (z0 + 2 * zb) ** 2)
    phi = (exp(-k * r1) / r1 - exp(-k * r2) / r2) / (4 * pi * D)
    j = (
        z0 * (1 + k * r1) * exp(-k * r1) / r1 ** 3
        + (z0 + 2 * zb) * (1 + k * r2) * exp(-k * r2) / r2 ** 3
    ) / (4 * pi)
    return fresTphi * phi + fresTj * j


@jit
def model_ss(rho, mua, musp, n, n_ext):
    """Model Steady-State Reflectance with Extrapolated Boundary Condition.
    Source: "Improved solutions of the steady-state and the time-resolved diffusion equations for reflectance from a semi-infinite turbid medium"
    parameters:
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
    """
    D = 1 / (3 * (mua + musp))
    mu_eff = sqrt(mua / D)
    return _ecbc(mu_eff, rho, D, n, n_ext)


@jit
def model_fd(rho, mua, musp, n, n_ext, freq, c):
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
    """
    D = 1 / (3 * (mua + musp))
    v = c / n
    omega = 2 * pi * freq
    k_in = sqrt(1 + (omega / (mua * v)) ** 2)
    k = sqrt(mua / D / 2) * (sqrt(k_in + 1) + 1j * sqrt(k_in - 1))
    return _ecbc(k, rho, D, n, n_ext)


@jit
def model_td(t, rho, mua, musp, n, n_ext, c):
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
    """
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    v = c / n  # Speed of light in turbid medium
    D = 1 / (3 * (mua + musp))  # diffusion constant
    z = 0
    z0 = 3 * D
    zb = 2 * D * imp
    r1_sq = (z - z0) ** 2 + rho ** 2
    r2_sq = (z + z0 + 2 * zb) ** 2 + rho ** 2
    reflectance = (
        0.5
        * t ** (-5 / 2)
        * (4 * pi * D * v) ** (-3 / 2)
        * exp(-mua * v * t)
        * (
            z0 * exp(-r1_sq / (4 * D * v * t))
            + (z0 + 2 * zb) * exp(-r2_sq / (4 * D * v * t))
        )
    )
    fluence_rate = (
        v
        * (4 * pi * D * v * t) ** (-3 / 2)
        * exp(-mua * v * t)
        * (exp(-(r1_sq / (4 * D * v * t))) - exp(-(r2_sq / (4 * D * v * t))))
    )
    return flu_coeff * fluence_rate + refl_coeff * reflectance


@jit
def model_g1(tau, bfi, mua, musp, wavelength, rho, first_tau_delay, n, n_ext=1):
    """Model g1 (autocorelation) for Diffuse correlation spectroscopy.
    Source: "Diffuse optics for tissue monitoring and tomography"
    parameters:
        tau := Correlation time [time]
        bfi := Blood-Flow Index [1/length^2/time]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        rho := Source-Detector Seperation [length]
        first_tau_delay := The first tau for normalization [time]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
    """
    D = 1 / (3 * (mua + musp))
    k0 = 2 * pi * n / wavelength
    k_tau = sqrt((mua + 2 * musp * k0 ** 2 * bfi * tau) / D)
    k_norm = sqrt((mua + 2 * musp * k0 ** 2 * bfi * first_tau_delay) / D)
    return _ecbc(k_tau, rho, D, n, n_ext) / _ecbc(k_norm, rho, D, n, n_ext)


@jit
def model_g2(tau, bfi, beta, mua, musp, wavelength, rho, first_tau_delay, n, n_ext=1):
    """Model g2 (autocorelation) for Diffuse correlation spectroscopy.
    Source: "Diffuse optics for tissue monitoring and tomography"
    parameters:
        tau := Correlation time [time]
        bfi := Blood-Flow Index [1/length^2/time]
        beta := Beta derived for Siegert relation []
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        wavelength := Wavelength of Light [length]
        rho := Source-Detector Seperation [length]
        first_tau_delay := The first tau for normalization [time]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
    """
    g1 = model_g1(tau, bfi, mua, musp, wavelength, rho, first_tau_delay, n, n_ext)
    return 1 + beta * g1 ** 2
