from numpy import pi, exp, sqrt
from .utils import jit, gen_impedance, gen_coeffs


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
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D = 1 / (3 * (mua + musp))
    z = 0
    z0 = 1 / (mua + musp)
    zb = 2 * D * imp
    mu_eff = sqrt(3 * mua * (mua + musp))
    r1_sq = (z - z0)**2 + rho**2
    r1 = sqrt(r1_sq)
    r2_sq = (z + z0 + 2*zb)**2 + rho**2
    r2 = sqrt(r2_sq)
    flu_rate = 1/(4*pi*D)*(exp(-mu_eff*r1)/r1 - exp(-mu_eff*r2)/r2)
    diff_refl = 1/(4*pi)*(z0*(mu_eff + 1/r1)*exp(-mu_eff*r1)/r1_sq + (z0 + 2*zb)*(mu_eff + 1/r2)*exp(-mu_eff*r2)/r2_sq)
    return flu_coeff * flu_rate + refl_coeff * diff_refl


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
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D = 1 / (3 * (mua + musp))
    z = 0
    z0 = 1 / (mua + musp)
    zb = 2 * D * imp
    v = c / n
    w = 2 * pi * freq
    k_in = sqrt(1+(w/(mua*v))**2)
    k = sqrt(3 / 2 * mua * (mua + musp)) * (sqrt(k_in+1) + 1j*sqrt(k_in-1))
    r1_sq = (z - z0)**2 + rho**2
    r1 = sqrt(r1_sq)
    r2_sq = (z + z0 + 2*zb)**2 + rho**2
    r2 = sqrt(r2_sq)
    flu_rate = 1/(4*pi*D)*(exp(-k*r1)/r1 - exp(-k*r2)/r2)
    diff_refl = 1/(4*pi)*(z0*(k + 1/r1)*exp(-k*r1)/r1_sq + (z0 + 2*zb)*(k + 1/r2)*exp(-k*r2)/r2_sq)
    return flu_coeff * flu_rate + refl_coeff * diff_refl


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
    z0 = 1 / (mua + musp)
    zb = 2 * imp * D
    r1_sq = (z - z0)**2 + rho**2
    r2_sq = (z + z0 + 2*zb)**2 + rho**2
    reflectance = 0.5*t**(-5/2)*(4*pi*D*v)**(-3/2)*exp(-mua*v*t)*(z0*exp(-r1_sq/(4*D*v*t))+(z0+2*zb)*exp(-r2_sq/(4*D*v*t)))
    fluence_rate = v*(4*pi*D*v*t)**(-3/2)*exp(-mua*v*t)*(exp(-(r1_sq/(4*D*v*t)))-exp(-(r2_sq/(4*D*v*t))))
    return flu_coeff*fluence_rate + refl_coeff*reflectance


@jit
def model_g2(tau, bfi, beta, mua, musp, wavelength, rho, first_tau_delay, n, n_ext=1):
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
    """
    imp = gen_impedance(n / n_ext)
    D = 1 / (3 * (mua + musp))  # diffusion constant
    z = 0
    z0 = 1 / (mua + musp)
    zb = 2 * imp * D
    k0 = 2 * pi * n / wavelength
    k_tau = sqrt(3 * mua * musp + musp**2 * k0**2 * 6 * bfi * tau)
    k_norm = sqrt(3 * mua * musp + musp**2 * k0**2 * 6 * bfi * first_tau_delay)
    r1 = sqrt(rho**2 + (z - z0)**2)
    r2 = sqrt(rho**2 + (z + z0 + 2 * zb)**2)
    g1 = (exp(-k_tau*r1)/r1 - exp(-k_tau*r2)/r2) / (exp(-k_norm*r1)/r1 - exp(-k_norm*r2)/r2)
    return 1 + beta * g1 ** 2
