"""
Homogenous Modeling of Reflectance
"""
from numpy import pi, exp, sqrt
from scipy.special import erfc
from .utils import jit, gen_impedance, gen_coeffs
from numba import vectorize

try:
    import numba_scipy
except ImportError:
    from numba.extending import get_cython_function_address, overload

    @overload(erfc)
    def _erfc(x):
        import ctypes

        addr = get_cython_function_address("scipy.special.cython_special", "__pyx_fuse_1erfc")
        functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
        erfc_impl_inner = functype(addr)

        def erfc_impl(x):
            return erfc_impl_inner(x)

        return erfc_impl


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


def _model_ss(rho, mua, musp, n, n_ext):
    """Model Steady-State Reflectance with Extrapolated Boundary Condition.
    Source: "Improved solutions of the steady-state and the time-resolved diffusion equations for reflectance from a semi-infinite turbid medium"
    parameters:
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
    Returns:
        Steady-State Reflectance [1/length^2]
    """
    D = 1 / (3 * (mua + musp))
    mu_eff = sqrt(mua / D)
    return _ecbc(mu_eff, rho, D, n, n_ext)
_model_ss_sig = ()  # ["f4(f4, f4, f4, f4, f4)", "f8(f8, f8, f8, f8, f8)"]
model_ss = vectorize(_model_ss_sig, target="cpu")(_model_ss)


def _model_fd(rho, mua, musp, n, n_ext, freq, c):
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
_model_fd_sig = ()  # ["c8(f4, f4, f4, f4, f4, f4, f4)", "c16(f8, f8, f8, f8, f8, f8, f8)"]
model_fd = vectorize(_model_fd_sig, target="cpu")(_model_fd)



def _model_td(t, rho, mua, musp, n, n_ext, c):
    """Model Time-Domain Reflectance with Partial-Current Boundary Condition.
    Source: "Improved solutions of the steady-state and the time-resolved diffusion equations for reflectance from a semi-infinite turbid medium"
    parameters:
        t := Time of Flight [time]
        rho := Source-Detector Seperation [length]
        mua := Absorption Coefficent [1/length]
        musp := Reduced Scattering Coefficent [1/length]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        c := Speed of Light in vacuum [length/time]
    Returns:
        Time-Domain Reflectance [1/length^2/time]
    """
    imp = gen_impedance(n / n_ext)
    v = c / n  # Speed of light in turbid medium
    D = 1 / (3 * (mua + musp))  # diffusion constant
    z = 0
    z_0 = 3 * D
    z_b = 2 * D * imp

    alpha = 4 * D * v * t
    beta = mua
    return (
        -D
        * v
        * (
            pi
            * sqrt(alpha)
            * exp((alpha + 2 * z_0 * z_b) ** 2 / (4 * alpha * z_b ** 2))
            * erfc((alpha + 2 * z_0 * z_b) / (2 * sqrt(alpha) * z_b))
            - 2 * sqrt(pi) * z_b
        )
        * exp(-beta * v * t - (rho ** 2 + z_0 ** 2) / alpha)
        / (pi ** 2 * alpha ** (3 / 2) * z_b ** 2)
    )
_model_td_sig = ()  # ["f4(f4, f4, f4, f4, f4, f4, f4)", "f8(f8, f8, f8, f8, f8, f8, f8)"]
model_td = vectorize(_model_td_sig, target="cpu")(_model_td)


def _model_g1(tau, bfi, mua, musp, wavelength, rho, first_tau_delay, n, n_ext):
    """Model g1 (autocorelation) for Diffuse correlation spectroscopy with Extrapolated Boundary Condition.
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
_model_g1_sig = ()  # ["f4(f4, f4, f4, f4, f4, f4, f4, f4, f4)", "f8(f8, f8, f8, f8, f8, f8, f8, f8, f8)"]
model_g1 = vectorize(_model_g1_sig, target="cpu")(_model_g1)
_j_model_g1 = jit(_model_g1)


def _model_g2(tau, bfi, beta, mua, musp, wavelength, rho, first_tau_delay, n, n_ext):
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
        first_tau_delay := The first tau for normalization [time]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
    """
    g1 = _j_model_g1(tau, bfi, mua, musp, wavelength, rho, first_tau_delay, n, n_ext)
    return 1 + beta * g1 ** 2
_model_g2_sig = ()  # ["f4(f4, f4, f4, f4, f4, f4, f4, f4, f4, f4)", "f8(f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)"]
model_g2 = vectorize(_model_g2_sig, target="cpu")(_model_g2)


def _model_fd_g1(tau, bfi, mua, musp, wavelength, rho, first_tau_delay, n, n_ext, freq, c):
    """Model g1 (autocorelation) for Frequency Domain Diffuse correlation spectroscopy with Extrapolated Boundary Condition.
    Source: "Frequency Domain Diffuse Correlation Spectroscopy: A New Method for Simultaneous Estimation of Static and Dynamic Tissue Optical Properties"
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
        freq := Frequncy of Source [1/time]
        c := Speed of Light in vacuum [length/time]
    """
    D = 1 / (3 * (mua + musp))
    v = c / n
    omega = 2 * pi * freq
    k0 = 2 * pi * n / wavelength
    k_tau = sqrt((mua + 2 * musp * k0**2 * bfi * tau - 1j * (omega / v)) / D)
    k_norm = sqrt((mua + 2 * musp * k0**2 * bfi * first_tau_delay - 0j) / D)
    return abs(_ecbc(k_tau, rho, D, n, n_ext) / _ecbc(k_norm, rho, D, n, n_ext))


model_fd_g1 = vectorize((), target="cpu")(_model_fd_g1)
# _j_model_fd_g1 = jit(_model_fd_g1)


def _model_fd_g2(
    tau,
    bfi,
    beta,
    src_mod_depth,
    mua,
    musp,
    wavelength,
    rho,
    first_tau_delay,
    n,
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
        first_tau_delay := The first tau for normalization [time]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        freq := Frequncy of Source [1/time]
        c := Speed of Light in vacuum [length/time]
    """
    # return 1 + beta * (1 - src_mod_depth) * _j_model_g1(...)**2 + beta * src_mod_depth * abs(_j_model_fd_g1(...))**2
    D = 1 / (3 * (mua + musp))
    v = c / n
    omega = 2 * pi * freq
    k0 = 2 * pi * n / wavelength
    k_tau = sqrt((mua + 2 * musp * k0**2 * bfi * tau - 1j * (omega / v)) / D)
    k_tau_no_w = sqrt((mua + 2 * musp * k0**2 * bfi * tau) / D)
    k_norm = sqrt((mua + 2 * musp * k0**2 * bfi * first_tau_delay) / D)
    G1_norm = _ecbc(k_norm, rho, D, n, n_ext)
    g1 = _ecbc(k_tau, rho, D, n, n_ext) / G1_norm
    g1_no_w = _ecbc(k_tau_no_w, rho, D, n, n_ext) / G1_norm
    g1_sq = g1.real * g1.real + g1.imag * g1.imag
    return 1 + beta * (1 - src_mod_depth) * g1_no_w**2 + beta * src_mod_depth * g1_sq


model_fd_g2 = vectorize((), target="cpu")(_model_fd_g2)
