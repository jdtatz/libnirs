from numpy import pi, exp, sqrt, sinh, cosh
from scipy.special import j0
from .utils import jit, integrate, gen_coeffs
from .model import model_ss, model_fd, model_g2

"""
WARNING
The D and alpha values are not precomputed due to the fact that the
number of layers isn't known during numba compilation, and the dynamic
memory allocation needed for storing the precumputed values isn't allowed
for the cuda target.
"""

@jit
def _n_layer_refl(s, z, z0, zb, ls, muas, musps, alphas, alpha_args, flu_coeff, refl_coeff):
    n = len(muas)
    
    def ds(i):
        return 1 / (3 * (muas[i] + musps[i]))
    
    d2 = ds(n-1)
    d1 = ds(n-2)
    alpha2 = alphas(s, n-1, d2, *alpha_args)
    alpha1 = alphas(s, n-2, d1, *alpha_args)

    if n == 2:
        l1 = ls[0]
        coeff_n = d1 * alpha1 * cosh(alpha1 * (l1 - z0)) + d2 * alpha2 * sinh(alpha1 * (l1 - z0))
        coeff_d = d1 * (d1 * alpha1 * cosh(alpha1 * (l1 + zb)) + d2 * alpha2 * sinh(alpha1 * (l1 + zb)))
        coeff = coeff_n / coeff_d
        phi = coeff * sinh(alpha1 * (z + zb)) / alpha1
        dz_phi = coeff * cosh(alpha1 * (z + zb))
        return flu_coeff * phi + refl_coeff * d1 * dz_phi
    
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
    coeff = (
        exp(zb*alpha1) * (
            alpha1 * d1 * (exp(2 * l1 *alpha2) * k_num + k_dem) *
            cosh(alpha1*(l1-z0)) +
            alpha2 * d2 * (-exp(2 * l1 *alpha2) * k_num + k_dem) *
            sinh(alpha1*(l1-z0))
        ) / (
            2 * alpha1 * d1 * (
                alpha1 * d1 * (exp(2 * l1 *alpha2) * k_num + k_dem) *
                cosh(alpha1*(l1+zb)) +
                alpha2 * d2 * (-exp(2 * l1 *alpha2) * k_num + k_dem) *
                sinh(alpha1*(l1+zb))
            )
        )
    )

    phi = 2 * coeff * exp(-alpha1 * zb) * sinh(alpha1 * (zb + z))
    dz_phi = 2 * alpha1 * coeff * exp(-alpha1 * zb) * cosh(alpha1 * (zb + z))
    return flu_coeff * phi + refl_coeff * d1 * dz_phi


@jit
def _refl_integrator(s, z, rho, z0, zb, ls, muas, musps, alphas, alpha_args, flu_coeff, refl_coeff):
    return s*j0(s*rho)*_n_layer_refl(s, z, z0, zb, ls, muas, musps, alphas, alpha_args, flu_coeff, refl_coeff)


@jit
def _ss_alphas(s, i, d, muas):
    return sqrt(s**2 + muas[i] / d)


@jit
def model_nlayer_ss(rho, mua, musp, depths, n, n_ext=1, int_limit=10, int_divs=10):
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
    """
    nlayer = len(mua)
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1 / (3 * (mua[0] + musp[0]))
    z0 = 3*D1
    zb = 2*D1*imp
    alpha_args = (mua,)
    return integrate(_refl_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, mua, musp, _ss_alphas, alpha_args, flu_coeff, refl_coeff)) / (2*pi)


@jit
def _fd_alphas(s, i, d, wave, muas):
    return sqrt(s**2 + (wave + muas[i]) / d)


@jit
def model_nlayer_fd(rho, mua, musp, depths, freq, c, n, n_ext=1, int_limit=10, int_divs=10):
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
    """
    nlayer = len(mua)
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1 / (3 * (mua[0] + musp[0]))
    z0 = 3*D1
    zb = 2*D1*imp
    w = 2*pi*freq
    v = c / n
    wave = w/v*1j
    alpha_args = wave, mua
    return integrate(_refl_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, mua, musp, _fd_alphas, alpha_args, flu_coeff, refl_coeff)) / (2*pi)


@jit
def _g2_alphas(s, i, d, muas, musps, k0, BFi, tau):
    return sqrt(s**2 + (muas[i] + 2 * musps[i] * k0**2 * BFi[i] * tau) / d)


@jit
def model_nlayer_g2(rho, tau, mua, musp, depths, BFi, wavelength, n, n_ext=1, beta=0.5, tau_0=0, int_limit=10, int_divs=10):
    """Model g2 (autocorelation) for Diffuse Correlation Spectroscopy in N Layers with Extrapolated Boundary Condition.
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
    nlayer = len(mua)
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1 / (3 * (mua[0] + musp[0]))
    z0 = 3*D1
    zb = 2*D1*imp
    k0 = 2 * pi * n / wavelength
    alpha_args = mua, musp, k0, BFi, tau
    alpha_norm_args = mua, musp, k0, BFi, tau_0
    refl = integrate(_refl_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, mua, musp, _g2_alphas, alpha_args, flu_coeff, refl_coeff))
    refl_norm = integrate(_refl_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, mua, musp, _g2_alphas, alpha_norm_args, flu_coeff, refl_coeff))
    g1 = refl / refl_norm
    return 1 + beta * g1 ** 2
