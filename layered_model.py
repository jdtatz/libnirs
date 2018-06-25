import numpy as np
from .utils import jit, vectorize, integrate, gen_coeffs
from .overrides import pi, exp, sqrt, sin, cos, sinh, cosh, arcsin, j0


def gen_mathmatica_code(layer_count):
    """Generates Mathmatica code to generate python code for n>1 layer models"""
    initals = ", ".join('D[p%d[z, s], {z, 2}] - a%d^2 p%d[z, s] == 0' % (i, i, i) for i in range(2, layer_count+1))
    boundrys = ", ".join('p{0}[l{0}, s] == p{1}[l{0}, s], D{0} (D[p{0}[z, s], z] /. z -> l{0}) == D{1} (D[p{1}[z, s], z] /. z -> l{0})'.format(i, i+1) for i in range(1, layer_count))
    funcs = ", ".join("p{}[z, s]".format(i+1) for i in range(layer_count))
    assumptions = " && ".join("a{0} > 0 && D{0} > 0".format(i+1) for i in range(layer_count))
    layers = " > ".join("l{}".format(i) for i in range(layer_count - 1, 0, -1))
    code = """
solvs = DSolve[{D[p1[z, s], {z, 2}] - a1^2 p1[z, s] == -DiracDelta[z - z0]/D1, %s, p1[-zb, s] == 0, p%d[Infinity, s] == 0, %s }, {%s}, {z, s}, Assumptions -> %s && %s > z0 > 0 && zb > 0];
soln = FullSimplify[First[p1[z, s] /. solvs]]
pw = PageWidth /. Options[$Output];
SetOptions[$Output, PageWidth -> Infinity];
FortranForm[Simplify[soln, z < z0] //. {Exp[x_] :> exp[x], Sinh[x_] :> sinh[x], Cosh[x_] :> cosh[x]}]
SetOptions[$Output, PageWidth -> pw];""" % (initals, layer_count, boundrys, funcs, assumptions, layers)
    return code


@jit
def _2_layer_phi(s, z, z0, zb, depths, D, alpha_args):
    l = depths[0]
    D1, D2 = D[0], D[1]
    a1, a2 = sqrt(s**2 + alpha_args[0]), sqrt(s**2 + alpha_args[1])
    return (2*exp(a1*(l + zb))*(a1*D1*cosh(a1*(l - z0)) + a2*D2*sinh(a1*(l - z0)))*sinh(a1*(z + zb)))/(a1*D1*(a2*D2*(-1 + exp(2*a1*(l + zb))) + a1*D1*(1 + exp(2*a1*(l + zb)))))


@jit
def _3_layer_phi(s, z, z0, zb, depths, D, alpha_args):
    l1, l2 = depths[0], depths[1]
    D1, D2, D3 = D[0], D[1], D[2]
    a1, a2, a3 = sqrt(s**2 + alpha_args[0]), sqrt(s**2 + alpha_args[1]), sqrt(s**2 + alpha_args[2])
    return ((((a1*D1*cosh(a1*(l1 - z0))*(a2*D2*cosh(a2*(l1 - l2)) - a3*D3*sinh(a2*(l1 - l2))) + a2*D2*(a3*D3*cosh(a2*(l1 - l2)) - a2*D2*sinh(a2*(l1 - l2)))*sinh(a1*(l1 - z0)))*sinh(a1*(z + zb)))/(-(sinh(a2*(l1 - l2))*(a1*a3*D1*D3*cosh(a1*(l1 + zb)) + a2**2*D2**2*sinh(a1*(l1 + zb)))) + a2*D2*cosh(a2*(l1 - l2))*(a1*D1*cosh(a1*(l1 + zb)) + a3*D3*sinh(a1*(l1 + zb)))))/(a1*D1))


@jit
def _4_layer_phi(s, z, z0, zb, depths, D, alpha_args):
    l1, l2, l3 = depths[0], depths[1], depths[2]
    D1, D2, D3, D4 = D[0], D[1], D[2], D[3]
    a1, a2, a3, a4 = sqrt(s**2 + alpha_args[0]), sqrt(s**2 + alpha_args[1]), sqrt(s**2 + alpha_args[2]), sqrt(s**2 + alpha_args[3])
    return (exp(-(a1*(z + z0)))*(-1 + exp(2*a1*(z + zb)))*(a1*D1*(exp(2*a1*l1) + exp(2*a1*z0))*(a3*D3*sinh(a2*(l1 - l2))*(-(a4*D4*cosh(a3*(l2 - l3))) + a3*D3*sinh(a3*(l2 - l3))) + a2*D2*cosh(a2*(l1 - l2))*(a3*D3*cosh(a3*(l2 - l3)) - a4*D4*sinh(a3*(l2 - l3)))) + a2*D2*(exp(2*a1*l1) - exp(2*a1*z0))*(a3*D3*cosh(a2*(l1 - l2))*(a4*D4*cosh(a3*(l2 - l3)) - a3*D3*sinh(a3*(l2 - l3))) + a2*D2*sinh(a2*(l1 - l2))*(-(a3*D3*cosh(a3*(l2 - l3))) + a4*D4*sinh(a3*(l2 - l3))))))/(2.*a1*D1*(a1*D1*(1 + exp(2*a1*(l1 + zb)))*(a3*D3*sinh(a2*(l1 - l2))*(-(a4*D4*cosh(a3*(l2 - l3))) + a3*D3*sinh(a3*(l2 - l3))) + a2*D2*cosh(a2*(l1 - l2))*(a3*D3*cosh(a3*(l2 - l3)) - a4*D4*sinh(a3*(l2 - l3)))) + a2*D2*(-1 + exp(2*a1*(l1 + zb)))*(a3*D3*cosh(a2*(l1 - l2))*(a4*D4*cosh(a3*(l2 - l3)) - a3*D3*sinh(a3*(l2 - l3))) + a2*D2*sinh(a2*(l1 - l2))*(-(a3*D3*cosh(a3*(l2 - l3))) + a4*D4*sinh(a3*(l2 - l3))))))


@jit
def _phi_integrator(s, z, rho, z0, zb, ls, Ds, phi_func, alpha_args):
    return s*j0(s*rho)*phi_func(s, z, z0, zb, ls, Ds, alpha_args)


@jit
def model_nlayer_ss(rho, mua, musp, depths, n, n_ext=1, int_limit=10, int_divs=10, eps=1e-16):
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
    """
    nlayer = len(mua)
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1 / (3 * (mua[0] + musp[0]))
    z0 = 3*D1
    zb = 2*D1*imp
    if nlayer == 2:
        D2 = D1, 1 / (3 * (mua[1] + musp[1]))
        alpha_args2 = mua[0] / D1, mua[1] / D2[1]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D2, _2_layer_phi, alpha_args2)) / (2*pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D2, _2_layer_phi, alpha_args2)) / (2*pi)
    elif nlayer == 3:
        D3 = D1, 1 / (3 * (mua[1] + musp[1])), 1 / (3 * (mua[2] + musp[2]))
        alpha_args3 = mua[0] / D1, mua[1] / D3[1], mua[2] / D3[2]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D3, _3_layer_phi, alpha_args3)) / (2*pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D3, _3_layer_phi, alpha_args3)) / (2*pi)
    elif nlayer == 4:
        D4 = D1, 1 / (3 * (mua[1] + musp[1])), 1 / (3 * (mua[2] + musp[2])), 1 / (3 * (mua[3] + musp[3]))
        alpha_args4 = mua[0] / D1, mua[1] / D4[1], mua[2] / D4[2], mua[3] / D4[3]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D4, _4_layer_phi, alpha_args4)) / (2*pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D4, _4_layer_phi, alpha_args4)) / (2*pi)
    dp = (p_eps - p_0) / eps
    return flu_coeff*p_0 + refl_coeff*D1*dp


@jit
def model_nlayer_fd(rho, mua, musp, depths, freq, c, n, n_ext=1, int_limit=10, int_divs=10, eps=1e-16):
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
    """
    nlayer = len(mua)
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1 / (3 * (mua[0] + musp[0]))
    z0 = 3*D1
    zb = 2*D1*imp
    w = 2*pi*freq
    v = c / n
    wave = w/v*1j
    if nlayer == 2:
        D2 = D1, 1 / (3 * (mua[1] + musp[1]))
        alpha_args2 = (wave + mua[0]) / D1, (wave + mua[1]) / D2[1]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D2, _2_layer_phi, alpha_args2)) / (2*pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D2, _2_layer_phi, alpha_args2)) / (2*pi)
    elif nlayer == 3:
        D3 = D1, 1 / (3 * (mua[1] + musp[1])), 1 / (3 * (mua[2] + musp[2]))
        alpha_args3 = (wave + mua[0]) / D1, (wave + mua[1]) / D3[1], (wave + mua[2]) / D3[2]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D3, _3_layer_phi, alpha_args3)) / (2*pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D3, _3_layer_phi, alpha_args3)) / (2*pi)
    elif nlayer == 4:
        D4 = D1, 1 / (3 * (mua[1] + musp[1])), 1 / (3 * (mua[2] + musp[2])), 1 / (3 * (mua[3] + musp[3]))
        alpha_args4 = (wave + mua[0]) / D1, (wave + mua[1]) / D4[1], (wave + mua[2]) / D4[2], (wave + mua[3]) / D4[3]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D4, _4_layer_phi, alpha_args4)) / (2*pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D4, _4_layer_phi, alpha_args4)) / (2*pi)
    dp = (p_eps - p_0) / eps
    return flu_coeff*p_0 + refl_coeff*D1*dp


@jit
def model_nlayer_g2(rho, tau, mua, musp, depths, BFi, wavelength, n, n_ext=1, beta=0.5, tau_0=0, int_limit=10, int_divs=10, eps=1e-16):
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
    """
    nlayer = len(mua)
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1 / (3 * (mua[0] + musp[0]))
    z0 = 3*D1
    zb = 2*D1*imp
    k0 = 2 * pi * n / wavelength
    if nlayer == 2:
        D2 = D1, 1/(3 * (mua[1] + musp[1]))
        alpha_args2 = (mua[0]+2*musp[0]*k0**2*BFi[0]*tau)/D2[0], (mua[1]+2*musp[1]*k0**2*BFi[1]*tau)/D2[1]
        alpha_norm_args2 = (mua[0]+2*musp[0]*k0**2*BFi[0]*tau_0)/D2[0], (mua[1]+2*musp[1]*k0**2*BFi[1]*tau_0)/D2[1]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D2, _2_layer_phi, alpha_args2)) / (2*pi)
        p_0_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D2, _2_layer_phi, alpha_norm_args2)) / (2*pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D2, _2_layer_phi, alpha_args2)) / (2*pi)
        p_eps_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D2, _2_layer_phi, alpha_norm_args2)) / (2*pi)
    elif nlayer == 3:
        D3 = D1, 1/(3 * (mua[1] + musp[1])), 1/(3 * (mua[2] + musp[2]))
        alpha_args3 = (mua[0]+2*musp[0]*k0**2*BFi[0]*tau)/D3[0], (mua[1]+2*musp[1]*k0**2*BFi[1]*tau)/D3[1], (mua[2]+2*musp[2]*k0**2*BFi[2]*tau)/D3[2]
        alpha_norm_args3 = (mua[0]+2*musp[0]*k0**2*BFi[0]*tau_0)/D3[0], (mua[1]+2*musp[1]*k0**2*BFi[1]*tau_0)/D3[1], (mua[2]+2*musp[2]*k0**2*BFi[2]*tau_0)/D3[2]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D3, _3_layer_phi, alpha_args3)) / (2*pi)
        p_0_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D3, _3_layer_phi, alpha_norm_args3)) / (2*pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D3, _3_layer_phi, alpha_args3)) / (2*pi)
        p_eps_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D3, _3_layer_phi, alpha_norm_args3)) / (2*pi)
    elif nlayer == 4:
        D4 = D1, 1/(3 * (mua[1] + musp[1])), 1/(3 * (mua[2] + musp[2])), 1/(3 * (mua[3] + musp[3]))
        alpha_args4 = (mua[0]+2*musp[0]*k0**2*BFi[0]*tau)/D4[0], (mua[1]+2*musp[1]*k0**2*BFi[1]*tau)/D4[1], (mua[2]+2*musp[2]*k0**2*BFi[2]*tau)/D4[2], (mua[3]+2*musp[3]*k0**2*BFi[3]*tau)/D4[3]
        alpha_norm_args4 = (mua[0]+2*musp[0]*k0**2*BFi[0]*tau_0)/D4[0], (mua[1]+2*musp[1]*k0**2*BFi[1]*tau_0)/D4[1], (mua[2]+2*musp[2]*k0**2*BFi[2]*tau_0)/D4[2], (mua[3]+2*musp[3]*k0**2*BFi[3]*tau_0)/D4[3]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D4, _4_layer_phi, alpha_args4)) / (2*pi)
        p_0_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D4, _4_layer_phi, alpha_norm_args4)) / (2*pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D4, _4_layer_phi, alpha_args4)) / (2*pi)
        p_eps_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D4, _4_layer_phi, alpha_norm_args4)) / (2*pi)
    dp = (p_eps - p_0) / eps
    dp_norm = (p_eps_norm - p_0_norm) / eps
    g1 = (flu_coeff*p_0 + refl_coeff*D1*dp) / (flu_coeff*p_0_norm + refl_coeff*D1*dp_norm)
    return 1 + beta * g1 ** 2
