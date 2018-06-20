import numpy as np
from .utils import jit, vectorize, integrate, gen_coeffs, j0


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
    a1, a2 = np.sqrt(s**2 + alpha_args[0]), np.sqrt(s**2 + alpha_args[1])
    return (2*np.exp(a1*(l + zb))*(a1*D1*np.cosh(a1*(l - z0)) + a2*D2*np.sinh(a1*(l - z0)))*np.sinh(a1*(z + zb)))/(a1*D1*(a2*D2*(-1 + np.exp(2*a1*(l + zb))) + a1*D1*(1 + np.exp(2*a1*(l + zb)))))


@jit
def _3_layer_phi(s, z, z0, zb, depths, D, alpha_args):
    l1, l2 = depths[0], depths[1]
    D1, D2, D3 = D[0], D[1], D[2]
    a1, a2, a3 = np.sqrt(s**2 + alpha_args[0]), np.sqrt(s**2 + alpha_args[1]), np.sqrt(s**2 + alpha_args[2])
    return ((((a1*D1*np.cosh(a1*(l1 - z0))*(a2*D2*np.cosh(a2*(l1 - l2)) - a3*D3*np.sinh(a2*(l1 - l2))) + a2*D2*(a3*D3*np.cosh(a2*(l1 - l2)) - a2*D2*np.sinh(a2*(l1 - l2)))*np.sinh(a1*(l1 - z0)))*np.sinh(a1*(z + zb)))/(-(np.sinh(a2*(l1 - l2))*(a1*a3*D1*D3*np.cosh(a1*(l1 + zb)) + a2**2*D2**2*np.sinh(a1*(l1 + zb)))) + a2*D2*np.cosh(a2*(l1 - l2))*(a1*D1*np.cosh(a1*(l1 + zb)) + a3*D3*np.sinh(a1*(l1 + zb)))))/(a1*D1))


@jit
def _4_layer_phi(s, z, z0, zb, depths, D, alpha_args):
    l1, l2, l3 = depths[0], depths[1], depths[2]
    D1, D2, D3, D4 = D[0], D[1], D[2], D[3]
    a1, a2, a3, a4 = np.sqrt(s**2 + alpha_args[0]), np.sqrt(s**2 + alpha_args[1]), np.sqrt(s**2 + alpha_args[2]), np.sqrt(s**2 + alpha_args[3])
    return (np.exp(-(a1*(z + z0)))*(-1 + np.exp(2*a1*(z + zb)))*(a1*D1*(np.exp(2*a1*l1) + np.exp(2*a1*z0))*(a3*D3*np.sinh(a2*(l1 - l2))*(-(a4*D4*np.cosh(a3*(l2 - l3))) + a3*D3*np.sinh(a3*(l2 - l3))) + a2*D2*np.cosh(a2*(l1 - l2))*(a3*D3*np.cosh(a3*(l2 - l3)) - a4*D4*np.sinh(a3*(l2 - l3)))) + a2*D2*(np.exp(2*a1*l1) - np.exp(2*a1*z0))*(a3*D3*np.cosh(a2*(l1 - l2))*(a4*D4*np.cosh(a3*(l2 - l3)) - a3*D3*np.sinh(a3*(l2 - l3))) + a2*D2*np.sinh(a2*(l1 - l2))*(-(a3*D3*np.cosh(a3*(l2 - l3))) + a4*D4*np.sinh(a3*(l2 - l3))))))/(2.*a1*D1*(a1*D1*(1 + np.exp(2*a1*(l1 + zb)))*(a3*D3*np.sinh(a2*(l1 - l2))*(-(a4*D4*np.cosh(a3*(l2 - l3))) + a3*D3*np.sinh(a3*(l2 - l3))) + a2*D2*np.cosh(a2*(l1 - l2))*(a3*D3*np.cosh(a3*(l2 - l3)) - a4*D4*np.sinh(a3*(l2 - l3)))) + a2*D2*(-1 + np.exp(2*a1*(l1 + zb)))*(a3*D3*np.cosh(a2*(l1 - l2))*(a4*D4*np.cosh(a3*(l2 - l3)) - a3*D3*np.sinh(a3*(l2 - l3))) + a2*D2*np.sinh(a2*(l1 - l2))*(-(a3*D3*np.cosh(a3*(l2 - l3))) + a4*D4*np.sinh(a3*(l2 - l3))))))


@jit
def _phi_integrator(s, z, rho, z0, zb, ls, Ds, phi_func, alpha_args):
    return s*j0(s*rho)*phi_func(s, z, z0, zb, ls, Ds, alpha_args)


@jit
def model_2_layer_ss(rho, mua1, musp1, mua2, musp2, l1, n, n_ext=1, int_limit=10, int_divs=10, eps=1e-16):
    """Model Steady-State Reflectance in 2 Layers with Extrapolated Boundary Condition.
    Source: "Noninvasive determination of the optical properties of two-layered turbid media"
    parameters:
        rho := Source-Detector Seperation [length]
        mua1 := Absorption Coefficent of 1st Layer [1/length]
        musp1 := Reduced Scattering Coefficent of 1st Layer [1/length]
        mua2 := Absorption Coefficent of 2nd Layer [1/length]
        musp2 := Reduced Scattering Coefficent of 2nd Layer [1/length]
        l1 := 1st Layer Depth [length]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        int_limit := Integration Limit [length]
        int_divs := Number of subregions to integrate over []
        eps := Epsilion for numerical diffrention [length]
    """
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1/(3*(mua1 + musp1))
    D2 = 1/(3*(mua2 + musp2))
    z0 = 3*D1
    zb = 2*D1*imp
    
    alpha_args = mua1 / D1, mua2 / D2
    ls = (l1,)
    Ds = D1, D2

    p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, ls, Ds, _2_layer_phi, alpha_args)) / (2*np.pi)
    p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, ls, Ds, _2_layer_phi, alpha_args)) / (2*np.pi)
    dp = (p_eps - p_0) / eps
    return flu_coeff*p_0 + refl_coeff*D1*dp


@jit
def model_2_layer_fd(rho, mua1, musp1, mua2, musp2, l1, freq, c, n, n_ext=1, int_limit=10, int_divs=10, eps=1e-16):
    """Model Frequncy-Domain Reflectance in 2 Layers with Extrapolated Boundary Condition.
    Source: "Noninvasive determination of the optical properties of two-layered turbid media"
    parameters:
        rho := Source-Detector Seperation [length]
        mua1 := Absorption Coefficent of 1st Layer [1/length]
        musp1 := Reduced Scattering Coefficent of 1st Layer [1/length]
        mua2 := Absorption Coefficent of 2nd Layer [1/length]
        musp2 := Reduced Scattering Coefficent of 2nd Layer [1/length]
        l1 := 1st Layer Depth [length]
        freq := Frequncy of Source [1/time]
        c := Speed of Light in vacuum [length/time]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        int_limit := Integration Limit [length]
        int_divs := Number of subregions to integrate over []
        eps := Epsilion for numerical diffrention [length]
    """
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1/(3*(mua1 + musp1))
    D2 = 1/(3*(mua2 + musp2))
    z0 = 3*D1
    zb = 2*D1*imp
    w = 2*np.pi*freq
    v = c / n
    wave = w/v*1j
    
    alpha_args = (wave + mua1) / D1, (wave + mua2) / D2
    ls = (l1,)
    Ds = D1, D2

    p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, ls, Ds, _2_layer_phi, alpha_args)) / (2*np.pi)
    p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, ls, Ds, _2_layer_phi, alpha_args)) / (2*np.pi)
    dp = (p_eps - p_0) / eps
    return flu_coeff*p_0 + refl_coeff*D1*dp


@jit
def model_2_layer_g2(rho, tau, mua1, musp1, mua2, musp2, l1, BFi1, BFi2, wavelength, n, n_ext=1, beta=0.5, tau_0=0, int_limit=10, int_divs=10, eps=1e-16):
    """Model g2 (autocorelation) for Diffuse Correlation Spectroscopy in 2 Layers with Extrapolated Boundary Condition.
    Source1: "Noninvasive determination of the optical properties of two-layered turbid media"
    Source2: "Diffuse optics for tissue monitoring and tomography"
    parameters:
        rho := Source-Detector Seperation [length]
        tau := Correlation Time [time]
        mua1 := Absorption Coefficent of 1st Layer [1/length]
        musp1 := Reduced Scattering Coefficent of 1st Layer [1/length]
        mua2 := Absorption Coefficent of 2nd Layer [1/length]
        musp2 := Reduced Scattering Coefficent of 2nd Layer [1/length]
        l1 := 1st Layer Depth [length]
        BFi1 := Blood Flow index of 1st Layer []
        BFi2 := Blood Flow index of 2nd Layer []
        wavelength := Measurement Wavelength [length]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        beta := Beta derived for Siegert relation []
        tau_0 := The first tau for normalization [time]
        int_limit := Integration Limit [length]
        int_divs := Number of subregions to integrate over []
        eps := Epsilion for numerical diffrention [length]
    """
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1/(3*(mua1 + musp1))
    D2 = 1/(3*(mua2 + musp2))
    z0 = 3*D1
    zb = 2*D1*imp
    k0 = 2 * np.pi * n / wavelength
    
    alpha_args = (mua1 + 2 * musp1 * k0**2 * BFi1 * tau) / D1, (mua2 + 2 * musp2 * k0**2 * BFi2 * tau) / D2
    alpha_norm_args = (mua1 + 2 * musp1 * k0**2 * BFi1 * tau_0) / D1, (mua2 + 2 * musp2 * k0**2 * BFi2 * tau_0) / D2
    ls = (l1,)
    Ds = D1, D2

    p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, ls, Ds, _2_layer_phi, alpha_args)) / (2*np.pi)
    p_0_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, ls, Ds, _2_layer_phi, alpha_norm_args)) / (2*np.pi)
    p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, ls, Ds, _2_layer_phi, alpha_args)) / (2*np.pi)
    p_eps_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, ls, Ds, _2_layer_phi, alpha_norm_args)) / (2*np.pi)

    dp = (p_eps - p_0) / eps
    dp_norm = (p_eps_norm - p_0_norm) / eps
    g1 = (flu_coeff*p_0 + refl_coeff*D1*dp) / (flu_coeff*p_0_norm + refl_coeff*D1*dp_norm)
    return 1 + beta * g1 ** 2


@jit
def model_3_layer_ss(rho, mua1, musp1, mua2, musp2, mua3, musp3, l1, l2, n, n_ext=1, int_limit=10, int_divs=10, eps=1e-16):
    """Model Steady-State Reflectance in 3 Layers with Extrapolated Boundary Condition.
    Source: "Noninvasive determination of the optical properties of two-layered turbid media"
    parameters:
        rho := Source-Detector Seperation [length]
        mua1 := Absorption Coefficent of 1st Layer [1/length]
        musp1 := Reduced Scattering Coefficent of 1st Layer [1/length]
        mua2 := Absorption Coefficent of 2nd Layer [1/length]
        musp2 := Reduced Scattering Coefficent of 2nd Layer [1/length]
        mua3 := Absorption Coefficent of 3rd Layer [1/length]
        musp3 := Reduced Scattering Coefficent of 3rd Layer [1/length]
        l1 := 1st Layer Depth [length]
        l2 := 2nd Layer Depth [length]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        int_limit := Integration Limit [length]
        int_divs := Number of subregions to integrate over []
        eps := Epsilion for numerical diffrention [length]
    """
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1/(3*(mua1 + musp1))
    D2 = 1/(3*(mua2 + musp2))
    D3 = 1/(3*(mua3 + musp3))
    z0 = 3*D1
    zb = 2*D1*imp
    
    alpha_args = mua1 / D1, mua2 / D2, mua3 / D3
    ls = l1, l2
    Ds = D1, D2, D3

    p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, ls, Ds, _3_layer_phi, alpha_args)) / (2*np.pi)
    p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, ls, Ds, _3_layer_phi, alpha_args)) / (2*np.pi)

    dp = (p_eps - p_0) / eps
    return flu_coeff*p_0 + refl_coeff*D1*dp


@jit
def model_3_layer_fd(rho, mua1, musp1, mua2, musp2, mua3, musp3, l1, l2, freq, c, n, n_ext=1, int_limit=10, int_divs=10, eps=1e-16):
    """Model Frequncy-Domain Reflectance in 3 Layers with Extrapolated Boundary Condition.
    Source: "Noninvasive determination of the optical properties of two-layered turbid media"
    parameters:
        rho := Source-Detector Seperation [length]
        mua1 := Absorption Coefficent of 1st Layer [1/length]
        musp1 := Reduced Scattering Coefficent of 1st Layer [1/length]
        mua2 := Absorption Coefficent of 2nd Layer [1/length]
        musp2 := Reduced Scattering Coefficent of 2nd Layer [1/length]
        mua3 := Absorption Coefficent of 3rd Layer [1/length]
        musp3 := Reduced Scattering Coefficent of 3rd Layer [1/length]
        l1 := 1st Layer Depth [length]
        l2 := 2nd Layer Depth [length]
        freq := Frequncy of Source [1/time]
        c := Speed of Light in vacuum [length/time]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        int_limit := Integration Limit [length]
        int_divs := Number of subregions to integrate over []
        eps := Epsilion for numerical diffrention [length]
    """
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1/(3*(mua1 + musp1))
    D2 = 1/(3*(mua2 + musp2))
    D3 = 1/(3*(mua3 + musp3))
    z0 = 3*D1
    zb = 2*D1*imp
    w = 2*np.pi*freq
    v = c / n
    wave = w/v*1j
    
    alpha_args = (wave + mua1) / D1, (wave + mua2) / D2, (wave + mua3) / D3
    ls = l1, l2
    Ds = D1, D2, D3

    p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, ls, Ds, _3_layer_phi, alpha_args)) / (2*np.pi)
    p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, ls, Ds, _3_layer_phi, alpha_args)) / (2*np.pi)

    dp = (p_eps - p_0) / eps
    return flu_coeff*p_0 + refl_coeff*D1*dp


@jit
def model_3_layer_g2(rho, tau, mua1, musp1, mua2, musp2, mua3, musp3, l1, l2, BFi1, BFi2, BFi3, wavelength, n, n_ext=1, beta=0.5, tau_0=0, int_limit=10, int_divs=10, eps=1e-16):
    """Model g2 (autocorelation) for Diffuse Correlation Spectroscopy in 3 Layers with Extrapolated Boundary Condition.
    Source1: "Noninvasive determination of the optical properties of two-layered turbid media"
    Source2: "Diffuse optics for tissue monitoring and tomography"
    parameters:
        rho := Source-Detector Seperation [length]
        tau := Correlation Time [time]
        mua1 := Absorption Coefficent of 1st Layer [1/length]
        musp1 := Reduced Scattering Coefficent of 1st Layer [1/length]
        mua2 := Absorption Coefficent of 2nd Layer [1/length]
        musp2 := Reduced Scattering Coefficent of 2nd Layer [1/length]
        mua3 := Absorption Coefficent of 3rd Layer [1/length]
        musp3 := Reduced Scattering Coefficent of 3rd Layer [1/length]
        l1 := 1st Layer Depth [length]
        l2 := 2nd Layer Depth [length]
        BFi1 := Blood Flow index of 1st Layer []
        BFi2 := Blood Flow index of 2nd Layer []
        BFi3 := Blood Flow index of 3rd Layer []
        wavelength := Measurement Wavelength [length]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        beta := Beta derived for Siegert relation []
        tau_0 := The first tau for normalization [time]
        int_limit := Integration Limit [length]
        int_divs := Number of subregions to integrate over []
        eps := Epsilion for numerical diffrention [length]
    """
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1/(3*(mua1 + musp1))
    D2 = 1/(3*(mua2 + musp2))
    D3 = 1/(3*(mua3 + musp3))
    z0 = 3*D1
    zb = 2*D1*imp
    k0 = 2 * np.pi * n / wavelength
    
    alpha_args = (mua1 + 2 * musp1 * k0**2 * BFi1 * tau) / D1, (mua2 + 2 * musp2 * k0**2 * BFi2 * tau) / D2, (mua3 + 2 * musp3 * k0**2 * BFi3 * tau) / D3
    alpha_norm_args = (mua1 + 2 * musp1 * k0**2 * BFi1 * tau_0) / D1, (mua2 + 2 * musp2 * k0**2 * BFi2 * tau_0) / D2, (mua3 + 2 * musp3 * k0**2 * BFi3 * tau_0) / D3

    ls = l1, l2
    Ds = D1, D2, D3

    p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, ls, Ds, _3_layer_phi, alpha_args)) / (2*np.pi)
    p_0_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, ls, Ds, _3_layer_phi, alpha_norm_args)) / (2*np.pi)
    p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, ls, Ds, _3_layer_phi, alpha_args)) / (2*np.pi)
    p_eps_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, ls, Ds, _3_layer_phi, alpha_norm_args)) / (2*np.pi)

    dp = (p_eps - p_0) / eps
    dp_norm = (p_eps_norm - p_0_norm) / eps
    g1 = (flu_coeff*p_0 + refl_coeff*D1*dp) / (flu_coeff*p_0_norm + refl_coeff*D1*dp_norm)
    return 1 + beta * g1 ** 2




@jit
def model_4_layer_ss(rho, mua1, musp1, mua2, musp2, mua3, musp3, mua4, musp4, l1, l2, l3, n, n_ext=1, int_limit=10, int_divs=10, eps=1e-16):
    """Model Steady-State Reflectance in 4 Layers with Extrapolated Boundary Condition.
    Source: "Noninvasive determination of the optical properties of two-layered turbid media"
    parameters:
        rho := Source-Detector Seperation [length]
        mua1 := Absorption Coefficent of 1st Layer [1/length]
        musp1 := Reduced Scattering Coefficent of 1st Layer [1/length]
        mua2 := Absorption Coefficent of 2nd Layer [1/length]
        musp2 := Reduced Scattering Coefficent of 2nd Layer [1/length]
        mua3 := Absorption Coefficent of 3rd Layer [1/length]
        musp3 := Reduced Scattering Coefficent of 3rd Layer [1/length]
        mua4 := Absorption Coefficent of 4th Layer [1/length]
        musp4 := Reduced Scattering Coefficent of 4th Layer [1/length]
        l1 := 1st Layer Depth [length]
        l2 := 2nd Layer Depth [length]
        l3 := 3rd Layer Depth [length]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        int_limit := Integration Limit [length]
        int_divs := Number of subregions to integrate over []
        eps := Epsilion for numerical diffrention [length]
    """
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1/(3*(mua1 + musp1))
    D2 = 1/(3*(mua2 + musp2))
    D3 = 1/(3*(mua3 + musp3))
    D4 = 1/(3*(mua4 + musp4))
    z0 = 3*D1
    zb = 2*D1*imp
    
    alpha_args = mua1 / D1, mua2 / D2, mua3 / D3, mua4 / D4
    ls = l1, l2, l3
    Ds = D1, D2, D3, D4

    p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, ls, Ds, _4_layer_phi, alpha_args)) / (2*np.pi)
    p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, ls, Ds, _4_layer_phi, alpha_args)) / (2*np.pi)

    dp = (p_eps - p_0) / eps
    return flu_coeff*p_0 + refl_coeff*D1*dp


@jit
def model_4_layer_fd(rho, mua1, musp1, mua2, musp2, mua3, musp3, mua4, musp4, l1, l2, l3, freq, c, n, n_ext=1, int_limit=10, int_divs=10, eps=1e-16):
    """Model Frequncy-Domain Reflectance in 4 Layers with Extrapolated Boundary Condition.
    Source: "Noninvasive determination of the optical properties of two-layered turbid media"
    parameters:
        rho := Source-Detector Seperation [length]
        mua1 := Absorption Coefficent of 1st Layer [1/length]
        musp1 := Reduced Scattering Coefficent of 1st Layer [1/length]
        mua2 := Absorption Coefficent of 2nd Layer [1/length]
        musp2 := Reduced Scattering Coefficent of 2nd Layer [1/length]
        mua3 := Absorption Coefficent of 3rd Layer [1/length]
        musp3 := Reduced Scattering Coefficent of 3rd Layer [1/length]
        mua4 := Absorption Coefficent of 4th Layer [1/length]
        musp4 := Reduced Scattering Coefficent of 4th Layer [1/length]
        l1 := 1st Layer Depth [length]
        l2 := 2nd Layer Depth [length]
        l3 := 3rd Layer Depth [length]
        freq := Frequncy of Source [1/time]
        c := Speed of Light in vacuum [length/time]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        int_limit := Integration Limit [length]
        int_divs := Number of subregions to integrate over []
        eps := Epsilion for numerical diffrention [length]
    """
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1/(3*(mua1 + musp1))
    D2 = 1/(3*(mua2 + musp2))
    D3 = 1/(3*(mua3 + musp3))
    D4 = 1/(3*(mua4 + musp4))
    z0 = 3*D1
    zb = 2*D1*imp
    w = 2*np.pi*freq
    v = c / n
    wave = w/v*1j
    
    alpha_args = (wave + mua1) / D1, (wave + mua2) / D2, (wave + mua3) / D3, (wave + mua4) / D4
    ls = l1, l2, l3
    Ds = D1, D2, D3, D4

    p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, ls, Ds, _4_layer_phi, alpha_args)) / (2*np.pi)
    p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, ls, Ds, _4_layer_phi, alpha_args)) / (2*np.pi)

    dp = (p_eps - p_0) / eps
    return flu_coeff*p_0 + refl_coeff*D1*dp


@jit
def model_4_layer_g2(rho, tau, mua1, musp1, mua2, musp2, mua3, musp3, mua4, musp4, l1, l2, l3, BFi1, BFi2, BFi3, BFi4, wavelength, n, n_ext=1, beta=0.5, tau_0=0, int_limit=10, int_divs=10, eps=1e-16):
    """Model g2 (autocorelation) for Diffuse Correlation Spectroscopy in 4 Layers with Extrapolated Boundary Condition.
    Source1: "Noninvasive determination of the optical properties of two-layered turbid media"
    Source2: "Diffuse optics for tissue monitoring and tomography"
    parameters:
        rho := Source-Detector Seperation [length]
        tau := Correlation Time [time]
        mua1 := Absorption Coefficent of 1st Layer [1/length]
        musp1 := Reduced Scattering Coefficent of 1st Layer [1/length]
        mua2 := Absorption Coefficent of 2nd Layer [1/length]
        musp2 := Reduced Scattering Coefficent of 2nd Layer [1/length]
        mua3 := Absorption Coefficent of 3rd Layer [1/length]
        musp3 := Reduced Scattering Coefficent of 3rd Layer [1/length]
        mua4:= Absorption Coefficent of 4th Layer [1/length]
        musp4 := Reduced Scattering Coefficent of 4th Layer [1/length]
        l1 := 1st Layer Depth [length]
        l2 := 2nd Layer Depth [length]
        l2 := 3rd Layer Depth [length]
        BFi1 := Blood Flow index of 1st Layer []
        BFi2 := Blood Flow index of 2nd Layer []
        BFi3 := Blood Flow index of 3rd Layer []
        BFi4 := Blood Flow index of 4th Layer []
        wavelength := Measurement Wavelength [length]
        n := Media Index of Refraction []
        n_ext := External Index of Refraction []
        beta := Beta derived for Siegert relation []
        tau_0 := The first tau for normalization [time]
        int_limit := Integration Limit [length]
        int_divs := Number of subregions to integrate over []
        eps := Epsilion for numerical diffrention [length]
    """
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1/(3*(mua1 + musp1))
    D2 = 1/(3*(mua2 + musp2))
    D3 = 1/(3*(mua3 + musp3))
    D4 = 1/(3*(mua4 + musp4))
    z0 = 3*D1
    zb = 2*D1*imp
    k0 = 2 * np.pi * n / wavelength

    alpha_args = (mua1 + 2 * musp1 * k0**2 * BFi1 * tau) / D1, (mua2 + 2 * musp2 * k0**2 * BFi2 * tau) / D2, (mua3 + 2 * musp3 * k0**2 * BFi3 * tau) / D3, (mua4 + 2 * musp4 * k0**2 * BFi4 * tau) / D4
    alpha_norm_args = (mua1 + 2 * musp1 * k0**2 * BFi1 * tau_0) / D1, (mua2 + 2 * musp2 * k0**2 * BFi2 * tau_0) / D2, (mua3 + 2 * musp3 * k0**2 * BFi3 * tau_0) / D3, (mua4 + 2 * musp4 * k0**2 * BFi4 * tau_0) / D4
    ls = l1, l2, l3
    Ds = D1, D2, D3, D4

    p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, ls, Ds, _4_layer_phi, alpha_args)) / (2*np.pi)
    p_0_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, ls, Ds, _4_layer_phi, alpha_norm_args)) / (2*np.pi)
    p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, ls, Ds, _4_layer_phi, alpha_args)) / (2*np.pi)
    p_eps_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, ls, Ds, _4_layer_phi, alpha_norm_args)) / (2*np.pi)

    dp = (p_eps - p_0) / eps
    dp_norm = (p_eps_norm - p_0_norm) / eps
    g1 = (flu_coeff*p_0 + refl_coeff*D1*dp) / (flu_coeff*p_0_norm + refl_coeff*D1*dp_norm)
    return 1 + beta * g1 ** 2


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
    D = 1 / (3 * (mua + musp))
    D1 = D[0]
    z0 = 3*D1
    zb = 2*D1*imp
    alpha_args = mua / D
    if nlayer == 2:
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D, _2_layer_phi, alpha_args)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D, _2_layer_phi, alpha_args)) / (2*np.pi)
    elif nlayer == 3:
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D, _3_layer_phi, alpha_args)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D, _3_layer_phi, alpha_args)) / (2*np.pi)
    elif nlayer == 4:
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D, _4_layer_phi, alpha_args)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D, _4_layer_phi, alpha_args)) / (2*np.pi)
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
    D = 1 / (3 * (mua + musp))
    D1 = D[0]
    z0 = 3*D1
    zb = 2*D1*imp
    w = 2*np.pi*freq
    v = c / n
    wave = w/v*1j
    alpha_args = (wave + mua) / D
    if nlayer == 2:
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D, _2_layer_phi, alpha_args)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D, _2_layer_phi, alpha_args)) / (2*np.pi)
    elif nlayer == 3:
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D, _3_layer_phi, alpha_args)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D, _3_layer_phi, alpha_args)) / (2*np.pi)
    elif nlayer == 4:
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D, _4_layer_phi, alpha_args)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D, _4_layer_phi, alpha_args)) / (2*np.pi)
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
    D = 1/(3 * (mua + musp))
    D1 = D[0]
    z0 = 3*D1
    zb = 2*D1*imp
    k0 = 2 * np.pi * n / wavelength
    alpha_args = (mua + 2 * musp * k0**2 * BFi * tau) / D
    alpha_norm_args = (mua + 2 * musp * k0**2 * BFi * tau_0) / D
    if nlayer == 2:
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D, _2_layer_phi, alpha_args)) / (2*np.pi)
        p_0_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D, _2_layer_phi, alpha_norm_args)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D, _2_layer_phi, alpha_args)) / (2*np.pi)
        p_eps_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D, _2_layer_phi, alpha_norm_args)) / (2*np.pi)
    elif nlayer == 3:
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D, _3_layer_phi, alpha_args)) / (2*np.pi)
        p_0_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D, _3_layer_phi, alpha_norm_args)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D, _3_layer_phi, alpha_args)) / (2*np.pi)
        p_eps_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D, _3_layer_phi, alpha_norm_args)) / (2*np.pi)
    elif nlayer == 4:
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D, _4_layer_phi, alpha_args)) / (2*np.pi)
        p_0_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D, _4_layer_phi, alpha_norm_args)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D, _4_layer_phi, alpha_args)) / (2*np.pi)
        p_eps_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D, _4_layer_phi, alpha_norm_args)) / (2*np.pi)
    dp = (p_eps - p_0) / eps
    dp_norm = (p_eps_norm - p_0_norm) / eps
    g1 = (flu_coeff*p_0 + refl_coeff*D1*dp) / (flu_coeff*p_0_norm + refl_coeff*D1*dp_norm)
    return 1 + beta * g1 ** 2
