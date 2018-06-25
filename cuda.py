import os
os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'
import math
import numpy as np
import numba as nb
import numba.cuda

jit = nb.cuda.jit

_x_kr21 = np.array([-9.956571630258080807355272806890028e-01, -9.739065285171717200779640120844521e-01, -9.301574913557082260012071800595083e-01, -8.650633666889845107320966884234930e-01, -7.808177265864168970637175783450424e-01, -6.794095682990244062343273651148736e-01, -5.627571346686046833390000992726941e-01, -4.333953941292471907992659431657842e-01, -2.943928627014601981311266031038656e-01, -1.488743389816312108848260011297200e-01, 0.0, 1.488743389816312108848260011297200e-01, 2.943928627014601981311266031038656e-01, 4.333953941292471907992659431657842e-01, 5.627571346686046833390000992726941e-01, 6.794095682990244062343273651148736e-01, 7.808177265864168970637175783450424e-01, 8.650633666889845107320966884234930e-01, 9.301574913557082260012071800595083e-01, 9.739065285171717200779640120844521e-01, 9.956571630258080807355272806890028e-01])

_w_kr21 = np.array([1.169463886737187427806439606219205e-02, 3.255816230796472747881897245938976e-02, 5.475589657435199603138130024458018e-02, 7.503967481091995276704314091619001e-02, 9.312545458369760553506546508336634e-02, 1.093871588022976418992105903258050e-01, 1.234919762620658510779581098310742e-01, 1.347092173114733259280540017717068e-01, 1.427759385770600807970942731387171e-01, 1.477391049013384913748415159720680e-01, 1.494455540029169056649364683898212e-01, 1.477391049013384913748415159720680e-01, 1.427759385770600807970942731387171e-01, 1.347092173114733259280540017717068e-01, 1.234919762620658510779581098310742e-01, 1.093871588022976418992105903258050e-01, 9.312545458369760553506546508336634e-02, 7.503967481091995276704314091619001e-02, 5.475589657435199603138130024458018e-02, 3.255816230796472747881897245938976e-02, 1.169463886737187427806439606219205e-02])


@nb.cuda.jit(device=True)
def integrate(func, a, b, divs, args):
    """Integrate 'func' w/ 'args' over the region (a, b). The region can subdived by 'divs' for better numerical accuracy"""
    skip = (b - a) / divs
    c_1 = skip / 2
    c_2 = c_1 + a
    x_kr21 = nb.cuda.const.array_like(_x_kr21)
    w_kr21 = nb.cuda.const.array_like(_w_kr21)
    integrator = func(c_1 * x_kr21[0] + c_2, *args) * w_kr21[0]
    for i in range(1, x_kr21.shape[0]):
        integrator += func(c_1 * x_kr21[i] + c_2, *args) * w_kr21[i]
    for j in range(1, divs):
        c_2 += skip
        for i in range(x_kr21.shape[0]):
            integrator += func(c_1 * x_kr21[i] + c_2, *args) * w_kr21[i]
    return c_1 * integrator


@nb.cuda.jit(device=True)
def gen_impedance(n):
    if n <= 1:
        return 3.084635 - 6.531194 * n + 8.357854 * n * n - 5.082751 * n**3 + 1.171382 * n**4
    return 504.332889 - 2641.00214 * n + 5923.699064 * n * n - 7376.355814 * n**3 + 5507.53041 * n**4 - 2463.357945 * n**5 + 610.956547 * n**6 - 64.8047 * n**7


@nb.cuda.jit(device=True)
def _gen_reflectance_coeff(t, n, m):
    Rs = ((n*math.cos(t)-m*math.sqrt(1-(n/m*math.sin(t))**2)) / (n*math.cos(t)+m*math.sqrt(1-(n/m*math.sin(t))**2)))**2
    Rp = ((n*math.sqrt(1-(n/m*math.sin(t))**2)-m*math.cos(t)) / (n*math.sqrt(1-(n/m*math.sin(t))**2)+m*math.cos(t)))**2
    Rfres = (Rs + Rp) / 2
    return 3*(1 - Rfres)*math.cos(t)**2*math.sin(t) / 2


@nb.cuda.jit(device=True)
def _gen_fluence_rate_coeff(t, n, m):
    Rs = ((n*math.cos(t)-m*math.sqrt(1-(n/m*math.sin(t))**2)) / (n*math.cos(t)+m*math.sqrt(1-(n/m*math.sin(t))**2)))**2
    Rp = ((n*math.sqrt(1-(n/m*math.sin(t))**2)-m*math.cos(t)) / (n*math.sqrt(1-(n/m*math.sin(t))**2)+m*math.cos(t)))**2
    Rfres = (Rs + Rp) / 2
    return (1 - Rfres)*math.cos(t)*math.sin(t) / 2


@nb.cuda.jit(device=True)
def gen_coeffs(n, n_ext):
    return (
        gen_impedance(n/n_ext), 
        integrate(_gen_reflectance_coeff, 0, math.asin(n_ext/n), 10, (n, n_ext)), 
        integrate(_gen_fluence_rate_coeff, 0, math.asin(n_ext/n), 10, (n, n_ext))
    )


_RP = np.array([-4.79443220978201773821E9,1.95617491946556577543E12,-2.49248344360967716204E14,9.70862251047306323952E15])
_RQ = np.array([4.99563147152651017219E2,1.73785401676374683123E5,4.84409658339962045305E7,1.11855537045356834862E10,2.11277520115489217587E12,3.10518229857422583814E14,3.18121955943204943306E16,1.71086294081043136091E18])
_PP = np.array([  7.96936729297347051624E-4,8.28352392107440799803E-2,1.23953371646414299388E0,5.44725003058768775090E0,8.74716500199817011941E0,5.30324038235394892183E0,9.99999999999999997821E-1,])
_PQ = np.array([9.24408810558863637013E-4,8.56288474354474431428E-2,1.25352743901058953537E0,5.47097740330417105182E0,8.76190883237069594232E0,5.30605288235394617618E0,1.00000000000000000218E0,])
_QP = np.array([-1.13663838898469149931E-2,-1.28252718670509318512E0,-1.95539544257735972385E1,-9.32060152123768231369E1,-1.77681167980488050595E2,-1.47077505154951170175E2,-5.14105326766599330220E1,-6.05014350600728481186E0,])
_QQ = np.array([  6.43178256118178023184E1,8.56430025976980587198E2,3.88240183605401609683E3,7.24046774195652478189E3,5.93072701187316984827E3,2.06209331660327847417E3,2.42005740240291393179E2,])
_DR1 = 5.783185962946784521175995758455807035071
_DR2 = 30.47126234366208639907816317502275584842
_SQ2OPI = 0.79788456080286535588
_PIO4 = .78539816339744830962


@nb.cuda.jit(device=True)
def _polevl(x, coef):
    ans = coef[0]
    for i in range(1, coef.shape[0]):
        ans = ans*x + coef[i]
    return ans


@nb.cuda.jit(device=True)
def _p1evl(x, coef):
    ans = x + coef[0]
    for i in range(1, coef.shape[0]):
        ans = ans*x + coef[i]
    return ans


@nb.cuda.jit(device=True)
def j0(x):
    """Bessel function of the first kind of order 0. Adapted from "Cephes Mathematical Functions Library"."""
    PP = nb.cuda.const.array_like(_PP)
    PQ = nb.cuda.const.array_like(_PQ)
    QP = nb.cuda.const.array_like(_QP)
    QQ = nb.cuda.const.array_like(_QQ)
    RP = nb.cuda.const.array_like(_RP)
    RQ = nb.cuda.const.array_like(_RQ)
    x = abs(x)
    if x > 5:
        w = 5 / x
        q = 25 / (x*x)
        p = _polevl(q, PP) / _polevl(q, PQ)
        q = _polevl(q, QP) / _p1evl(q, QQ)
        xn = x - _PIO4
        p = p * math.cos(xn) - w * q * math.sin(xn)
        return p * _SQ2OPI / math.sqrt(x)
    elif x >= 1e-5:
        z = x*x
        return (z - _DR1) * (z - _DR2) * _polevl(z, RP)/_p1evl(z, RQ)
    else:
        return 1 - x*x/4


@nb.cuda.jit(device=True)
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
    mu_eff = math.sqrt(3 * mua * (mua + musp))
    r1_sq = (z - z0)**2 + rho**2
    r1 = math.sqrt(r1_sq)
    r2_sq = (z + z0 + 2*zb)**2 + rho**2
    r2 = math.sqrt(r2_sq)
    flu_rate = 1/(4*math.pi*D)*(math.exp(-mu_eff*r1)/r1 - math.exp(-mu_eff*r2)/r2)
    diff_refl = 1/(4*math.pi)*(z0*(mu_eff + 1/r1)*math.exp(-mu_eff*r1)/r1_sq + (z0 + 2*zb)*(mu_eff + 1/r2)*math.exp(-mu_eff*r2)/r2_sq)
    return flu_coeff * flu_rate + refl_coeff * diff_refl


@nb.cuda.jit(device=True)
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
    w = 2 * math.pi * freq
    k_in = math.sqrt(1+(w/(mua*v))**2)
    k = math.sqrt(3 / 2 * mua * (mua + musp)) * (math.sqrt(k_in+1) + 1j*math.sqrt(k_in-1))
    r1_sq = (z - z0)**2 + rho**2
    r1 = math.sqrt(r1_sq)
    r2_sq = (z + z0 + 2*zb)**2 + rho**2
    r2 = math.sqrt(r2_sq)
    flu_rate = 1/(4*math.pi*D)*(math.exp(-k*r1)/r1 - math.exp(-k*r2)/r2)
    diff_refl = 1/(4*math.pi)*(z0*(k + 1/r1)*math.exp(-k*r1)/r1_sq + (z0 + 2*zb)*(k + 1/r2)*math.exp(-k*r2)/r2_sq)
    return flu_coeff * flu_rate + refl_coeff * diff_refl


@nb.cuda.jit(device=True)
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
    reflectance = 0.5*t**(-5/2)*(4*np.pi*D*v)**(-3/2)*math.exp(-mua*v*t)*(z0*math.exp(-r1_sq/(4*D*v*t))+(z0+2*zb)*math.exp(-r2_sq/(4*D*v*t)))
    fluence_rate = v*(4*math.pi*D*v*t)**(-3/2)*math.exp(-mua*v*t)*(math.exp(-(r1_sq/(4*D*v*t)))-math.exp(-(r2_sq/(4*D*v*t))))
    return flu_coeff*fluence_rate + refl_coeff*reflectance


@nb.cuda.jit(device=True)
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
    k0 = 2 * math.pi * n / wavelength
    k_tau = math.sqrt(3 * mua * musp + musp**2 * k0**2 * 6 * bfi * tau)
    k_norm = math.sqrt(3 * mua * musp + musp**2 * k0**2 * 6 * bfi * first_tau_delay)
    r1 = math.sqrt(rho**2 + (z - z0)**2)
    r2 = math.sqrt(rho**2 + (z + z0 + 2 * zb)**2)
    g1 = (math.exp(-k_tau*r1)/r1 - math.exp(-k_tau*r2)/r2) / (math.exp(-k_norm*r1)/r1 - math.exp(-k_norm*r2)/r2)
    return 1 + beta * g1 ** 2


@nb.cuda.jit(device=True)
def _2_layer_phi(s, z, z0, zb, depths, D, alpha_args):
    l = depths[0]
    D1, D2 = D[0], D[1]
    a1, a2 = math.sqrt(s**2 + alpha_args[0]), math.sqrt(s**2 + alpha_args[1])
    return (2*math.exp(a1*(l + zb))*(a1*D1*math.cosh(a1*(l - z0)) + a2*D2*math.sinh(a1*(l - z0)))*math.sinh(a1*(z + zb)))/(a1*D1*(a2*D2*(-1 + math.exp(2*a1*(l + zb))) + a1*D1*(1 + math.exp(2*a1*(l + zb)))))


@nb.cuda.jit(device=True)
def _3_layer_phi(s, z, z0, zb, depths, D, alpha_args):
    l1, l2 = depths[0], depths[1]
    D1, D2, D3 = D[0], D[1], D[2]
    a1, a2, a3 = math.sqrt(s**2 + alpha_args[0]), math.sqrt(s**2 + alpha_args[1]), math.sqrt(s**2 + alpha_args[2])
    return ((((a1*D1*math.cosh(a1*(l1 - z0))*(a2*D2*math.cosh(a2*(l1 - l2)) - a3*D3*math.sinh(a2*(l1 - l2))) + a2*D2*(a3*D3*math.cosh(a2*(l1 - l2)) - a2*D2*math.sinh(a2*(l1 - l2)))*math.sinh(a1*(l1 - z0)))*math.sinh(a1*(z + zb)))/(-(math.sinh(a2*(l1 - l2))*(a1*a3*D1*D3*math.cosh(a1*(l1 + zb)) + a2**2*D2**2*math.sinh(a1*(l1 + zb)))) + a2*D2*math.cosh(a2*(l1 - l2))*(a1*D1*math.cosh(a1*(l1 + zb)) + a3*D3*math.sinh(a1*(l1 + zb)))))/(a1*D1))


@nb.cuda.jit(device=True)
def _4_layer_phi(s, z, z0, zb, depths, D, alpha_args):
    l1, l2, l3 = depths[0], depths[1], depths[2]
    D1, D2, D3, D4 = D[0], D[1], D[2], D[3]
    a1, a2, a3, a4 = math.sqrt(s**2 + alpha_args[0]), math.sqrt(s**2 + alpha_args[1]), math.sqrt(s**2 + alpha_args[2]), math.sqrt(s**2 + alpha_args[3])
    return (math.exp(-(a1*(z + z0)))*(-1 + math.exp(2*a1*(z + zb)))*(a1*D1*(math.exp(2*a1*l1) + math.exp(2*a1*z0))*(a3*D3*math.sinh(a2*(l1 - l2))*(-(a4*D4*math.cosh(a3*(l2 - l3))) + a3*D3*math.sinh(a3*(l2 - l3))) + a2*D2*math.cosh(a2*(l1 - l2))*(a3*D3*math.cosh(a3*(l2 - l3)) - a4*D4*math.sinh(a3*(l2 - l3)))) + a2*D2*(math.exp(2*a1*l1) - math.exp(2*a1*z0))*(a3*D3*math.cosh(a2*(l1 - l2))*(a4*D4*math.cosh(a3*(l2 - l3)) - a3*D3*math.sinh(a3*(l2 - l3))) + a2*D2*math.sinh(a2*(l1 - l2))*(-(a3*D3*math.cosh(a3*(l2 - l3))) + a4*D4*math.sinh(a3*(l2 - l3))))))/(2.*a1*D1*(a1*D1*(1 + math.exp(2*a1*(l1 + zb)))*(a3*D3*math.sinh(a2*(l1 - l2))*(-(a4*D4*math.cosh(a3*(l2 - l3))) + a3*D3*math.sinh(a3*(l2 - l3))) + a2*D2*math.cosh(a2*(l1 - l2))*(a3*D3*math.cosh(a3*(l2 - l3)) - a4*D4*math.sinh(a3*(l2 - l3)))) + a2*D2*(-1 + math.exp(2*a1*(l1 + zb)))*(a3*D3*math.cosh(a2*(l1 - l2))*(a4*D4*math.cosh(a3*(l2 - l3)) - a3*D3*math.sinh(a3*(l2 - l3))) + a2*D2*math.sinh(a2*(l1 - l2))*(-(a3*D3*math.cosh(a3*(l2 - l3))) + a4*D4*math.sinh(a3*(l2 - l3))))))


@nb.cuda.jit(device=True)
def _phi_integrator(s, z, rho, z0, zb, ls, Ds, phi_func, alpha_args):
    return s*j0(s*rho)*phi_func(s, z, z0, zb, ls, Ds, alpha_args)


@nb.cuda.jit
def model_nlayer_ss(rhos, muas, musps, depths, n, n_ext, int_limit, int_divs, eps, out):
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
    nlayer = muas.shape[1]
    i = nb.cuda.grid(1)
    if i > out.shape[0]:
        return
    rho = rhos[i]
    mua = muas[i]
    musp = musps[i]
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1 / (3 * (mua[0] + musp[0]))
    z0 = 3*D1
    zb = 2*D1*imp
    if nlayer == 2:
        D2 = D1, 1 / (3 * (mua[1] + musp[1]))
        alpha_args2 = mua[0] / D1, mua[1] / D2[1]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D2, _2_layer_phi, alpha_args2)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D2, _2_layer_phi, alpha_args2)) / (2*np.pi)
    elif nlayer == 3:
        D3 = D1, 1 / (3 * (mua[1] + musp[1])), 1 / (3 * (mua[2] + musp[2]))
        alpha_args3 = mua[0] / D1, mua[1] / D3[1], mua[2] / D3[2]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D3, _3_layer_phi, alpha_args3)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D3, _3_layer_phi, alpha_args3)) / (2*np.pi)
    elif nlayer == 4:
        D4 = D1, 1 / (3 * (mua[1] + musp[1])), 1 / (3 * (mua[2] + musp[2])), 1 / (3 * (mua[3] + musp[3]))
        alpha_args4 = mua[0] / D1, mua[1] / D4[1], mua[2] / D4[2], mua[3] / D4[3]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D4, _4_layer_phi, alpha_args4)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D4, _4_layer_phi, alpha_args4)) / (2*np.pi)
    dp = (p_eps - p_0) / eps
    out[i] = flu_coeff*p_0 + refl_coeff*D1*dp


@nb.cuda.jit
def model_nlayer_fd(rhos, muas, musps, depths, freq, c, n, n_ext, int_limit, int_divs, eps, out):
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
    nlayer = muas.shape[1]
    i = nb.cuda.grid(1)
    if i > out.shape[0]:
        return
    rho = rhos[i]
    mua = muas[i]
    musp = musps[i]
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1 / (3 * (mua[0] + musp[0]))
    z0 = 3*D1
    zb = 2*D1*imp
    w = 2*np.pi*freq
    v = c / n
    wave = w/v*1j
    if nlayer == 2:
        D2 = D1, 1 / (3 * (mua[1] + musp[1]))
        alpha_args2 = (wave + mua[0]) / D1, (wave + mua[1]) / D2[1]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D2, _2_layer_phi, alpha_args2)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D2, _2_layer_phi, alpha_args2)) / (2*np.pi)
    elif nlayer == 3:
        D3 = D1, 1 / (3 * (mua[1] + musp[1])), 1 / (3 * (mua[2] + musp[2]))
        alpha_args3 = (wave + mua[0]) / D1, (wave + mua[1]) / D3[1], (wave + mua[2]) / D3[2]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D3, _3_layer_phi, alpha_args3)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D3, _3_layer_phi, alpha_args3)) / (2*np.pi)
    elif nlayer == 4:
        D4 = D1, 1 / (3 * (mua[1] + musp[1])), 1 / (3 * (mua[2] + musp[2])), 1 / (3 * (mua[3] + musp[3]))
        alpha_args4 = (wave + mua[0]) / D1, (wave + mua[1]) / D4[1], (wave + mua[2]) / D4[2], (wave + mua[3]) / D4[3]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D4, _4_layer_phi, alpha_args4)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D4, _4_layer_phi, alpha_args4)) / (2*np.pi)
    dp = (p_eps - p_0) / eps
    out[i] = flu_coeff*p_0 + refl_coeff*D1*dp


@nb.cuda.jit
def model_nlayer_g2(rhos, taus, muas, musps, depths, BFi, wavelength, n, n_ext, beta, tau_0, int_limit, int_divs, eps, out):
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
    nlayer = muas.shape[1]
    i = nb.cuda.grid(1)
    if i > out.shape[0]:
        return
    rho = rhos[i]
    tau = taus[i]
    mua = muas[i]
    musp = musps[i]
    imp, refl_coeff, flu_coeff = gen_coeffs(n, n_ext)
    D1 = 1 / (3 * (mua[0] + musp[0]))
    z0 = 3*D1
    zb = 2*D1*imp
    k0 = 2 * np.pi * n / wavelength
    if nlayer == 2:
        D2 = D1, 1/(3 * (mua[1] + musp[1]))
        alpha_args2 = (mua[0]+2*musp[0]*k0**2*BFi[0]*tau)/D2[0], (mua[1]+2*musp[1]*k0**2*BFi[1]*tau)/D2[1]
        alpha_norm_args2 = (mua[0]+2*musp[0]*k0**2*BFi[0]*tau_0)/D2[0], (mua[1]+2*musp[1]*k0**2*BFi[1]*tau_0)/D2[1]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D2, _2_layer_phi, alpha_args2)) / (2*np.pi)
        p_0_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D2, _2_layer_phi, alpha_norm_args2)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D2, _2_layer_phi, alpha_args2)) / (2*np.pi)
        p_eps_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D2, _2_layer_phi, alpha_norm_args2)) / (2*np.pi)
    elif nlayer == 3:
        D3 = D1, 1/(3 * (mua[1] + musp[1])), 1/(3 * (mua[2] + musp[2]))
        alpha_args3 = (mua[0]+2*musp[0]*k0**2*BFi[0]*tau)/D3[0], (mua[1]+2*musp[1]*k0**2*BFi[1]*tau)/D3[1], (mua[2]+2*musp[2]*k0**2*BFi[2]*tau)/D3[2]
        alpha_norm_args3 = (mua[0]+2*musp[0]*k0**2*BFi[0]*tau_0)/D3[0], (mua[1]+2*musp[1]*k0**2*BFi[1]*tau_0)/D3[1], (mua[2]+2*musp[2]*k0**2*BFi[2]*tau_0)/D3[2]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D3, _3_layer_phi, alpha_args3)) / (2*np.pi)
        p_0_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D3, _3_layer_phi, alpha_norm_args3)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D3, _3_layer_phi, alpha_args3)) / (2*np.pi)
        p_eps_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D3, _3_layer_phi, alpha_norm_args3)) / (2*np.pi)
    elif nlayer == 4:
        D4 = D1, 1/(3 * (mua[1] + musp[1])), 1/(3 * (mua[2] + musp[2])), 1/(3 * (mua[3] + musp[3]))
        alpha_args4 = (mua[0]+2*musp[0]*k0**2*BFi[0]*tau)/D4[0], (mua[1]+2*musp[1]*k0**2*BFi[1]*tau)/D4[1], (mua[2]+2*musp[2]*k0**2*BFi[2]*tau)/D4[2], (mua[3]+2*musp[3]*k0**2*BFi[3]*tau)/D4[3]
        alpha_norm_args4 = (mua[0]+2*musp[0]*k0**2*BFi[0]*tau_0)/D4[0], (mua[1]+2*musp[1]*k0**2*BFi[1]*tau_0)/D4[1], (mua[2]+2*musp[2]*k0**2*BFi[2]*tau_0)/D4[2], (mua[3]+2*musp[3]*k0**2*BFi[3]*tau_0)/D4[3]
        p_0 = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D4, _4_layer_phi, alpha_args4)) / (2*np.pi)
        p_0_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (0, rho, z0, zb, depths, D4, _4_layer_phi, alpha_norm_args4)) / (2*np.pi)
        p_eps = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D4, _4_layer_phi, alpha_args4)) / (2*np.pi)
        p_eps_norm = integrate(_phi_integrator, 0, int_limit, int_divs, (eps, rho, z0, zb, depths, D4, _4_layer_phi, alpha_norm_args4)) / (2*np.pi)
    dp = (p_eps - p_0) / eps
    dp_norm = (p_eps_norm - p_0_norm) / eps
    g1 = (flu_coeff*p_0 + refl_coeff*D1*dp) / (flu_coeff*p_0_norm + refl_coeff*D1*dp_norm)
    out[i] = 1 + beta * g1 ** 2
