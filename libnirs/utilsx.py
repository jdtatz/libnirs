"""
Aronson, R. (1995). Boundary conditions for diffusion of light. Journal of the Optical Society of America A, 12(11), 2532. doi:10.1364/josaa.12.002532
Contini, D., Martelli, F., & Zaccanti, G. (1997). Photon migration through a turbid slab described by a model based on diffusion approximation I Theory. Applied Optics, 36(19), 4587. doi:10.1364/ao.36.004587
"""

from functools import partial

import jax
from jax.lax import cond, select
from jax.numpy import atanh, log, reciprocal, sqrt, stack, zeros

_USE_ALT_G_COEF = True


def _fma(a, b, c):
    # FIXME: https://github.com/jax-ml/jax/issues/23919
    return a * b + c


def _abs_sym_norm(a, b):
    # TODO: add an expansion for when `a ≈ b`
    return abs(a - b) / (a + b)


def _p_expr(n1, n2):
    return sqrt(_abs_sym_norm(n1, n2))


def _g_expr(n1, n2):
    return _abs_sym_norm(n1**2, n2**2)


def _rf_uk_integrands(t, g, is_lower):
    tr = reciprocal(t)
    t2 = t**2
    t3 = t**3
    t4 = t**4
    t6 = t**6
    t8 = t**8
    g2 = g**2
    integrand1 = -((t2 - 1) * (t2 + 1) * (2 * t4 - 2 * g * (t2 + t6) + g2 * (1 + t8))) / (t3 * (g * t2 - 1) ** 2)
    integrand2 = select(is_lower, tr - t, tr + t) * integrand1
    # return integrand1, integrand2
    return stack((integrand1, integrand2), 0)


def _r_phi_j_from_ints(n1, n2, g, rf_u1_int, rf_u2_int):
    is_lte = n1 <= n2
    g_r = 2 * g / (1 + g)
    u1 = 2 * rf_u1_int
    u2 = 3 * rf_u2_int
    r_phi = select(is_lte, u1, u1 + g_r)
    r_j = select(is_lte, u2, u2 + sqrt(g_r) ** 3)
    return r_phi, r_j


def _r_phi_j_cond(n1, n2, rf_uk_int_func):
    def _eq_cond_fn(n1, n2):
        out = jax.eval_shape(lambda a, b: a / b, n1, n2)
        z = zeros(shape=out.shape, dtype=out.dtype, device=out.sharding)
        return z, z

    return cond(n1 == n2, _eq_cond_fn, rf_uk_int_func, n1, n2)


def _impedence_from_parts(r_phi, r_j):
    return (1 + r_j) / (1 - r_phi)


def _refl_from_parts(r_j):
    return (1 - r_j) / 2


def _fluen_from_parts(r_phi):
    return (1 - r_phi) / 4


def _r_phi_j_quad_inner(n1, n2):
    from quadax import quadgk

    is_lower = n1 < n2
    p = _p_expr(n1, n2)
    g = _g_expr(n1, n2)
    if _USE_ALT_G_COEF:
        rat = g / (2 * select(is_lower, (1 - g), (1 + g)))
    else:
        n_r2 = (n2 / n1) ** 2
        rat = select(is_lower, n_r2 - 1, 1 - n_r2) / 4
    c1 = rat / 2
    c2 = sqrt(rat) ** 3 / 2
    integrand = _rf_uk_integrands
    (integral1, integral2), _info = quadgk(integrand, [p, 1], args=(g, is_lower))
    # print(_info)
    return _r_phi_j_from_ints(n1, n2, g, c1 * integral1, c2 * integral2)


def _r_phi_j_quad(n1, n2):
    return _r_phi_j_cond(n1, n2, _r_phi_j_quad_inner)


def impedence_quad(n1, n2):
    r_phi, r_j = _r_phi_j_quad(n1, n2)
    return _impedence_from_parts(r_phi, r_j)


# NOTE: Testing Only
def _reflectance_coeff_quad(n1, n2):
    _, r_j = _r_phi_j_quad(n1, n2)
    return _refl_from_parts(r_j)


# NOTE: Testing Only
def _fluence_rate_coeff_quad(n1, n2):
    r_phi, _ = _r_phi_j_quad(n1, n2)
    return _fluen_from_parts(r_phi)


def ecbc_coeffs_quad(n1, n2):
    r_phi, r_j = _r_phi_j_quad(n1, n2)
    return (
        _impedence_from_parts(r_phi, r_j),
        _refl_from_parts(r_j),
        _fluen_from_parts(r_phi),
    )


def _rf_u1_int_exact(g, p):
    return (
        -3
        - 6 * g
        + 6 * g**2
        + 8 * g**3
        - 3 * g**4
        - 6 * g**5
        + (3 * g**5) / p**2
        - 3 * g * (-1 + g**2) * p**2
        + g**3 * p**6
        + (3 * (-1 + g**2) ** 3) / (-1 + g * p**2)
        + 6 * (-1 + g**2) * (-2 * g**4 * log(p) + (-1 + g**4) * log((-1 + g * p**2) / (-1 + g)))
    ) / (24 * g**2)


def _upper_rf_u2_int_exact(g, p):
    return (
        sqrt(g)
        * (
            -70 * g**5
            - 70 * g**4 * (-6 + g * (3 + 5 * g)) * p**2
            + (525 + g * (490 + g * (-525 + g * (-628 + 35 * g * (-15 + g * (14 + 15 * g)))))) * p**3
            + 105 * (-5 + g * (-3 + g * (7 + g * (5 + g * (-3 + g * (-7 + g * (3 + 5 * g))))))) * p**4
            + g * (-525 + g * (-490 + g * (525 + g * (628 - 35 * g * (-15 + g * (14 + 15 * g)))))) * p**5
            - 70 * (-1 + g) * g * (1 + g) * (5 + 3 * g) * p**6
            - 14 * g**2 * (-5 + g * (3 + 5 * g)) * p**8
            + 6 * g**3 * (-5 + 7 * g) * p**10
            + 30 * g**4 * p**12
        )
        / (p**3 * (-1 + g * p**2))
        + 105 * (-1 + g) ** 2 * (1 + g) ** 3 * (5 + g * (-2 + 5 * g)) * (atanh(sqrt(g)) - atanh(sqrt(g) * p))
    ) / (840 * sqrt(2) * g**2 * sqrt(1 + g) ** 3)


def _lower_rf_u2_int_exact(g, p):
    return (
        2
        * sqrt(g)
        * (
            -70 * g**5
            - 70 * g**4 * (-6 + g * (-3 + 5 * g)) * p**2
            + (-525 + g * (140 + g * (735 + g * (-128 + 35 * g * (-21 + g * (-4 + 15 * g)))))) * p**3
            + 105 * (5 + g * (-3 + g * (-7 + g * (5 + g * (3 + g * (-7 + g * (-3 + 5 * g))))))) * p**4
            + g * (525 + g * (-140 + g * (-735 + g * (128 + 35 * g * (21 + (4 - 15 * g) * g))))) * p**5
            - 70 * (-1 + g) * g * (1 + g) * (-5 + 3 * g) * p**6
            + 14 * g**2 * (-5 + g * (-3 + 5 * g)) * p**8
            + 6 * g**3 * (5 + 7 * g) * p**10
            - 30 * g**4 * p**12
        )
        + 210
        * (-1 + g) ** 3
        * (1 + g) ** 2
        * (5 + g * (2 + 5 * g))
        * p**3
        * (-1 + g * p**2)
        * (atanh(sqrt(g)) - atanh(sqrt(g) * p))
    ) / (1680 * sqrt(2) * sqrt(1 - g) ** 3 * g**2 * p**3 * (-1 + g * p**2))


def _r_phi_j_exact_inner(n1, n2):
    is_lower = n1 < n2
    p = _p_expr(n1, n2)
    g = _g_expr(n1, n2)
    rf_u1_int = _rf_u1_int_exact(g, p) / select(is_lower, (1 - g), (1 + g))
    rf_u2_int = cond(is_lower, _lower_rf_u2_int_exact, _upper_rf_u2_int_exact, g, p)
    return _r_phi_j_from_ints(n1, n2, g, rf_u1_int, rf_u2_int)


def _r_phi_j_exact(n1, n2):
    return _r_phi_j_cond(n1, n2, _r_phi_j_exact_inner)


def impedence_exact(n1, n2):
    r_phi, r_j = _r_phi_j_exact(n1, n2)
    return _impedence_from_parts(r_phi, r_j)


# NOTE: Testing Only
def _reflectance_coeff_exact(n1, n2):
    _, r_j = _r_phi_j_exact(n1, n2)
    return _refl_from_parts(r_j)


# NOTE: Testing Only
def _fluence_rate_coeff_exact(n1, n2):
    r_phi, _ = _r_phi_j_exact(n1, n2)
    return _fluen_from_parts(r_phi)


def ecbc_coeffs_exact(n1, n2):
    r_phi, r_j = _r_phi_j_exact(n1, n2)
    return (
        _impedence_from_parts(r_phi, r_j),
        _refl_from_parts(r_j),
        _fluen_from_parts(r_phi),
    )
