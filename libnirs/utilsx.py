"""
Aronson, R. (1995). Boundary conditions for diffusion of light. Journal of the Optical Society of America A, 12(11), 2532. doi:10.1364/josaa.12.002532
Contini, D., Martelli, F., & Zaccanti, G. (1997). Photon migration through a turbid slab described by a model based on diffusion approximation I Theory. Applied Optics, 36(19), 4587. doi:10.1364/ao.36.004587
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.lax import cond

from .quadrature import quad

_USE_ALT_G_COEF = True
_USE_SEP_QUAD = False


def _fma(a, b, c):
    # FIXME: https://github.com/jax-ml/jax/issues/23919
    return a * b + c


def _abs_sym_norm(a, b):
    # TODO: add an expansion for when `a ≈ b`
    return abs(a - b) / (a + b)


def _p_expr(n1, n2, *, xp):
    return xp.sqrt(_abs_sym_norm(n1, n2))


def _g_expr(n1, n2):
    return _abs_sym_norm(n1**2, n2**2)


def _rf_uk_integrands(t, g, is_lower, *, k, xp):
    tr = xp.reciprocal(t)
    t2 = t**2
    t3 = t**3
    t4 = t**4
    t6 = t**6
    t8 = t**8
    g2 = g**2
    integrand1 = -((t2 - 1) * (t2 + 1) * (2 * t4 - 2 * g * (t2 + t6) + g2 * (1 + t8))) / (t3 * (g * t2 - 1) ** 2)
    if k == 1:
        return integrand1
    integrand2 = xp.where(is_lower, tr - t, tr + t) * integrand1
    if k == 2:
        return integrand2
    elif k == (1, 2):
        # return integrand1, integrand2
        return xp.stack((integrand1, integrand2), 0)
    else:
        raise ValueError(k)


# NOTE: `quad` requires scalar integration limits, so use u-sub'd integrand where `t = (u * (1 - p) + (1 + p))/2`, so limits transform `[p, 1] => [-1, 1]`
def _rf_uk_integrands_trans(u, p, g, is_lower, *, k, xp):
    c = (1 - p) / 2
    d = (1 + p) / 2
    t = d + c * u
    du = c
    integrand = _rf_uk_integrands(t, g, is_lower, k=k, xp=xp)
    return du * integrand


def _r_phi_j_from_ints(n1, n2, g, rf_u1_int, rf_u2_int, *, xp):
    is_lte = n1 <= n2
    g_r = 2 * g / (1 + g)
    r_phi = xp.where(is_lte, rf_u1_int, rf_u1_int + g_r)
    r_j = xp.where(is_lte, rf_u2_int, rf_u2_int + xp.sqrt(g_r) ** 3)
    return r_phi, r_j


def _r_phi_j_cond(n1, n2, rf_uk_int_func, *, xp):
    def _eq_cond_fn(n1, n2):
        out = jax.eval_shape(lambda a, b: a / b, n1, n2)
        z = xp.zeros(shape=out.shape, dtype=out.dtype, device=out.sharding)
        return z, z

    # FIXME: use a lazy_where
    if xp is jnp and jnp.ndim(n1) == 0 and jnp.ndim(n2) == 0:
        return cond(n1 == n2, _eq_cond_fn, rf_uk_int_func, n1, n2)
    else:
        return rf_uk_int_func(n1, n2)


def _impedence_from_parts(r_phi, r_j):
    return (1 + r_j) / (1 - r_phi)


def _refl_from_parts(r_j):
    return (1 - r_j) / 2


def _fluen_from_parts(r_phi):
    return (1 - r_phi) / 4


def _r_phi_j_quad_inner(n1, n2, *, xp):
    is_lower = n1 < n2
    p = _p_expr(n1, n2, xp=xp)
    g = _g_expr(n1, n2)
    if _USE_ALT_G_COEF:
        rat = g / (2 * xp.where(is_lower, (1 - g), (1 + g)))
    else:
        n_r2 = (n2 / n1) ** 2
        rat = xp.where(is_lower, n_r2 - 1, 1 - n_r2) / 4
    c1 = rat
    c2 = (3 / 2) * xp.sqrt(rat) ** 3
    if _USE_SEP_QUAD:
        integrand = partial(_rf_uk_integrands_trans, xp=xp, k=1)
        integral1 = quad(integrand, -1, 1, args=(p, g, is_lower), xp=xp)
        integrand = partial(_rf_uk_integrands_trans, xp=xp, k=2)
        integral2 = quad(integrand, -1, 1, args=(p, g, is_lower), xp=xp)
    else:
        integrand = partial(_rf_uk_integrands_trans, xp=xp)
        integral1, integral2 = quad(integrand, -1, 1, args=(p, g, is_lower), xp=xp)
    return _r_phi_j_from_ints(n1, n2, g, c1 * integral1, c2 * integral2, xp=xp)


def _r_phi_j_quad(n1, n2, *, xp):
    return _r_phi_j_cond(n1, n2, partial(_r_phi_j_quad_inner, xp=xp), xp=xp)


def impedence_quad(n1, n2, *, xp=jnp):
    r_phi, r_j = _r_phi_j_quad(n1, n2, xp=xp)
    return _impedence_from_parts(r_phi, r_j)


# NOTE: Testing Only
def _reflectance_coeff_quad(n1, n2, *, xp=jnp):
    _, r_j = _r_phi_j_quad(n1, n2, xp=xp)
    return _refl_from_parts(r_j)


# NOTE: Testing Only
def _fluence_rate_coeff_quad(n1, n2, *, xp=jnp):
    r_phi, _ = _r_phi_j_quad(n1, n2, xp=xp)
    return _fluen_from_parts(r_phi)


def ecbc_coeffs_quad(n1, n2, *, xp=jnp):
    r_phi, r_j = _r_phi_j_quad(n1, n2, xp=xp)
    return (
        _impedence_from_parts(r_phi, r_j),
        _refl_from_parts(r_j),
        _fluen_from_parts(r_phi),
    )


def _rf_u1_int_exact(g, p, *, xp):
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
        + 6 * (-1 + g**2) * (-2 * g**4 * xp.log(p) + (-1 + g**4) * xp.log((-1 + g * p**2) / (-1 + g)))
    ) / (12 * g**2)


def _upper_rf_u2_int_exact(g, p, *, xp):
    return (
        xp.sqrt(g)
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
        + 105
        * (-1 + g) ** 2
        * (1 + g) ** 3
        * (5 + g * (-2 + 5 * g))
        * (xp.atanh(xp.sqrt(g)) - xp.atanh(xp.sqrt(g) * p))
    ) / (280 * math.sqrt(2) * g**2 * xp.sqrt(1 + g) ** 3)


def _lower_rf_u2_int_exact(g, p, *, xp):
    return (
        xp.sqrt(g)
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
        + 105
        * (-1 + g) ** 3
        * (1 + g) ** 2
        * (5 + g * (2 + 5 * g))
        * p**3
        * (-1 + g * p**2)
        * (xp.atanh(xp.sqrt(g)) - xp.atanh(xp.sqrt(g) * p))
    ) / (280 * math.sqrt(2) * xp.sqrt(1 - g) ** 3 * g**2 * p**3 * (-1 + g * p**2))


def _r_phi_j_exact_inner(n1, n2, *, xp):
    is_lower = n1 < n2
    p = _p_expr(n1, n2, xp=xp)
    g = _g_expr(n1, n2)
    rf_u1_int = _rf_u1_int_exact(g, p, xp=xp) / xp.where(is_lower, (1 - g), (1 + g))
    # FIXME: use either a lazy_where or a unified
    if xp is jnp and jnp.ndim(is_lower) == 0:
        rf_u2_int = cond(is_lower, partial(_lower_rf_u2_int_exact, xp=xp), partial(_upper_rf_u2_int_exact, xp=xp), g, p)
    else:
        rf_u2_int = xp.where(is_lower, _lower_rf_u2_int_exact(g, p, xp=xp), _upper_rf_u2_int_exact(g, p, xp=xp))
    return _r_phi_j_from_ints(n1, n2, g, rf_u1_int, rf_u2_int, xp=xp)


# FIXME: numerical issues in the range 0.99 < n1/n2 < 1.005
def _r_phi_j_exact(n1, n2, *, xp):
    return _r_phi_j_cond(n1, n2, partial(_r_phi_j_exact_inner, xp=xp), xp=xp)


def impedence_exact(n1, n2, *, xp=jnp):
    r_phi, r_j = _r_phi_j_exact(n1, n2, xp=xp)
    return _impedence_from_parts(r_phi, r_j)


# NOTE: Testing Only
def _reflectance_coeff_exact(n1, n2, *, xp=jnp):
    _, r_j = _r_phi_j_exact(n1, n2, xp=xp)
    return _refl_from_parts(r_j)


# NOTE: Testing Only
def _fluence_rate_coeff_exact(n1, n2, *, xp=jnp):
    r_phi, _ = _r_phi_j_exact(n1, n2, xp=xp)
    return _fluen_from_parts(r_phi)


def ecbc_coeffs_exact(n1, n2, *, xp=jnp):
    r_phi, r_j = _r_phi_j_exact(n1, n2, xp=xp)
    return (
        _impedence_from_parts(r_phi, r_j),
        _refl_from_parts(r_j),
        _fluen_from_parts(r_phi),
    )
