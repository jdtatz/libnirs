from functools import partial
from collections.abc import Callable
from typing import Concatenate, Literal, TypeAlias, SupportsFloat

import jax
import jax.numpy as jnp
from jax import Array
from scipy.integrate import cubature, quad_vec
from jax.custom_derivatives import CustomVJPPrimal, zero_from_primal

try:
    from quadax import GaussKronrodRule, adaptive_quadrature

    _HAVE_QUADAX = True
except ImportError:
    _HAVE_QUADAX = False


def _get_quadax_rule(rule: str):
    if rule.startswith("gk"):
        order = int(rule.removeprefix("gk"))
        return GaussKronrodRule(order=order)
    elif rule == "genz-malik":
        raise NotImplementedError("quadax doesn't support n-dimensional genz-malik cubature yet")
    else:
        raise ValueError(rule)


def _get_quadax_integrator(rule: str):
    qrule = _get_quadax_rule(rule)

    def _wrapped_quad(f, a, b, args):
        y, _info = adaptive_quadrature(qrule, f, (a, b), args=args)
        return y

    return _wrapped_quad


# TODO: include 0-dim Arrays, and maybe size=1 Arrays?
RealScalar: TypeAlias = int | float | SupportsFloat


def quad(
    # TODO: replace `...` with ParamSpec once it's valid to use `P.args` anywhere
    integrand: Callable[Concatenate[Array, ...], Array],
    a: RealScalar,
    b: RealScalar,
    *,
    # NOTE: this is `P.args`, but it's invalid to use `ParamSpecArgs` outside of annotating `*args`
    args=(),
    rule: Literal["gk21", "gk15"] = "gk21",
    _use_quadax: bool = _HAVE_QUADAX,
    _use_custom_quadrature: bool = True,
    xp=jnp,
) -> Array:
    """Compute a definite integral.

    Integrate `integrand` from `a` to `b` (possibly infinite interval)
    """
    # FIXME: this might be a mistake later if the array-api officaly adds quadrature, but with a diffrent signature
    if hasattr(xp, "quad"):
        integral, _info = xp.quad(integrand, a, b, args=args, rule=rule)
    elif _use_quadax:
        assert _HAVE_QUADAX
        integrator = _get_quadax_integrator(rule)
        wrapped_integrator = partial(_custom_quadrature, integrator) if _use_custom_quadrature else integrator
        integral = wrapped_integrator(integrand, a, b, args)
    else:
        integral, _err, _info = quad_vec(integrand, a, b, args=args, quadrature=rule)
    return integral


def dblquad(
    # TODO: replace `...` with ParamSpec once it's valid to use `P.args` anywhere
    integrand: Callable[Concatenate[Array, Array, ...], Array],
    a: RealScalar,
    b: RealScalar,
    g: RealScalar,
    h: RealScalar,
    *,
    # NOTE: this is `P.args`, but it's invalid to use `ParamSpecArgs` outside of annotating `*args`
    args=(),
    rule: Literal["gk21", "gk15", "genz-malik"] = "gk21",
    _use_quadax: bool = _HAVE_QUADAX,
    _use_custom_quadrature: bool = True,
    xp=jnp,
) -> Array:
    """Compute a double integral.

    Return the double (definite) integral of `integrand(y, x)` from x = a..b and y = g..h.
    """
    # FIXME: this might be a mistake later if the array-api officaly adds quadrature, but with a diffrent signature
    if hasattr(xp, "dblquad"):
        integral, _info = xp.dblquad(integrand, a, b, g, h, args=args, rule=rule)
    elif _use_quadax:
        assert _HAVE_QUADAX
        integrator = _get_quadax_integrator(rule)
        wrapped_integrator = partial(_custom_quadrature, integrator) if _use_custom_quadrature else integrator

        def inner_integrand(x: Array, g: RealScalar, h: RealScalar, *args):
            return wrapped_integrator(integrand, g, h, (x, *args))

        integral = wrapped_integrator(inner_integrand, a, b, (g, h, *args))
    else:

        def wrapped_integrand(v: Array, *args):
            # FIXME: not strictly correct
            x = v[..., 0][..., xp.newaxis]
            y = v[..., 1][..., xp.newaxis]
            return integrand(y, x, *args)

        res = cubature(wrapped_integrand, xp.stack([a, g]), xp.stack([b, h]), args=args, rule=rule)
        integral = res.estimate
    return integral


@partial(jax.custom_vjp, nondiff_argnums=[0, 1])
def _custom_quadrature(integrate, f, a, b, args):
    return integrate(f, a, b, args)


def _custom_quadrature_vjp_fwd(integrate, f, a: CustomVJPPrimal, b: CustomVJPPrimal, args: tuple[CustomVJPPrimal, ...]):
    integral = integrate(f, a.value, b.value, args=tuple(arg.value for arg in args))
    aux = (a.value, a.perturbed), (b.value, b.perturbed), tuple((arg.value, arg.perturbed) for arg in args)
    return integral, aux


def vjp_single(f, argnum, *primals):
    pre, post = primals[:argnum], primals[1 + argnum :]

    def inner(pre, post, primal):
        return f(*pre, primal, *post)

    return jax.vjp(jax.tree_util.Partial(inner, pre, post), primals[argnum])


def _custom_quadrature_vjp_bwd(integrate, f, aux, grad):
    (a, a_p), (b, b_p), args_t = aux
    if args_t:
        args, args_p = zip(*args_t)
    else:
        args = args_p = ()

    grad_a = (-grad * f(a, *args)) if a_p else zero_from_primal(a, True)
    grad_b = (grad * f(b, *args)) if b_p else zero_from_primal(b, True)

    grad_args = []
    for i, i_p in enumerate(args_p):

        def inner(x, *args):
            _primals_out, f_vjp = vjp_single(f, 1 + i, x, *args)
            (g_arg,) = f_vjp(grad)
            return g_arg

        grad_arg = (integrate(inner, a, b, args=args)) if i_p else zero_from_primal(args[i], True)
        grad_args.append(grad_arg)

    return grad_a, grad_b, tuple(grad_args)


_custom_quadrature.defvjp(_custom_quadrature_vjp_fwd, _custom_quadrature_vjp_bwd, symbolic_zeros=True)
