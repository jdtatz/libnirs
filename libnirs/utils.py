from functools import partial
from typing import Sequence

import numpy as np
from llvmlite import ir
from numba import jit as _jit
from numba import vectorize
from numba.core import cgutils, types
from numba.core.base import BaseContext as CodegenContext
from numba.core.typing import BaseContext as TypingContext
from numba.core.typing import Signature
from numba.extending import intrinsic
from numpy import sqrt, minimum

jit = partial(_jit, nopython=True, nogil=True, error_model="numpy")


@intrinsic
def _fma(typingctx: TypingContext, a: types.Type, b: types.Type, c: types.Type):
    # check for accepted types
    args = a, b, c
    if all(isinstance(ty, types.Float) for ty in args):
        # create the expected type signature
        fty = max(args)
        sig = fty(fty, fty, fty)

        # defines the custom code generation
        def codegen(context: CodegenContext, builder: ir.IRBuilder, signature: Signature, args: Sequence[ir.Value]):
            # llvm IRBuilder code here
            return builder.fma(*args)

        return sig, codegen


@vectorize(["f4(f4, f4, f4)", "f8(f8, f8, f8)"])
def fma(a, b, c):
    return _fma(a, b, c)


@intrinsic
def map_tuples(typingctx: TypingContext, map_fn: types.Type, *tuples: types.Type):
    # check for accepted types
    if isinstance(map_fn, types.Callable) and all(isinstance(t, types.BaseTuple) for t in tuples):
        fn_sigs = [map_fn.get_call_type(typingctx, tys, {}) for tys in zip(*(t.types for t in tuples))]
        # create the expected type signature
        result_type = types.BaseTuple.from_types([sig.return_type for sig in fn_sigs])
        sig = result_type(map_fn, types.StarArgTuple(tuples))

        # defines the custom code generation
        def codegen(context: CodegenContext, builder: ir.IRBuilder, signature: Signature, args: Sequence[ir.Value]):
            # llvm IRBuilder code here
            map_fn_ty, _ = signature.args
            funcs = (context.get_function(map_fn_ty, sig) for sig in fn_sigs)
            _, tuples = args
            tuples = cgutils.unpack_tuple(builder, tuples)
            elems = (func(builder, [builder.extract_value(t, i) for t in tuples]) for i, func in enumerate(funcs))
            return context.make_tuple(builder, signature.return_type, elems)

        return sig, codegen


_x_kr21 = (
    -9.956571630258080807355272806890028e-01,
    -9.739065285171717200779640120844521e-01,
    -9.301574913557082260012071800595083e-01,
    -8.650633666889845107320966884234930e-01,
    -7.808177265864168970637175783450424e-01,
    -6.794095682990244062343273651148736e-01,
    -5.627571346686046833390000992726941e-01,
    -4.333953941292471907992659431657842e-01,
    -2.943928627014601981311266031038656e-01,
    -1.488743389816312108848260011297200e-01,
    0.0,
    1.488743389816312108848260011297200e-01,
    2.943928627014601981311266031038656e-01,
    4.333953941292471907992659431657842e-01,
    5.627571346686046833390000992726941e-01,
    6.794095682990244062343273651148736e-01,
    7.808177265864168970637175783450424e-01,
    8.650633666889845107320966884234930e-01,
    9.301574913557082260012071800595083e-01,
    9.739065285171717200779640120844521e-01,
    9.956571630258080807355272806890028e-01,
)

_w_kr21 = (
    1.169463886737187427806439606219205e-02,
    3.255816230796472747881897245938976e-02,
    5.475589657435199603138130024458018e-02,
    7.503967481091995276704314091619001e-02,
    9.312545458369760553506546508336634e-02,
    1.093871588022976418992105903258050e-01,
    1.234919762620658510779581098310742e-01,
    1.347092173114733259280540017717068e-01,
    1.427759385770600807970942731387171e-01,
    1.477391049013384913748415159720680e-01,
    1.494455540029169056649364683898212e-01,
    1.477391049013384913748415159720680e-01,
    1.427759385770600807970942731387171e-01,
    1.347092173114733259280540017717068e-01,
    1.234919762620658510779581098310742e-01,
    1.093871588022976418992105903258050e-01,
    9.312545458369760553506546508336634e-02,
    7.503967481091995276704314091619001e-02,
    5.475589657435199603138130024458018e-02,
    3.255816230796472747881897245938976e-02,
    1.169463886737187427806439606219205e-02,
)


@jit
def integrate(func, a, b, divs=1, args=()):
    """Integrate 'func' w/ 'args' over the region (a, b). The region can subdived by 'divs' for better numerical accuracy"""
    skip = (b - a) / divs
    c_1 = skip / 2
    c_2 = c_1 + a
    integrator = func(c_1 * _x_kr21[0] + c_2, *args) * _w_kr21[0]
    for i in range(1, len(_x_kr21)):
        integrator += func(c_1 * _x_kr21[i] + c_2, *args) * _w_kr21[i]
    for j in range(1, divs):
        c_2 += skip
        for i in range(len(_x_kr21)):
            integrator += func(c_1 * _x_kr21[i] + c_2, *args) * _w_kr21[i]
    return c_1 * integrator


@jit
def gen_impedance(n):
    if n <= 1:
        return 3.084635 - 6.531194 * n + 8.357854 * n * n - 5.082751 * n**3 + 1.171382 * n**4
    return (
        504.332889
        - 2641.00214 * n
        + 5923.699064 * n * n
        - 7376.355814 * n**3
        + 5507.53041 * n**4
        - 2463.357945 * n**5
        + 610.956547 * n**6
        - 64.8047 * n**7
    )


@jit
def _gen_reflectance_coeff(u, n2):
    s = sqrt(u**2 + n2 - 1)
    Rs = ((u - s) / (u + s)) ** 2
    Rp = ((s - n2 * u) / (s + n2 * u)) ** 2
    Rfres = (Rs + Rp) / 2
    return 3 * (1 - Rfres) * u**2 / 2


@jit
def _gen_fluence_rate_coeff(u, n2):
    s = sqrt(u**2 + n2 - 1)
    Rs = ((u - s) / (u + s)) ** 2
    Rp = ((s - n2 * u) / (s + n2 * u)) ** 2
    Rfres = (Rs + Rp) / 2
    return (1 - Rfres) * u / 2


@jit
def gen_coeffs(n, n_ext):
    if n == n_ext:
        return 1.0, 0.5, 0.25
    n2 = (n_ext / n)**2
    int_u = minimum(1.0, n2)
    lb = sqrt(1 - int_u)
    return (
        gen_impedance(n / n_ext),
        integrate(_gen_reflectance_coeff, lb, 1.0, 10, (n2,)),
        integrate(_gen_fluence_rate_coeff, lb, 1.0, 10, (n2,)),
    )


@jit
def _qrng_phi(d, tol=1e-6):
    l, u = 1.0, 2.0
    while True:
        mid = (l + u) / 2
        f_mid = mid ** (d + 1) - mid - 1
        if f_mid == 0 or (u - l) < 2 * tol:
            return mid
        elif f_mid < 0:
            l = mid
        else:
            u = mid


@jit
def qrng(ndim, seed=None):
    g = _qrng_phi(ndim)
    alpha = g ** -np.arange(1, ndim + 1)
    if seed is None:
        state = np.full(shape=ndim, fill_value=0.5, dtype=np.float64)
    else:
        state = seed
    while True:
        state += alpha
        state %= 1.0
        yield state.copy()
