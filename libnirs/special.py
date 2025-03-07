import jax.numpy as jnp
from jax import custom_jvp


@custom_jvp
def erfcx(x):
    """
    Scaled complementary error function, `exp(x**2) * erfc(x)`.

    Sources:
    * https://stackoverflow.com/a/39777361
    * Shepherd, M. M., & Laframboise, J. G. (1981). Chebyshev Approximation of (1 + 2 x) exp(x2) erfc x in 0 ≤ x < ∞. Mathematics of Computation, 36(153), 249. doi:10.2307/2007742
    """
    # (x,) = promote_args_inexact("erfcx", x)
    a = abs(x)
    p = a + 2.0
    r = 1.0 / p
    q = (a - 2.0) * r
    t = (q + 1.0) * (-2.0) + a
    e = q * (-a) + t
    q = r * e + q
    p = float.fromhex("0x1.f10000p-15")  # 5.92470169e-5
    p = p * q + (float.fromhex("0x1.521cc6p-13"))  #  1.61224554e-4
    p = p * q + (-float.fromhex("0x1.6b4ffep-12"))  # -3.46481771e-4
    p = p * q + (-float.fromhex("0x1.6e2a7cp-10"))  # -1.39681227e-3
    p = p * q + (float.fromhex("0x1.3c1d7ep-10"))  #  1.20588380e-3
    p = p * q + (float.fromhex("0x1.1cc236p-07"))  #  8.69014394e-3
    p = p * q + (-float.fromhex("0x1.069940p-07"))  # -8.01387429e-3
    p = p * q + (-float.fromhex("0x1.bc1b6cp-05"))  # -5.42122945e-2
    p = p * q + (float.fromhex("0x1.4ff8acp-03"))  #  1.64048523e-1
    p = p * q + (-float.fromhex("0x1.54081ap-03"))  # -1.66031078e-1
    p = p * q + (-float.fromhex("0x1.7bf5cep-04"))  # -9.27637145e-2
    p = p * q + (float.fromhex("0x1.1ba03ap-02"))  #  2.76978403e-1
    d = a + 0.5
    r = 1.0 / d
    r = r * 0.5
    q = p * r + r
    e = (p - q) - (q + q) * a + 1.0
    r = e * r + q
    r = jnp.where(a > float.fromhex("0x1.fffffep127"), 0.0, r)
    s = x * x
    d = x * x - s
    e = jnp.exp(s)
    r = jnp.where(x < 0, jnp.where(e > float.fromhex("0x1.fffffep127"), e, e - r + e * (d + d) + e), r)
    return r


@erfcx.defjvp
def _erfcx_jvp(primals, tangents):
    # FRAC_2_SQRT_PI = 2/sqrt(pi)
    FRAC_2_SQRT_PI = 1.12837916709551257389615890312154517
    (x,) = primals
    (t,) = tangents
    w = erfcx(x)
    dw_dx = 2 * x * w - FRAC_2_SQRT_PI
    return w, dw_dx * t
