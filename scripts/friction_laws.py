r"""Basal friction laws shared by the SSA and hybrid inversions and forwards.

Both laws take the friction coefficient ``C = C_w0 * exp(theta * grounded)``,
where ``C_w0`` is the driving-stress balance anchor from
``physical_prior.py``, so theta is the same log-deviation under either law and
the two share one prior mean. ``phi`` below is the normalized effective pressure
``N / (rho_I g h)`` (1 on grounded ice far above flotation, 0 afloat).

Weertman (the repo's existing law):

    tau_W = C phi U^(1/m),    U = sqrt(|u|^2 + u_min^2)

(u_min = 1 m/yr keeps the stress finite where the ice is stagnant, as in ISMIP7;
icepack's own bed_friction is singular at u = 0).

Regularized Coulomb (the ISMIP7 ``regularized_coulomb`` law,
~/projects/ismip7/icepack2_tools/dual_friction.py): Weertman at low speed,
capped at the Coulomb yield stress ``tau_c = c0 N`` at high speed,

    tau_b = tau_W tau_c / (tau_W + tau_c).

icepack minimizes an action, so the RC law enters through its potential
``P(U) = int_0^U tau_b(s) ds``. For m = 3, with ``X = tau_W / tau_c`` (the phi
factors cancel: ``X = C U^(1/3) / (c0 rho_I g h)``, finite at flotation),

    P = C phi U^(4/3) G(X),   G(X) = 3 (X^3/3 - X^2/2 + X - ln(1+X)) / X^4,

with ``G -> 3/4`` as X -> 0 (the Weertman action) and ``P -> tau_c U`` as
X -> infinity (Coulomb). Below X = 0.1, G uses its Taylor series
3 sum_k (-X)^k / (k+4) to avoid cancellation in the closed form.
"""
from firedrake import Constant, conditional, inner, ln, lt, max_value, sqrt
from icepack.constants import (
    gravity as g, ice_density as ρ_I, water_density as ρ_W,
    weertman_sliding_law as m,
)

assert m == 3.0, "the RC potential below is derived for m = 3"

RC_C0 = 0.5          # Coulomb yield coefficient tau_c = c0 N (ISMIP7 default)
U_MIN = 1.0          # m/yr, sliding-speed regularization
_X_SERIES = 0.1      # below this, G(X) uses its Taylor series
_N_SERIES = 12       # series terms; truncation error < 1e-13 at X = 0.1


def flotation_factor(h, s):
    r"""phi = N / (rho_I g h) with ocean-connected water pressure."""
    return 1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h)


def _G(X):
    series = 3 * sum((-1) ** k * X**k / (k + 4) for k in range(_N_SERIES))
    Xs = max_value(X, Constant(_X_SERIES))
    closed = 3 * (Xs**3 / 3 - Xs**2 / 2 + Xs - ln(1 + Xs)) / Xs**4
    return conditional(lt(X, _X_SERIES), series, closed)


def sliding_speed(u):
    return sqrt(inner(u, u) + Constant(U_MIN) ** 2)


def weertman(**kwargs):
    u, h, s, C = (kwargs[k] for k in ("velocity", "thickness", "surface", "friction"))
    return m / (m + 1) * C * flotation_factor(h, s) * sliding_speed(u) ** (1 / m + 1)


def make_regularized_coulomb(c0=RC_C0):
    c0 = Constant(c0)

    def regularized_coulomb(**kwargs):
        u, h, s, C = (kwargs[k] for k in ("velocity", "thickness", "surface", "friction"))
        U = sliding_speed(u)
        X = C * U ** (1 / m) / (c0 * ρ_I * g * max_value(h, Constant(1.0)))
        return C * flotation_factor(h, s) * U ** (1 / m + 1) * _G(X)

    return regularized_coulomb


FRICTION_LAWS = {"weertman": weertman, "rc": make_regularized_coulomb()}
