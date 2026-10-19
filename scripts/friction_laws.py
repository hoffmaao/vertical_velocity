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

Yield-stress control (``schoof``, the law of Joughin's icesheetModels, GitHub
user fastice, ``getFrictionLaw(frictionLaw='schoof')``): Schoof's (2005)
regularized Coulomb form with the critical yield stress as the control and a
fixed transition speed,

    tau_b = tau_c U^(1/m) / (U_0^(m+1)/m + U^(m+1)/m)^(1/(m+1)),   U_0 = SCHOOF_U0,

so ``tau_b -> tau_c`` for U >> U_0 (Coulomb) and ``tau_b -> tau_c (U/U_0)^(1/m)``
below (Weertman). ``friction`` is ``tau_c = tau_c0 exp(theta He) g(HAF)``: no
effective pressure appears, the control scales the whole curve and
``d tau_b / d theta = tau_b`` everywhere, whereas under ``rc`` theta scales
only the Weertman branch and ``d tau_b / d theta = tau_c X / (1 + X)^2`` dies
off on the Coulomb plateau, which is where the inverted theta_C spikes
(figures/diag_coulomb_cap_hybrid_rc_L3km.png). Its potential is closed form,

    P = tau_c [ (U_0^(4/3) + U^(4/3))^(3/4) - U_0 ]    (m = 3).

``g(HAF) = min(1, HAF / GL_THRESH)`` is fastice's grounding-line treatment
(PigForward.py, reduceNearGLBeta): the yield stress falls linearly to zero
over the last GL_THRESH m of height above flotation, which is how a
thinning forward loses basal drag before it floats. It is in the law, so it
follows the model geometry on the tape; the anchor tau_c0 divides it out at
the observed geometry (physical_prior.py), so theta = 0 is still balance.

Grounded gate (all laws): ``He = 0.5 (1 + tanh(HAF / GL_WIDTH))`` on the
control, ISMIP7's smooth Heaviside (icepack2_tools/grounding.py), in place of
the hard phi > 0.01 mask used before 2026-10-12.
"""
from firedrake import Constant, conditional, inner, ln, lt, max_value, min_value, sqrt, tanh
from icepack.constants import (
    gravity as g, ice_density as ρ_I, water_density as ρ_W,
    weertman_sliding_law as m,
)

assert m == 3.0, "the RC potential below is derived for m = 3"

RC_C0 = 0.5          # Coulomb yield coefficient tau_c = c0 N (ISMIP7 default; the rc law)
SCHOOF_U0 = 300.0    # m/yr, Weertman-to-Coulomb transition speed (the schoof law; fastice uThresh)
GL_THRESH = 41.0     # m HAF over which the schoof yield stress tapers to zero (fastice GLThresh)
GL_WIDTH = 10.0      # m HAF, width of the smooth Heaviside grounded gate (ISMIP7 GL_WIDTH)
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


def height_above_flotation(h, s):
    r"""HAF = h phi: the thickness in excess of flotation, zero on a hydrostatic shelf."""
    return h * flotation_factor(h, s)


def grounded_heaviside(h, s, width=GL_WIDTH):
    r"""He = 0.5 (1 + tanh(HAF / width)): the smooth grounded gate on the friction control."""
    return 0.5 * (1 + tanh(height_above_flotation(h, s) / Constant(width)))


def gl_taper(h, s, thresh=GL_THRESH):
    r"""g = min(1, max(HAF, 0) / thresh): the yield-stress taper of fastice's reduceNearGLBeta."""
    return min_value(Constant(1.0), max_value(height_above_flotation(h, s), Constant(0.0)) / Constant(thresh))


def make_schoof(u0=SCHOOF_U0, thresh=GL_THRESH):
    u0 = Constant(u0)
    p = 1 / m + 1

    def schoof(**kwargs):
        u, h, s, τ_c = (kwargs[k] for k in ("velocity", "thickness", "surface", "friction"))
        U = sliding_speed(u)
        return τ_c * gl_taper(h, s, thresh) * ((u0**p + U**p) ** (1 / p) - u0)

    return schoof


FRICTION_LAWS = {"weertman": weertman, "rc": make_regularized_coulomb(), "schoof": make_schoof()}
# The prior-mean field each law's control is a log-deviation from (physical_prior.py).
ANCHOR = {"weertman": "C_w0", "rc": "C_w0", "schoof": "tau_c0"}
