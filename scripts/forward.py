r"""Prognostic forward run from a MAP state, for any (model, friction) configuration.

Successor to run_forward.py, which was wired to one hard-coded MAP file, one
configuration and a direct MUMPS solve for every step. This one takes the MAP by
name, builds the model through problem.build_problem (so the forward operator is
byte-for-byte the one the inversion saw), and takes its solver stack from
forward_solvers.py. Those presets are ported from ismip7, which turns out to
have no custom preconditioner at all - every solve there is a direct MUMPS
factorisation - so what transfers is its robustness stack: nleqerr line search,
MUMPS null-pivot handling, a trust-region rescue and a dt subcycling ladder.

    A = A_prior exp(theta_A),    C = C_w0 exp(theta_C * grounded)

Each step advances the thickness with the mass-transport solve, recomputes the
surface from flotation, then re-solves the diagnostic velocity. The climate
forcing is ISMIP7's (ismip7_forcing.py): RACMO SMB climatology plus the ESM's
re-referenced SMB anomaly, and Burgard et al. (2022) quadratic mixed-slope melt
from the ESM's ocean thermal forcing with the per-basin calibrated K. No
apparent or reference mass balance is added: the MAP state is not in balance
and its imbalance is part of the projection. The data-derived melt coefficient
and the floating extent are frozen per step and kept off the tape; so, with
--melt-slope frozen, is the slope of the ice base the melt multiplies (taped, it
makes the adjoint unstable; see advance()).

    python scripts/forward.py --model hybrid --friction weertman --T 50
    python scripts/forward.py --model hybrid --friction weertman --T 1 \
        --solver lu --map sweep_hybrid_weertman_d1e-05_L2km      # smoke test
    python scripts/forward.py --model ssa --friction rc --T 50 --gradient

With --gradient the run is taped through tlm_adjoint and dVAF/dtheta is computed
by adjoint at the end, which is what uq_vaf.py consumes. Without it the run is
untaped and much cheaper, which is what you want for plain projections and for
timing solver presets.

Writes results/forward_<name>_<T>yr.h5 (gradients, final state) and
results/forward_<name>_<T>yr_ts.npz (VAF and flux time series), <name> being
the --map value: the --name the inversion ran with, whose MAP is read from
mesh/map_<name>.h5 where invert.py put it.
"""
import argparse
import time
from pathlib import Path

import firedrake
import icepack
import numpy as np
from firedrake import (
    CheckpointFile, Constant, Function, assemble, conditional, dx, inner, max_value,
    sqrt,
)
from firedrake.adjoint import *  # noqa: F401,F403  (tlm_adjoint needs the annotation hooks)
from icepack.constants import gravity as g, ice_density as ρ_I, water_density as ρ_W
from tlm_adjoint.firedrake import (
    Functional, compute_gradient, reset_manager, start_manager, stop_manager,
)

from forward_solvers import SOLVERS, SUBCYCLES, describe_solver
from ismip7_forcing import Ismip7Forcing, sin_slope
from physical_prior import MESH_FILE
from problem import build_problem, tag

DATA = Path(__file__).resolve().parent.parent

DT = 1.0 / 12.0         # yr; monthly steps
H_MIN = 10.0            # m, thickness floor
GT_PER_M3 = 917.0 / 1e12


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=["ssa", "hybrid"], required=True)
    ap.add_argument("--friction", required=True)
    ap.add_argument("--esm", default="CESM2-WACCM", help="ISMIP7 forcing model")
    ap.add_argument("--scenario", default="ssp585", help="ISMIP7 forcing scenario")
    ap.add_argument("--start-year", type=int, default=2015,
                    help="calendar year of t = 0; the forcing of year Y is applied over "
                         "model time [Y - start, Y - start + 1)")
    ap.add_argument("--no-apres", action="store_true",
                    help="hybrid: the MAP came from a velocity-only inversion")
    ap.add_argument("--map", default=None,
                    help="the --name the inversion ran with (default: <tag>, the production "
                         "MAP), read from mesh/map_<name>.h5; sweep points are "
                         "sweep_<tag>_d<delta>_L<ell>km<run suffix>")
    ap.add_argument("--T", type=float, default=50.0, help="run length (years)")
    ap.add_argument("--dt", type=float, default=DT, help="timestep (years)")
    ap.add_argument("--solver", default="ismip7", choices=sorted(SOLVERS),
                    help="solver/preconditioner preset from forward_solvers.py")
    ap.add_argument("--init-geometry", default="consistent",
                    choices=["consistent", "asis"],
                    help="consistent (default): replace the observed surface with "
                         "compute_surface(h, b) BEFORE the run and re-solve the "
                         "velocity there, so the model's own flotation relation is "
                         "imposed once at t=0 instead of as a step-1 shock; asis: "
                         "start from the observed surface")
    ap.add_argument("--transport", default=None,
                    choices=["lax-wendroff", "implicit-euler"],
                    help="override the preset's mass-transport scheme; the presets "
                         "use icepack's default, Lax-Wendroff")
    ap.add_argument("--gradient", action="store_true",
                    help="tape the run and compute dVAF/dtheta by adjoint (expensive)")
    ap.add_argument("--gradient-times", type=float, nargs="+", default=[],
                    help="with --gradient, also differentiate VAF at these times (years); "
                         "every functional shares the one adjoint pass, and uq_vaf.py turns "
                         "them into sigma_post(t) and the uncertainty of the loss rate")
    ap.add_argument("--melt-slope", choices=["frozen", "taped"], default="frozen",
                    help="with --gradient, whether the ice-base slope in the melt law is "
                         "differentiated (taped) or frozen per step like the melt coefficient")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.apres = args.model == "hybrid" and not args.no_apres
    args.tag = tag(args.model, args.friction, args.apres)
    args.map = args.map or args.tag
    return args


def vaf_form(h, s, b):
    r"""Volume above flotation as a UFL form: integrand is the height above flotation."""
    s_float = b + Constant(float(ρ_W / ρ_I)) * max_value(-b, Constant(0.0))
    return max_value(s - s_float, Constant(0.0))


def lift_geometry(h, b, Q):
    r"""Clamp the thickness, then re-derive the surface from it. The only way
    geometry is allowed to change.

    ismip7 funnels every thickness change through one such routine
    (simulation.py:1463-1494 ``_lift_h``) and re-derives s there, which is why a
    stale or smoothed surface cannot exist in its forward. The order matters: the
    clamp is applied first and the surface derived after, as in
    simulation.py:490-492 and gia-icepack ase_model.py:106-110.
    """
    h = Function(Q).interpolate(max_value(h, Constant(H_MIN)))
    return h, icepack.compute_surface(thickness=h, bed=b)



def main():
    args = parse_args()
    t_start = time.time()
    n_steps = max(1, int(round(args.T / args.dt)))
    out = Path(args.out) if args.out else (
        DATA / "results" / f"forward_{args.map}_{args.T:g}yr.h5")
    out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print(f"Prognostic forward [{args.tag}] from {args.map}", flush=True)
    print(f"  forcing: ISMIP7 {args.esm} {args.scenario} from {args.start_year}; no "
          f"reference balance, so the run starts from the state's own imbalance", flush=True)
    print(f"  T={args.T:g} yr, {n_steps} steps of dt={args.dt:.4f} yr, "
          f"{'taped (adjoint VAF gradient)' if args.gradient else 'untaped'}", flush=True)
    print(f"  solver preset '{args.solver}': {describe_solver(args.solver)}", flush=True)
    print("=" * 72, flush=True)

    preset = SOLVERS[args.solver]
    P = build_problem(args.model, args.friction, args.apres,
                      solver_parameters=preset["diagnostic"],
                      prognostic_parameters=preset["prognostic"],
                      prognostic_solver_type=args.transport or preset["transport"])
    Q, mesh_2d, Q_2d = P["Q"], P["mesh_2d"], P["Q_2d"]

    with CheckpointFile(str(MESH_FILE), "r") as chk:
        b_2d = chk.load_function(chk.load_mesh(), "bed")
    if args.model == "ssa":
        # b_2d lives on the mesh this file just loaded, not the problem's own copy;
        # same file, so same dof order, but UFL refuses to mix the two meshes.
        b = Function(Q, name="bed")
        b.dat.data[:] = b_2d.dat.data_ro
    else:
        b = icepack.utilities.lift3d(b_2d, Q)
    forcing = Ismip7Forcing(Q_2d, esm=args.esm, scenario=args.scenario,
                            start_year=args.start_year)
    print(f"  {forcing.describe()}", flush=True)
    # The hybrid's CG1 x DG0 space has one dof per footprint vertex, in the
    # footprint's order (the MAP controls are copied across the same way), so
    # the per-vertex forcing arrays go straight into Q.
    assert Q.dof_dset.size == Q_2d.dof_dset.size

    def on_Q(values, name):
        r"""A fresh field on Q holding ``values``; never modified after use, so the tape stays valid."""
        f = Function(Q, name=name)
        f.dat.data[:] = values
        return f

    smb_fields = {}

    def smb_at(t):
        r"""SMB field for the year containing t, one Function per year."""
        yr = forcing.year(t)
        if yr not in smb_fields:
            smb_fields.clear()
            smb_fields[yr] = on_Q(forcing.smb(t), f"smb_{yr}")
        return smb_fields[yr]

    def melt_expr(h, s, coeff):
        r"""Sub-shelf melt rate (m ice/yr) wherever the coefficient is nonzero; masked by the caller."""
        return coeff * sin_slope(h, s)

    with CheckpointFile(str(DATA / "mesh" / f"map_{args.map}.h5"), "r") as chk:
        m = chk.load_mesh()
        θA_2d = chk.load_function(m, "log_fluidity")
        θC_2d = chk.load_function(m, "log_friction")
    θ_A = Function(Q, name="log_fluidity")
    θ_C = Function(Q, name="log_friction")
    θ_A.dat.data[:] = θA_2d.dat.data_ro
    θ_C.dat.data[:] = θC_2d.dat.data_ro
    print(f"MAP controls: theta_A in [{θ_A.dat.data_ro.min():+.2f}, "
          f"{θ_A.dat.data_ro.max():+.2f}], theta_C in [{θ_C.dat.data_ro.min():+.2f}, "
          f"{θ_C.dat.data_ro.max():+.2f}]", flush=True)

    h, s = P["h"].copy(deepcopy=True), P["s"].copy(deepcopy=True)
    # thwaites.h5 carries observed h and s that do not satisfy the model's own
    # flotation relation: compute_surface(h, b) differs from the observed surface
    # by up to +362 m, at 21376 of 67600 nodes by more than 10 m. The inversions
    # run at fixed geometry and never feel it, but the first prognostic step calls
    # compute_surface and imposes the whole difference at once, which multiplies
    # the peak speed by four and blows the run up within three steps. Imposing it
    # at t = 0, before the velocity is primed, keeps the run stable - at the price
    # of a MAP fitted at a surface the forward does not use. Invert on a mesh that
    # is already consistent instead: docs/geometry_consistency.md.
    if args.init_geometry == "consistent":
        h, s_model = lift_geometry(h, b, Q)
        ds = s_model.dat.data_ro - s.dat.data_ro
        print(f"Initial surface replaced by compute_surface(h, b): shift "
              f"[{ds.min():+.1f}, {ds.max():+.1f}] m, |shift| > 10 m at "
              f"{int((np.abs(ds) > 10).sum())} of {ds.size} nodes", flush=True)
        if np.abs(ds).max() > 1.0:
            print("  NOTE: a shift this large means the MAP was inverted at a "
                  "geometry the forward cannot use. Invert on a mesh whose stored "
                  "surface already satisfies the flotation relation "
                  "(scripts/check_geometry.py).", flush=True)
        s = s_model
    h_inflow = h.copy(deepcopy=True)
    u = Function(P["u0"].function_space()).assign(P["forward"](θ_A, θ_C, h=h, s=s))

    def max_speed(u):
        r"""Peak depth-averaged speed: the blow-up detector ismip7 watches."""
        u2 = u if args.model == "ssa" else icepack.depth_average(u)
        return float(Function(Q_2d).interpolate(sqrt(inner(u2, u2))).dat.data.max())

    u_max0 = max_speed(u)
    print(f"Primed the velocity at the MAP (max speed {u_max0:.0f} m/yr)", flush=True)

    A, C = P["fields"](θ_A, θ_C)
    solver = P["solver"]
    rescue = None
    if preset["rescue"] is not None:
        rescue = P["make_solver"](diagnostic=preset["rescue"],
                                  prognostic=preset["prognostic"],
                                  transport=args.transport or preset["transport"])
    n_rescue = n_subcycle = 0

    if args.gradient:
        reset_manager()
        start_manager()
        # A and C must be built from theta ON the tape, or every diagnostic solve
        # sees them as constants and dVAF/dtheta comes out identically zero.
        A, C = P["fields"](θ_A, θ_C)

    Js = []     # (t, VAF(t) functional) at the --gradient-times, on the tape

    def diagnostic(u, h, s):
        r"""Diagnostic solve, escalating to the trust-region solver if it fails.

        ismip7's escalation ladder (simulation.py:1405-1461) in the form classic
        icepack allows: a second FlowSolver carrying the rescue options, rather
        than mutating the live SNES.
        """
        nonlocal n_rescue
        try:
            return solver.diagnostic_solve(velocity=u, thickness=h, surface=s,
                                           fluidity=A, friction=C)
        except firedrake.exceptions.ConvergenceError:
            if rescue is None:
                raise
            n_rescue += 1
            print("    diagnostic failed; trust-region rescue", flush=True)
            return rescue.diagnostic_solve(velocity=u, thickness=h, surface=s,
                                           fluidity=A, friction=C)

    def advance(h, s, u, dt, t0):
        r"""Transport-first substep from model time t0: move the geometry, then re-solve the velocity."""
        # The floating EXTENT is a discrete mask and the melt coefficient comes
        # from data looked up at the current draft, so both are frozen per step
        # and kept off the tape. The slope of the ice base is too with
        # --melt-slope frozen; taped, its linearisation is an explicit, unstabilised
        # advection of the thickness perturbation at speeds ~ the melt coefficient
        # (10^3-10^4 m/yr where the slope is below the cap), and the adjoint grows
        # exponentially in time.
        if args.gradient:
            stop_manager()
        floating = Function(Q).interpolate(
            conditional(1 - ρ_W * g * max_value(0, h - s) / (ρ_I * g * h) < 0.01,
                        Constant(1.0), Constant(0.0)))
        t_mid = t0 + 0.5 * dt
        coeff = on_Q(forcing.melt_coefficient(t_mid, s.dat.data_ro - h.dat.data_ro),
                     "melt_coefficient")
        smb = smb_at(t_mid)
        if args.melt_slope == "frozen":
            melt = Function(Q, name="melt_rate").interpolate(melt_expr(h, s, coeff))
        if args.gradient:
            start_manager()
        if args.melt_slope == "taped":
            melt = melt_expr(h, s, coeff)
        # The accumulation is the climate forcing alone. The MAP state is not in
        # balance, and no reference term is added to hide that: the imbalance
        # is part of the projection.
        accum = Function(Q).interpolate(smb - floating * melt)
        h = solver.prognostic_solve(dt, thickness=h, velocity=u,
                                    accumulation=accum, thickness_inflow=h_inflow)
        h, s = lift_geometry(h, b, Q)
        return h, s, diagnostic(u, h, s), (floating, coeff)

    if args.gradient:
        # The first transport step's velocity, re-solved on the tape so it depends
        # on theta. Start from the observed-velocity guess the priming solve used,
        # NOT from the primed (converged) u: the ismip7 preset has snes_stol = 0 and
        # no divergence test, so a solve that starts converged can never meet its
        # relative tolerance and grinds through all snes_max_it iterations.
        u = diagnostic(Function(u.function_space()).assign(P["u0"]), h, s)

    haf = vaf_form(h, s, b)
    vaf0 = float(assemble(haf * dx)) * GT_PER_M3
    times, vaf_ts, melt_ts, umax_ts = [0.0], [vaf0], [0.0], [u_max0]
    print(f"\n  t= 0.00: VAF={vaf0:10.1f} Gt, u_max={u_max0:7.0f} m/yr", flush=True)

    for step in range(1, n_steps + 1):
        t = step * args.dt
        # dt ladder: a step the diagnostic solve cannot get through is retried as
        # m substeps (ismip7 simulation.py:1720-1747). Nothing is committed until
        # a whole step succeeds, so a failed attempt rewinds by construction.
        for m in SUBCYCLES:
            try:
                hh, ss, uu = h, s, u
                for k in range(m):
                    hh, ss, uu, (floating, coeff) = advance(
                        hh, ss, uu, args.dt / m, (step - 1) * args.dt + k * args.dt / m)
            except firedrake.exceptions.ConvergenceError:
                if m == SUBCYCLES[-1]:
                    print(f"  t={t:5.2f}: step failed at every subcycle; stopping",
                          flush=True)
                    break
                print(f"    step failed; retrying as {SUBCYCLES[SUBCYCLES.index(m) + 1]} "
                      f"substeps", flush=True)
                continue
            if m > 1:
                n_subcycle += 1
            h, s, u = hh, ss, uu
            break
        else:
            break

        # Monitoring only: keep it off the tape (icepack's depth_average builds
        # plain Constants that tlm_adjoint's annotated project rejects).
        if args.gradient:
            stop_manager()
        vaf = float(assemble(vaf_form(h, s, b) * dx)) * GT_PER_M3
        melt = float(assemble(floating * melt_expr(h, s, coeff) * dx)) * GT_PER_M3
        u_max = max_speed(u)
        if args.gradient:
            start_manager()
        n_floor = int((h.dat.data_ro <= H_MIN + 1e-9).sum())
        times.append(t); vaf_ts.append(vaf); melt_ts.append(melt); umax_ts.append(u_max)
        if args.gradient and step < n_steps and any(
                abs(t - tg) < 0.5 * args.dt for tg in args.gradient_times):
            J_t = Functional(name=f"VAF_{t:g}yr")
            J_t.assign(vaf_form(h, s, b) * dx)
            Js.append((t, J_t))
        if step % max(1, n_steps // 20) == 0 or step == n_steps:
            print(f"  t={t:5.2f}: VAF={vaf:10.1f} Gt (dVAF={vaf - vaf0:+8.2f}), "
                  f"melt={melt:6.1f} Gt/yr, u_max={u_max:7.0f} m/yr, "
                  f"floor={n_floor:5d}  [{(time.time() - t_start) / 60:.1f} min]",
                  flush=True)
        # ismip7's blow-up signature: the diagnostic velocity roughly doubling per
        # step. Stop early rather than burn hours on a diverging run.
        if u_max > 10.0 * u_max0:
            print(f"  u_max {u_max:.0f} m/yr is {u_max / u_max0:.1f}x the initial "
                  f"{u_max0:.0f}: the forward is diverging, stopping", flush=True)
            break

    result = {"times": np.array(times), "vaf_Gt": np.array(vaf_ts),
              "melt_Gt_per_yr": np.array(melt_ts), "T_years": args.T, "dt": args.dt,
              "map": args.map, "tag": args.tag, "solver": args.solver,
              "forcing": f"ISMIP7 {args.esm} {args.scenario} from {args.start_year}",
              "u_max_m_per_yr": np.array(umax_ts),
              "melt_slope": args.melt_slope if args.gradient else "untaped",
              "n_rescue": n_rescue, "n_subcycled_steps": n_subcycle,
              "steps_completed": len(times) - 1}

    grads = []
    if args.gradient:
        J = Functional(name="VAF_T")
        J.assign(vaf_form(h, s, b) * dx)
        Js.append((times[-1], J))
        print(f"\nAdjoint gradient dVAF/dtheta at t = "
              f"{', '.join(f'{t:g}' for t, _ in Js)} yr ...", flush=True)
        dJ = compute_gradient([j for _, j in Js], (θ_A, θ_C))
        stop_manager()
        for k, ((t, Jt), (dA, dC)) in enumerate(zip(Js, dJ)):
            # The final time keeps the plain names uq_vaf.py has always read.
            sfx = "" if k == len(Js) - 1 else f"_t{t:g}yr"
            gA = Function(Q_2d, name=f"dVAF_dthetaA{sfx}")
            gC = Function(Q_2d, name=f"dVAF_dthetaC{sfx}")
            gA.dat.data[:] = dA.dat.data_ro
            gC.dat.data[:] = dC.dat.data_ro
            grads += [gA, gC]
            print(f"  t={t:5.2f}: |dVAF/dtheta_A| = {np.linalg.norm(gA.dat.data_ro):.4e}, "
                  f"|dVAF/dtheta_C| = {np.linalg.norm(gC.dat.data_ro):.4e}", flush=True)
        result["J_raw_m3"] = float(J)
        result["gradient_times"] = np.array([t for t, _ in Js])

    h_2d = Function(Q_2d, name="thickness_final")
    s_2d = Function(Q_2d, name="surface_final")
    h_2d.dat.data[:] = h.dat.data_ro
    s_2d.dat.data[:] = s.dat.data_ro
    u_2d = u if args.model == "ssa" else icepack.depth_average(u)
    u_2d.rename("velocity_final")
    with CheckpointFile(str(out), "w") as chk:
        chk.save_mesh(mesh_2d)
        for fn in (h_2d, s_2d, u_2d, *grads):
            chk.save_function(fn)
    np.savez(str(out).replace(".h5", "_ts.npz"), **result)
    if n_rescue or n_subcycle:
        print(f"\nrescues: {n_rescue} trust-region, {n_subcycle} subcycled steps",
              flush=True)
    if len(times) - 1 < n_steps:
        print(f"WARNING: stopped after {len(times) - 1}/{n_steps} steps", flush=True)
    print(f"\nVAF {vaf0:.1f} -> {vaf_ts[-1]:.1f} Gt "
          f"({vaf_ts[-1] - vaf0:+.1f} Gt in {args.T:g} yr) in "
          f"{(time.time() - t_start) / 3600:.2f} h wall", flush=True)
    print(f"Saved {out} and {str(out.name).replace('.h5', '_ts.npz')}", flush=True)


if __name__ == "__main__":
    main()
