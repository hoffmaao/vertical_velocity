r"""Sweep the prior hyperparameters (delta, ELL) over every model configuration.

The prior is

    R(theta) = 0.5 (delta int theta^2 + gamma int |grad theta|^2),  gamma = delta ELL^2,

so delta sets the prior strength and ELL its correlation length. A 1D sweep in
ELL at fixed delta traces the classic L-curve but says nothing about the prior
strength, which is the axis that matters for UQ: Recinos et al. (2023) warn that
the L-curve elbow gives a prior too strong for honest posterior variance, and
that the fix is to relax delta at fixed ELL. This driver therefore sweeps the
full (delta, ELL) surface.

Each point is an independent cold-started MAP inversion (invert.py from the
physical prior mean, not warm-started from a neighbour), so the points are
directly comparable and the sweep parallelizes trivially. Every run gets its own
--name, which keys its checkpoint and outputs: without that, sweep points
overwrite the production MAP and resume from each other's iterates.

    python scripts/sweep_regularization.py --dry-run
    python scripts/sweep_regularization.py --workers 12
    python scripts/sweep_regularization.py --workers 12 --skip-existing

The fluidity and friction priors are independent (invert.py --delta-A/--ell-A,
--delta-C/--ell-C). --mode picks which (delta, ELL) the grid varies:

  shared    both priors move together (the default, and the original sweep)
  friction  the friction prior moves; the fluidity prior is held at
            --fixed-delta/--fixed-ell
  fluidity  the fluidity prior moves; the friction prior is held fixed

    python scripts/sweep_regularization.py --mode friction --configs hybrid_weertman

The two single-control modes held at the same fixed values cross each other, and
the shared sweep, at the point where both priors equal that value.

Writes results/map_sweep_<tag>_d<delta>_L<ell>km.json per point (plus the usual
h5 and history), a log per point under output/logs/sweep/, and a manifest at
results/sweep_manifest.json recording the status and wall time of every job.
Single-control points are named sweep_<tag>_C_d<delta>_L<ell>km_Afix_d<delta>_L<ell>km
(friction; _A_..._Cfix_... for fluidity), and their manifests carry the mode and
the fixed prior.

VV_MESH (physical_prior.py) selects the mesh, and the point names, logs and
manifest carry it as a suffix whenever it is not the default ``thwaites``:

    VV_MESH=thwaites_fixed python scripts/sweep_regularization.py --workers 12

writes map_sweep_<tag>_d<delta>_L<ell>km_thwaites_fixed.* and
sweep_manifest_thwaites_fixed.json. Without the suffix a sweep on a new mesh
would overwrite the old one's results and resume from its checkpoints, and
--skip-existing would skip every point the old mesh had already converged.
VV_TAU_SMOOTH (the anchor's driving-stress smoothing length, physical_prior.py)
adds a second suffix the same way, e.g. _thwaites_fixed_tau5km. plot_lsurface.py
reads the same two variables to select which sweep it plots.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent
SCRIPTS = DATA / "scripts"
LOG_DIR = DATA / "output" / "logs" / "sweep"
MESH_NAME = os.environ.get("VV_MESH", "thwaites")   # same default as physical_prior.py
TAU_D_SMOOTH = float(os.environ.get("VV_TAU_SMOOTH", 2000.0))   # physical_prior.py default
RUN_SUFFIX = (("" if MESH_NAME == "thwaites" else f"_{MESH_NAME}")
              + ("" if TAU_D_SMOOTH == 2000.0 else f"_tau{TAU_D_SMOOTH / 1000:g}km"))
PYTHON = sys.executable
sys.path.insert(0, str(SCRIPTS))
from prior_precond import DEFAULT_FORM, FORMS  # noqa: E402

# Name infix per prior form (prior_precond.py): the bilaplacian points carry
# "_bilap" so they never collide with the h1 points swept before 2026-10-14.
FORM_INFIX = {"bilaplacian": "_bilap", "h1": ""}

# The strong-prior end (1e-3, 1e-2) and the long correlation lengths (10, 20 km)
# were added once the first sweeps showed straight log-log L-curves with no
# corner inside 1e-7..1e-4 x 0.5..5 km. The weak end is not extended: at 1e-7
# chi^2_vel is already below 1 and theta_A runs away (min near -21), so weaker
# shared priors only map further into the unphysical regime.
DELTAS = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7]
ELLS = [500.0, 1000.0, 2000.0, 3000.0, 5000.0, 10000.0, 20000.0]

# (model, friction, no_apres), ordered by the wall time of the production run at
# (delta, ELL) = (1e-5, 2 km) so the long jobs are dispatched first and the tail
# of the schedule is cheap.
#
# Scoped to the ApRES-assimilating hybrids and the SSA runs. The velocity-only
# hybrids (hybrid_*_noapres) are deliberately not swept: they exist as the
# controls for the ApRES-vs-no-ApRES comparison at the production
# hyperparameters, which does not need an L-surface of its own. Re-add the two
# ("hybrid", *, True) entries here to bring them back.
CONFIGS = [
    ("hybrid", "weertman", False),   # 2.40 h, ApRES assimilated
    ("ssa", "weertman", False),      # 2.12 h
    ("hybrid", "rc", False),         # 0.90 h, ApRES assimilated
    ("ssa", "rc", False),            # 0.43 h
]


# Points the sweep does not run. The prior is too weak to constrain the
# controls there, so the MAP is unphysical (theta_A down to -77, theta_C to
# -122, i.e. zero fluidity and zero friction on near-floating nodes where the
# data gradient vanishes) and the Newton solve fails on most trial steps. See
# results/_excluded/README.md.
EXCLUDE = {("hybrid", "weertman", False, 1.0e-7, 500.0)}


# invert.py runs scipy's L-BFGS-B with ftol = gtol = 0, so apart from its own
# tests ("converged: ...") and the iteration budget, scipy stops only on these.
# In every sweep point that has hit them (the first 14 friction-sweep points,
# 2026-09-25) the cause was the same: theta ran away, the Newton solve failed on
# most trial steps, and invert.py's rejected-step penalty, paired with the
# previous gradient, is an inconsistent pair the line search cannot recover
# from. The run is deterministic, so rerunning reproduces the stall.
STALL_STATUSES = ("CONVERGENCE: RELATIVE REDUCTION OF F", "ABNORMAL")


def stalled(status):
    r"""True for a MAP that stopped on a line-search stall from failed forward solves.

    Such a point is treated as outside the physical range of the prior, like
    EXCLUDE but found by running it: --skip-existing leaves it alone (unless
    --retry-stalled), and plot_lsurface.py hatches it and keeps it out of the
    L-curves.
    """
    return status.startswith(STALL_STATUSES)


def config_tag(model, friction, no_apres):
    r"""Same naming as problem.tag, without importing firedrake into the driver."""
    apres = model == "hybrid" and not no_apres
    return f"{model}_{friction}" + ("_noapres" if model == "hybrid" and not apres else "")


MODES = ("shared", "friction", "fluidity")
SWEPT = {"friction": "C", "fluidity": "A"}          # control whose prior the grid varies
HELD = {"friction": "A", "fluidity": "C"}
CONFIG_TAGS = [config_tag(*c) for c in CONFIGS] + [
    config_tag(m, f, True) for m, f, _ in CONFIGS if m == "hybrid"]


def point_label(delta, ell):
    return f"d{delta:.0e}_L{ell / 1000:g}km"


def mode_infix(mode):
    r"""Name fragment between the config tag and the swept point."""
    if mode == "shared":
        return ""
    return f"_{SWEPT[mode]}"


def mode_postfix(mode, fixed=None):
    r"""Name fragment after the swept point: the (delta, ell) the other control is held at."""
    if mode == "shared":
        return ""
    return f"_{HELD[mode]}fix_{point_label(*fixed)}"


def point_name(model, friction, no_apres, delta, ell, mode="shared", fixed=None,
               prior_form="h1"):
    return (f"sweep_{config_tag(model, friction, no_apres)}{mode_infix(mode)}"
            f"{FORM_INFIX[prior_form]}_{point_label(delta, ell)}{mode_postfix(mode, fixed)}"
            f"{RUN_SUFFIX}")


def manifest_path(mode, fixed=None, configs=None):
    r"""One manifest per driver: two drivers that differ only in --configs (run
    side by side for different friction laws) must not overwrite each other's."""
    tail = "" if mode == "shared" else f"_{mode}{mode_postfix(mode, fixed)}"
    cfg = "" if not configs else "_" + "+".join(sorted(configs))
    return DATA / "results" / f"sweep_manifest{tail}{cfg}{RUN_SUFFIX}.json"


_DISPATCH = threading.Lock()
LOAD_SETTLE_S = 90   # the 1-min load average needs about this long to register a new job


def start(cmd, fh, env, max_load):
    r"""Launch one job, first waiting for the machine to have room if ``max_load`` is set.

    With a load cap, dispatch is serialized and each launch is followed by
    LOAD_SETTLE_S of quiet, so idle worker threads cannot all read the same low
    load average and start together.
    """
    kw = dict(cwd=str(DATA), env=env, stdout=fh, stderr=subprocess.STDOUT)
    if max_load is None:
        return subprocess.Popen(cmd, **kw)
    with _DISPATCH:
        while os.getloadavg()[0] >= max_load:
            time.sleep(60)
        proc = subprocess.Popen(cmd, **kw)
        time.sleep(LOAD_SETTLE_S)
    return proc


def prior_args(mode, delta, ell, fixed=None):
    r"""invert.py prior arguments for one point."""
    if mode == "shared":
        return ["--delta", repr(delta), "--ell", repr(ell)]
    sw, hd = SWEPT[mode], HELD[mode]
    return [f"--delta-{sw}", repr(delta), f"--ell-{sw}", repr(ell),
            f"--delta-{hd}", repr(fixed[0]), f"--ell-{hd}", repr(fixed[1])]


def build_jobs(args):
    r"""One job per (config, delta, ELL), longest-running first."""
    jobs = []
    fixed = (args.fixed_delta, args.fixed_ell)
    for model, friction, no_apres in CONFIGS:
        if args.configs and config_tag(model, friction, no_apres) not in args.configs:
            continue
        for delta in sorted(args.deltas):          # weakest prior first: slowest
            for ell in args.ells:
                # EXCLUDE is a shared-prior finding; a single-control sweep runs
                # every point and lets the plot flag any runaway.
                if args.mode == "shared" and (model, friction, no_apres, delta, ell) in EXCLUDE:
                    continue
                name = point_name(model, friction, no_apres, delta, ell, args.mode, fixed,
                                  args.prior_form)
                cmd = [PYTHON, str(SCRIPTS / "invert.py"),
                       "--model", model, "--friction", friction,
                       *prior_args(args.mode, delta, ell, fixed),
                       "--prior-form", args.prior_form,
                       "--name", name, "--maxiter", str(args.maxiter)]
                if args.grtol is not None:
                    cmd += ["--grtol", repr(args.grtol)]
                if no_apres:
                    cmd.append("--no-apres")
                jobs.append({"name": name, "model": model, "friction": friction,
                             "no_apres": no_apres, "delta": delta, "ell": ell,
                             "cmd": cmd})
    return jobs


def run_job(job, args):
    r"""Run one MAP inversion as a subprocess; never raise, always report."""
    out = DATA / "results" / f"map_{job['name']}.json"
    log = LOG_DIR / f"{job['name']}.log"
    if args.skip_existing and out.exists():
        try:
            prev = json.load(open(out))["status"]
        except (ValueError, KeyError):
            prev = ""
        # A point that stopped on maxiter is rerun; invert.py resumes it from
        # its checkpoint with the new, larger iteration budget.
        if prev.startswith("converged"):
            return {**job, "status": "skipped (converged result exists)", "wall_h": 0.0, "rc": 0}
        if stalled(prev) and not args.retry_stalled:
            return {**job, "status": "skipped (stalled result exists: outside physical range)",
                    "wall_h": 0.0, "rc": 0}
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with open(log, "w") as fh:
        proc = start(job["cmd"], fh, env, args.max_load)
        t0 = time.time()           # wall time of the run itself, not of waiting for room
        rc = proc.wait()
    wall = (time.time() - t0) / 3600
    status = "ok"
    if rc != 0:
        status = f"FAILED rc={rc}"
    elif not out.exists():
        status = "FAILED (no result json)"
    else:
        try:
            status = json.load(open(out))["status"]
        except (ValueError, KeyError):
            status = "FAILED (unreadable result json)"
    return {**job, "status": status, "wall_h": wall, "rc": rc}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=12,
                    help="concurrent inversions; each is serial, and a hybrid run "
                         "peaks near 25 GB, so 12 keeps a 376 GB node out of OOM")
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--grtol", type=float, default=None,
                    help="passed to invert.py (its default otherwise); the relative "
                         "gradient test is scaled by each point's starting gradient")
    ap.add_argument("--skip-existing", action="store_true",
                    help="leave points that already have a converged result json "
                         "alone; points that stopped on maxiter are resumed, and "
                         "stalled points (see stalled()) are left alone too")
    ap.add_argument("--retry-stalled", action="store_true",
                    help="with --skip-existing, rerun stalled points instead of "
                         "treating them as outside the physical range")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mode", choices=MODES, default="shared",
                    help="which prior the grid varies (see the module docstring)")
    ap.add_argument("--fixed-delta", type=float, default=1.0e-5,
                    help="single-control modes: strength the other control is held at")
    ap.add_argument("--fixed-ell", type=float, default=2000.0,
                    help="single-control modes: correlation length (m) it is held at")
    ap.add_argument("--max-load", type=float, default=None,
                    help="dispatch a point only while the 1-minute load average is below "
                         "this, so a sweep can share a busy machine without oversubscribing")
    ap.add_argument("--configs", nargs="+", default=None, choices=CONFIG_TAGS,
                    help="only these configuration tags (default: every entry of CONFIGS)")
    ap.add_argument("--prior-form", choices=FORMS, default=DEFAULT_FORM,
                    help="prior form passed to invert.py; bilaplacian points are named with "
                         "a _bilap infix. delta is not comparable between forms "
                         "(prior_precond.py)")
    ap.add_argument("--deltas", type=float, nargs="+", default=DELTAS)
    ap.add_argument("--ells", type=float, nargs="+", default=ELLS,
                    help="correlation lengths in metres")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    manifest = manifest_path(args.mode, (args.fixed_delta, args.fixed_ell), args.configs)
    held = ("" if args.mode == "shared" else
            f", {HELD[args.mode]} prior held at delta={args.fixed_delta:.0e} "
            f"ELL={args.fixed_ell:g} m")
    n_cfg = len(args.configs) if args.configs else len(CONFIGS)
    print(f"mesh {MESH_NAME}, tau_d smoothing {TAU_D_SMOOTH:g} m, {args.prior_form} prior, "
          f"mode {args.mode}{held}: "
          f"{len(jobs)} points: "
          f"{n_cfg} configs x {len(args.deltas)} delta "
          f"x {len(args.ells)} ELL, {args.workers} at a time", flush=True)
    if args.dry_run:
        for j in jobs:
            print(f"  {j['name']:52s} delta={j['delta']:.0e} ell={j['ell']:.0f}", flush=True)
        return

    t0 = time.time()
    done, results = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, j, args): j for j in jobs}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            flag = "" if r["status"].startswith(("ok", "converged", "skipped")) else "  <-- CHECK"
            print(f"[{done}/{len(jobs)}] {r['name']:52s} {r['wall_h']:5.2f} h  "
                  f"{r['status']}{flag}", flush=True)
            manifest.write_text(json.dumps(
                {"elapsed_h": (time.time() - t0) / 3600,
                 "completed": done, "total": len(jobs),
                 "jobs": [{k: v for k, v in x.items() if k != "cmd"} for x in results]},
                indent=2))

    bad = [r for r in results if not r["status"].startswith(("converged", "skipped"))]
    print(f"\nSweep finished in {(time.time() - t0) / 3600:.2f} h wall; "
          f"{len(results) - len(bad)}/{len(results)} converged", flush=True)
    for r in bad:
        print(f"  {r['name']:52s} {r['status']}", flush=True)
    print(f"Manifest: {manifest}", flush=True)


if __name__ == "__main__":
    main()
