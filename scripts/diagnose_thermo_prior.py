r"""Why is the thermomechanical fluidity prior so much warmer than Pattyn (2013)?

physical_prior.py builds A_prior from the depth-averaged enthalpy model in
thermo_model.py. This script reruns that model with one ingredient switched off
at a time and compares every variant, as an equivalent depth-averaged
temperature, against the Pattyn (2013) temperature field that the other
initialization procedures (DotsonCrosson, Pine Island, Cook) use directly as
A = rate_factor(T).

    python scripts/diagnose_thermo_prior.py run baseline      # one variant
    python scripts/diagnose_thermo_prior.py run no_duval
    python scripts/diagnose_thermo_prior.py plot              # compare all that exist

Variants (VARIANTS below): the production defaults, then each heating term or
closure removed on its own, so the attribution is one factor at a time.

"Equivalent temperature" inverts icepack's rate_factor, T_eq = Q / (R ln(A0 / A)),
so the Duval meltwater enhancement shows up as an apparent warming above the
melting point's A. The production hybrid MAP's inferred fluidity
A_prior exp(theta_A) is converted the same way: it is what the velocity and
ApRES data say the depth-averaged rheology is, independent of either model.

Writes results/thermo_diag/<variant>.npz per run, and for plot
figures/thermo_vs_pattyn.png and results/thermo_diag/summary.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent
OUT = DATA / "results" / "thermo_diag"
PATTYN = Path("/home/andrew/projects/ismip7/antarctica/data/temp/Pattyn_2013.tif")
MAP_FILE = DATA / "mesh" / "map_hybrid_weertman_fixed_tau5km.h5"

# name: (thermo_model parameter overrides, friction heating on)
VARIANTS = {
    "baseline": ({}, True),
    "no_duval": ({"duval": 0.0}, True),
    "shear_amp1": ({"shear_amp": 1.0}, True),
    "no_friction": ({}, False),
    "no_strain": ({"shear_amp": 0.0}, True),
    "kappa2": ({"kappa": 2.0}, True),
    "conduction_only": ({"shear_amp": 0.0, "duval": 0.0}, False),
    "no_shelf_geo": ({"geo_under_shelf": False}, True),
}

# icepack.models.viscosity: A = A0 exp(-Q / (R T)), cold/warm branches split at 263.15 K
A0_COLD, A0_WARM, Q_COLD, Q_WARM, T_TRANS, R_GAS = (
    12575703600000.0, 6.04643616e28, 60.0, 139.0, 263.15, 0.0083144621)
A_TRANS = A0_COLD * np.exp(-Q_COLD / (R_GAS * T_TRANS))
TM = 273.15


def rate_factor_np(T):
    r"""icepack's rate_factor on arrays (same constants as T_equivalent)."""
    cold = T < T_TRANS
    return np.where(cold, A0_COLD, A0_WARM) * np.exp(-np.where(cold, Q_COLD, Q_WARM) / (R_GAS * T))


def T_equivalent(A):
    r"""Invert icepack's rate_factor; A above A(Tm) (Duval-enhanced) maps above Tm."""
    A = np.asarray(A, dtype=float)
    cold = A < A_TRANS
    A0 = np.where(cold, A0_COLD, A0_WARM)
    Q = np.where(cold, Q_COLD, Q_WARM)
    return Q / (R_GAS * np.log(A0 / A))


def run(variant):
    import firedrake
    from firedrake import Constant, Function, FunctionSpace, VectorFunctionSpace, max_value
    import thermo_model as tm
    from friction_laws import flotation_factor
    from physical_prior import (load_geometry, load_racmo_smb, load_surface_temperature,
                                weertman_anchor)

    overrides, friction = VARIANTS[variant]
    mesh, F = load_geometry()
    h, s, b, u_obs = F["thickness"], F["surface"], F["bed"], F["velocity"]
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    C_w0, _ = weertman_anchor(h, s, u_obs, Q, V)
    smb = load_racmo_smb(Q)
    T_srf = load_surface_temperature(Q)
    C_heat = Function(Q).interpolate(C_w0 * max_value(flotation_factor(h, s), 0)
                                     * Constant(1.0 if friction else 0.0))
    print(f"variant {variant}: overrides {overrides}, friction heating {friction}", flush=True)
    A = tm.compute_fluidity_prior(u_obs, h, s, b, C_heat, smb, T_srf, p=overrides)

    # Final energy fields at the converged A, for the temperature / meltwater split
    # and the column heat budget.
    P = dict(tm.DEFAULTS, **overrides)
    QD = firedrake.FunctionSpace(mesh, "DG", 0)
    h_th = Function(Q).interpolate(max_value(h, Constant(10.0)))
    E_eff, E_full = tm.thermo_update(u_obs, h_th, s, C_heat, Function(QD).project(A), b,
                                     Function(QD).project(smb), P,
                                     T_srf_field=Function(QD).project(T_srf))
    T_cell = Function(QD).interpolate(tm.temperature(E_eff))
    w_cell = Function(QD).interpolate(firedrake.min_value(tm.meltwater(E_eff),
                                                          Constant(P["w_max"])))
    T_node = tm.to_nodes(T_cell, Q)
    w_node = tm.to_nodes(w_cell, Q)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{variant}.npz", A=A.dat.data_ro, T=T_node.dat.data_ro,
             w=w_node.dat.data_ro, overrides=json.dumps(overrides), friction=friction)
    print(f"Saved {OUT / (variant + '.npz')}", flush=True)


def budget_and_inputs():
    r"""Per-node inputs shared by every variant, plus the baseline heat sources in mW/m^2."""
    import icepack
    import rasterio
    from firedrake import (CheckpointFile, Constant, Function, FunctionSpace,
                           VectorFunctionSpace, inner, max_value, sqrt, sym, grad)
    from friction_laws import flotation_factor
    from physical_prior import load_geometry, load_surface_temperature, weertman_anchor

    mesh, F = load_geometry()
    h, s, u_obs = F["thickness"], F["surface"], F["velocity"]
    Q = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 1)
    with rasterio.open(PATTYN) as src:
        T_p = icepack.interpolate(src, Q)
    C_w0, _ = weertman_anchor(h, s, u_obs, Q, V)
    phi = Function(Q).interpolate(flotation_factor(h, s))
    speed = Function(Q).interpolate(sqrt(inner(u_obs, u_obs)))
    T_srf = load_surface_temperature(Q)

    # Heat sources as the baseline solve sees them (MPa m/yr -> mW/m^2).
    to_mW = 1e6 / 3.15576e7 * 1e3
    base = np.load(OUT / "baseline.npz")
    A_b = Function(Q); A_b.dat.data[:] = base["A"]
    eps = sym(grad(u_obs))
    eps_e = sqrt(0.5 * inner(eps, eps) + Constant(1e-16))
    q_strain = Function(Q).interpolate(
        h * 2 * A_b ** (-1.0 / 3) * eps_e ** (1.0 / 3 + 1))
    q_fric = Function(Q).interpolate(
        C_w0 * max_value(phi, 0) * max_value(sqrt(inner(u_obs, u_obs)), Constant(1e-6))
        ** (1.0 / 3 + 1))

    with CheckpointFile(str(MAP_FILE), "r") as chk:
        mm = chk.load_mesh()
        th_A = chk.load_function(mm, "log_fluidity").dat.data_ro.copy()
        A_pr = chk.load_function(mm, "A_prior").dat.data_ro.copy()
    xy = mesh.coordinates.dat.data_ro
    return dict(x=xy[:, 0], y=xy[:, 1], h=h.dat.data_ro.copy(), phi=phi.dat.data_ro.copy(),
                speed=speed.dat.data_ro.copy(), T_srf=T_srf.dat.data_ro.copy(),
                T_pattyn=T_p.dat.data_ro.copy(),
                q_strain=q_strain.dat.data_ro * to_mW, q_fric=q_fric.dat.data_ro * to_mW,
                A_map=A_pr * np.exp(th_A), tri=mesh.coordinates.cell_node_map().values)


def stats(dT, mask):
    d = dT[mask]
    return {"median": float(np.median(d)), "mean": float(d.mean()),
            "p10": float(np.percentile(d, 10)), "p90": float(np.percentile(d, 90))}


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    I = budget_and_inputs()
    runs = {v: np.load(OUT / f"{v}.npz") for v in VARIANTS if (OUT / f"{v}.npz").exists()}
    if "baseline" not in runs:
        raise SystemExit("run the baseline variant first")
    ice = (I["h"] > 10.0) & np.isfinite(I["T_pattyn"]) & (I["T_pattyn"] > 200) & (I["T_pattyn"] < 274)
    regions = {"grounded": ice & (I["phi"] > 0.05), "floating": ice & (I["phi"] <= 0.05),
               "grounded slow (<100 m/yr)": ice & (I["phi"] > 0.05) & (I["speed"] < 100),
               "grounded fast (>500 m/yr)": ice & (I["phi"] > 0.05) & (I["speed"] > 500)}
    Tp = I["T_pattyn"]
    summary = {"pattyn_T": {k: stats(Tp, m) for k, m in regions.items()},
               "T_srf": {k: stats(I["T_srf"], m) for k, m in regions.items()},
               "upper_bound_(Ts+Tm)/2": {k: stats(0.5 * (I["T_srf"] + TM), m)
                                        for k, m in regions.items()},
               "sources_mW_m2_baseline": {k: {"strain_unamplified": float(np.median(I["q_strain"][m])),
                                              "friction": float(np.median(I["q_fric"][m])),
                                              "geothermal": 50.0}
                                          for k, m in regions.items()},
               "dT_eq_vs_pattyn": {}, "lnA_ratio_vs_pattyn": {}, "temperate_fraction": {}}
    A_pattyn = rate_factor_np(np.clip(Tp, 200.0, TM))

    def A_of(v):
        return I["A_map"] if v == "MAP (data)" else runs[v]["A"]

    Teq = {v: T_equivalent(r["A"]) for v, r in runs.items()}
    Teq["MAP (data)"] = T_equivalent(I["A_map"])
    for v, T in Teq.items():
        summary["dT_eq_vs_pattyn"][v] = {k: stats(T - Tp, m) for k, m in regions.items()}
        summary["lnA_ratio_vs_pattyn"][v] = {k: stats(np.log(A_of(v) / A_pattyn), m)
                                             for k, m in regions.items()}
    for v, r in runs.items():
        summary["temperate_fraction"][v] = {k: float((r["w"][m] > 1e-6).mean())
                                            for k, m in regions.items()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("median ln(A / A_Pattyn) [p10, p90]")
    print(f"{'variant':18s} " + " ".join(f"{k:>26s}" for k in regions))
    for v in Teq:
        print(f"{v:18s} " + " ".join(
            f"{summary['lnA_ratio_vs_pattyn'][v][k]['median']:+9.2f}  "
            f" [{summary['lnA_ratio_vs_pattyn'][v][k]['p10']:+5.2f},{summary['lnA_ratio_vs_pattyn'][v][k]['p90']:+5.2f}]"
            for k in regions))
    print("median T_eq - T_Pattyn [p10, p90]")
    print(f"{'variant':18s} " + " ".join(f"{k:>26s}" for k in regions))
    for v in Teq:
        print(f"{v:18s} " + " ".join(
            f"{summary['dT_eq_vs_pattyn'][v][k]['median']:+9.1f} K"
            f" [{summary['dT_eq_vs_pattyn'][v][k]['p10']:+5.1f},{summary['dT_eq_vs_pattyn'][v][k]['p90']:+5.1f}]"
            for k in regions))

    tri = Triangulation(I["x"] / 1e3, I["y"] / 1e3, I["tri"])
    order = ["baseline", "no_duval", "shear_amp1", "no_strain", "no_friction", "kappa2",
             "conduction_only", "no_shelf_geo"]
    shown = [v for v in order if v in runs]
    panels = [("Pattyn (2013) T", Tp, "T"), ("our baseline $T_{eq}$", Teq["baseline"], "T"),
              ("MAP-implied $T_{eq}$ (data)", Teq["MAP (data)"], "T"),
              ("baseline - Pattyn", Teq["baseline"] - Tp, "dT"),
              (r"baseline $\ln(A/A_{\rm Pattyn})$", np.log(runs["baseline"]["A"] / A_pattyn), "lnA"),
              ("MAP - Pattyn", Teq["MAP (data)"] - Tp, "dT")]
    panels += [(f"{v} - Pattyn", Teq[v] - Tp, "dT") for v in shown if v != "baseline"]
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.0 * nrow), squeeze=False)
    mask_tri = ~np.all(ice[I["tri"]], axis=1)
    tri.set_mask(mask_tri)
    for ax, (title, f, kind) in zip(axes.ravel(), panels):
        if kind == "T":
            im = ax.tripcolor(tri, np.clip(f, 230, 280), cmap="RdYlBu_r", vmin=235, vmax=275,
                              shading="gouraud", rasterized=True)
            label = "K"
        elif kind == "lnA":
            im = ax.tripcolor(tri, np.clip(f, -4, 4), cmap="PuOr_r", vmin=-3, vmax=3,
                              shading="gouraud", rasterized=True)
            label = r"$\ln(A/A_{\rm Pattyn})$"
            g = regions["grounded"]
            ax.text(0.03, 0.03, f"grounded median {np.median(f[g]):+.2f}",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(fc="white", ec="0.7", alpha=0.85))
        else:
            im = ax.tripcolor(tri, np.clip(f, -30, 30), cmap="RdBu_r", vmin=-25, vmax=25,
                              shading="gouraud", rasterized=True)
            label = "K"
            g = regions["grounded"]
            ax.text(0.03, 0.03, f"grounded median {np.median(f[g]):+.1f} K",
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(fc="white", ec="0.7", alpha=0.85))
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=label)
    for ax in axes.ravel()[len(panels):]:
        ax.set_axis_off()
    fig.suptitle("Depth-averaged ice temperature: thermo prior variants vs Pattyn (2013)\n"
                 r"(model and MAP as equivalent $T_{eq}$ = rate_factor$^{-1}(A)$; "
                 "Duval enhancement appears above 273 K)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = DATA / "figures" / "thermo_vs_pattyn.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"Saved {out} and {OUT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("variant", choices=list(VARIANTS))
    sub.add_parser("plot")
    a = ap.parse_args()
    run(a.variant) if a.cmd == "run" else plot()
