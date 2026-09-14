r"""Plot L-curve results for SSA and Hybrid+ApRES inversions.

Reads JSON results from data/ and produces a multi-panel figure.

Usage:
    python plot_lcurve.py
"""
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Load SSA ───────────────────────────────────────────────────────
with open(DATA_DIR / "data" / "lcurve_ssa_manual.json") as f:
    ssa = json.load(f)

# ── Load Hybrid+ApRES (per-L files; L now in km) ──────────────────
hybrid = []
for L in [0.5, 1, 2, 3, 5]:
    # File naming preserves the same convention as the CLI argument
    suffix = f"L{L:g}km"
    p = DATA_DIR / "data" / f"lcurve_hybrid_apres_{suffix}.json"
    if p.exists():
        with open(p) as f:
            hybrid.append(json.load(f))

# ── Figure ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Panel (a): SSA L-curve
ax = axes[0]
Ls = [d["L_km"] for d in ssa]
Es = [d["E"] for d in ssa]
Rs = [d["R"] for d in ssa]
ax.plot(Rs, Es, "ko-", ms=7, lw=1.5)
for L, E, R in zip(Ls, Es, Rs):
    ax.annotate(f"{L} km", (R, E), textcoords="offset points",
                xytext=(8, 4), fontsize=9, color="0.3")
ax.set_xlabel("Regularization $\\mathcal{R}$", fontsize=12)
ax.set_ylabel("Misfit $\\mathcal{E}$", fontsize=12)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("(a) SSA", fontsize=13)
ax.grid(True, alpha=0.2, which="both")

# Panel (b): Hybrid+ApRES L-curve (E_total vs R)
ax = axes[1]
Ls_h = [d["L_km"] for d in hybrid]
Es_h = [d["E_total"] for d in hybrid]
Rs_h = [d["R"] for d in hybrid]
iters_h = [d["iters"] for d in hybrid]
# Mark converged (>100 iters) vs not
converged = [it > 100 for it in iters_h]
ax.plot(Rs_h, Es_h, "s-", color="C0", ms=7, lw=1.5, zorder=2)
for L, E, R, conv in zip(Ls_h, Es_h, Rs_h, converged):
    marker = "s" if conv else "x"
    color = "C0" if conv else "C3"
    ax.plot(R, E, marker, color=color, ms=9, mew=2, zorder=3)
    ax.annotate(f"{L:g} km", (R, E), textcoords="offset points",
                xytext=(8, 4), fontsize=9,
                color="0.3" if conv else "C3")
ax.set_xlabel("Regularization $\\mathcal{R}$", fontsize=12)
ax.set_ylabel("Misfit $\\mathcal{E}_{\\mathrm{vel}} + \\mathcal{E}_{\\mathrm{ApRES}}$",
              fontsize=12)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("(b) Hybrid + ApRES", fontsize=13)
ax.grid(True, alpha=0.2, which="both")
# Legend
from matplotlib.lines import Line2D
ax.legend([Line2D([0], [0], marker="s", color="C0", ls="", ms=7),
           Line2D([0], [0], marker="x", color="C3", ls="", ms=9, mew=2)],
          ["converged (>100 iter)", "not converged"],
          fontsize=9, loc="upper right")

# Panel (c): Hybrid E_vel and E_apr separately vs L
ax = axes[2]
Ev = [d["E_vel"] for d in hybrid]
Ea = [d["E_apr"] for d in hybrid]
ax.semilogy(Ls_h, Ev, "o-", color="C0", ms=7, lw=1.5, label="$\\mathcal{E}_{\\mathrm{vel}}$")
ax.semilogy(Ls_h, Ea, "^-", color="C1", ms=7, lw=1.5, label="$\\mathcal{E}_{\\mathrm{ApRES}}$")
for i, (L, conv) in enumerate(zip(Ls_h, converged)):
    if not conv:
        ax.plot(L, Ev[i], "x", color="C3", ms=10, mew=2, zorder=3)
        ax.plot(L, Ea[i], "x", color="C3", ms=10, mew=2, zorder=3)
ax.set_xlabel("$L$ (km)", fontsize=12)
ax.set_ylabel("Misfit", fontsize=12)
ax.set_xscale("log")
ax.set_title("(c) Misfit components vs $L$", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, which="both")
ax.set_xticks(Ls_h)
ax.set_xticklabels([f"{L:g}" for L in Ls_h])

fig.tight_layout()
out = FIG_DIR / "lcurve_comparison.png"
fig.savefig(str(out), dpi=200, bbox_inches="tight")
print(f"Saved {out}")

# Also print a summary table
print("\n=== SSA ===")
print(f"{'L (km)':>8s}  {'E':>10s}  {'R':>10s}  {'iters':>6s}")
for d in ssa:
    print(f"{d['L_km']:8d}  {d['E']:10.3e}  {d['R']:10.3e}  {d['iters']:6d}")

print("\n=== Hybrid + ApRES ===")
print(f"{'L (km)':>8s}  {'E_vel':>10s}  {'E_apr':>10s}  {'R':>10s}  {'iters':>6s}")
for d in hybrid:
    flag = " *" if d["iters"] <= 100 else ""
    print(f"{d['L_km']:8.2g}  {d['E_vel']:10.3e}  {d['E_apr']:10.3e}  "
          f"{d['R']:10.3e}  {d['iters']:6d}{flag}")
print("  * = not converged (<100 iters)")
