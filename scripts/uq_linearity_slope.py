r"""Verdict: is the adjoint VAF gradient an over-estimate, or is the response saturating?

Computes the secant slope s(n) = [ΔVAF(n) − ΔVAF(0)] / n for every amplitude in
the directional sweep, sorted by |n|. As n→0 the secant → the true directional
derivative of VAF along δθ_1σ. Compare to the adjoint/linearized value σ_post:
  s(n→0) ≈ σ_post  ⇒  linearization locally valid; the flattening at larger n is
                       genuine saturation.
  s(n→0) ≈ σ_post/2 (flat at ~38)  ⇒  the taped adjoint OVER-estimates (max_value
                       flotation kink / quadrature artifact) — a gradient bug.
Also reports +/− asymmetry at matched |n| (tests whether the derivative is even
well-defined at the flotation kink).
"""
import glob
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent


def main():
    dvaf = {}
    sigma = None
    for f in glob.glob(str(DATA / "results" / "uq_linearity_n*.npz")):
        d = np.load(f)
        if bool(d["failed"]):
            continue
        dvaf[float(d["n"])] = float(d["dVAF"])
        sigma = float(d["sigma_post_Gt"])
    assert 0.0 in dvaf, "missing n=0 baseline"
    dVAF0 = dvaf[0.0]
    ns = sorted(k for k in dvaf if abs(k) > 1e-9)

    print(f"adjoint/linearized directional slope  σ_post = {sigma:.3f} Gt/n")
    print(f"baseline ΔVAF(0) = {dVAF0:.3f} Gt\n")
    print(f"{'|n|':>6} {'n':>7} {'R_true':>11} {'secant R/n':>12} {'/σ_post':>9}")
    print("-" * 50)
    for n in sorted(ns, key=abs):
        R = dvaf[n] - dVAF0
        s = R / n
        print(f"{abs(n):>6.2f} {n:>7.2f} {R:>11.3f} {s:>12.3f} {s/sigma:>8.2f}")

    # n→0 through-origin slope from the smallest points
    for cap in (0.5, 0.75, 1.0):
        sm = [n for n in ns if 0 < abs(n) <= cap]
        xs = np.array(sm)
        Rs = np.array([dvaf[n] - dVAF0 for n in sm])
        a0 = float(np.sum(xs * Rs) / np.sum(xs**2))   # LS slope of R = a·n
        print(f"\nthrough-origin slope (|n|≤{cap}): {a0:.3f} Gt/n  "
              f"= {a0/sigma:.2f}×σ_post  (n={len(sm)} pts)")

    # +/- asymmetry at matched |n|
    print("\nasymmetry at matched |n|  (secant+ vs secant−):")
    mags = sorted({abs(n) for n in ns})
    for m in mags:
        if m in dvaf and -m in dvaf:
            sp = (dvaf[m] - dVAF0) / m
            sm = (dvaf[-m] - dVAF0) / (-m)
            print(f"  |n|={m:>5.2f}:  s+={sp:>8.2f}   s−={sm:>8.2f}   "
                  f"Δ={sp - sm:>+8.2f} Gt/n")


if __name__ == "__main__":
    main()
