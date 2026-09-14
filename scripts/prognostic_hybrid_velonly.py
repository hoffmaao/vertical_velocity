r"""Prognostic experiment using the velocity-only hybrid inversion.

100-year forward simulation on the hires Thwaites mesh with RACMO SMB
and sub-shelf melt, driven by fluidity and friction fields recovered
from surface velocity assimilation alone (no ApRES data), using the
hybrid (Blatter-Pattyn) model.

Usage:
    OMP_NUM_THREADS=4 python prognostic_hybrid_velonly.py
"""
from prognostic_common import run_prognostic, DATA_DIR

INV_FILE = DATA_DIR / "mesh" / "inversion_hybrid.h5"

if __name__ == "__main__":
    run_prognostic("hybrid_velonly", INV_FILE)
