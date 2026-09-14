r"""Prognostic experiment using the ApRES-informed inversion.

50-year forward simulation on the hires Thwaites mesh with RACMO SMB
and sub-shelf melt, driven by fluidity and friction fields recovered
from joint assimilation of surface velocity and ApRES vertical strain
rates.

Usage:
    OMP_NUM_THREADS=8 python prognostic_apres.py
"""
from pathlib import Path
from prognostic_common import run_prognostic, DATA_DIR

INV_FILE = DATA_DIR / "mesh" / "inversion_hires_apres_vd1.h5"

if __name__ == "__main__":
    run_prognostic("apres", INV_FILE)
