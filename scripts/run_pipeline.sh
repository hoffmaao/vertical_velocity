#!/usr/bin/env bash
# Rebuild the ApRES vertical-velocity analysis and Figs. 1-5 of the GRL manuscript
# as it currently stands (figures/strain_pipeline/) from the raw data, in
# dependency order.  manuscript_prepare.py + manuscript_figures.py are a separate,
# later rebuild of the manuscript figures; see README.
#
#   ./scripts/run_pipeline.sh            full rebuild
#   ./scripts/run_pipeline.sh figures    figures only, from existing results
#
# Stage 2 is the expensive one (cross-correlation of 88 repeat pairs, ~30 min
# on 4 cores).  Stages 3 and 4 are seconds, which is why the decomposition is
# kept separate from the correlation: the fitting window, the polynomial degree
# and the screening thresholds can all be revised without reprocessing radar.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"

# The analysis stack lives in the `apres` env; PyGMT needs the GMT C library,
# which is only in the `pygmt` env.
PY_ANALYSIS="${PY_ANALYSIS:-/opt/anaconda3/envs/apres/bin/python}"
PY_FIGURES="${PY_FIGURES:-/opt/anaconda3/envs/pygmt/bin/python}"

# numpy's BLAS spawns a thread pool per worker; without this the process pool
# oversubscribes the machine and spends its time in the kernel rather than in
# the correlation.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

run() { echo; echo "==> $*"; "$@"; }

# Stage order matters: compute_strain_rates.py reads profiles.nc, and
# fit_legendre.py reads both profiles.nc and strain_profiles.nc.  A
# figures-only run reuses whatever those stages last wrote.
if [ "${1:-all}" != "figures" ]; then
  run "$PY_ANALYSIS" scripts/build_site_catalog.py
  run "$PY_ANALYSIS" scripts/compute_vertical_velocities.py --jobs "${JOBS:-4}"
  run "$PY_ANALYSIS" scripts/compute_strain_rates.py
  run "$PY_ANALYSIS" scripts/compute_surface_strain.py
  run "$PY_ANALYSIS" scripts/fit_legendre.py
fi

run "$PY_FIGURES" scripts/figures/fig01_overview.py
run "$PY_FIGURES" scripts/figures/fig02_section.py
run "$PY_FIGURES" scripts/figures/fig03_strain_regression.py
run "$PY_FIGURES" scripts/figures/fig04_map.py
run "$PY_FIGURES" scripts/figures/fig05_legendre.py
run "$PY_FIGURES" scripts/figures/figS10_bed_link.py

echo
echo "figures written to figures/strain_pipeline/"

