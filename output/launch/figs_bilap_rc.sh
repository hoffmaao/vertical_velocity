#!/bin/bash
# Figures for the ISMIP7-forced (no reference balance) rc runs, bilaplacian prior
# delta 4e-4, SSA vs hybrid. Each figure waits for its own inputs, so the
# eigendec comparison does not queue behind the (slower) hybrid taped forward.
# usage: systemd-run --user --unit vv-figs-bilap ... figs_bilap_rc.sh
source $HOME/venv-firedrake/bin/activate
export FIREDRAKE_DISABLE_OPTIONS_LEFT=1  # argparse flags are not PETSc options
cd /media/andrew/wd1/projects/vertical_velocity/scripts
S=_rc_bilap_d4e-04_L3km_thwaites_fixed_tau5km; R=../results
SSA=sweep_ssa$S; HYB=sweep_hybrid$S
wait_file() { while [ ! -f "$1" ]; do sleep 300; done; echo "$(date +%H:%M) have $(basename $1)"; }
( wait_file $R/forward_${SSA}_50yr.h5; wait_file $R/forward_${HYB}_50yr.h5
  python plot_forward.py --T 50 $SSA $HYB --suffix rc_bilap_L3km
  python plot_gz_compare.py --T 50 $SSA $HYB --suffix _rc_bilap_L3km
  echo "$(date +%H:%M) forward figures done" ) &
( wait_file $R/eigendec_${SSA}_K1500.npz; wait_file $R/eigendec_${HYB}_K1500.npz
  python plot_eigendec.py --map $SSA $HYB --suffix _rc_bilap_L3km
  echo "$(date +%H:%M) eigendec figure done" ) &
( wait_file $R/uq_${SSA}_50yr_pcg_summary.json; wait_file $R/uq_${HYB}_50yr_pcg_summary.json
  python plot_retreat_uq.py --T 50 $SSA $HYB --suffix _ssa_vs_hybrid_rc_bilap_L3km_pcg
  wait_file $R/uq_${SSA}_50yr_K1500_summary.json; wait_file $R/uq_${HYB}_50yr_K1500_summary.json
  python plot_retreat_uq.py --T 50 $SSA $HYB --uq eig --suffix _ssa_vs_hybrid_rc_bilap_L3km_K1500
  echo "$(date +%H:%M) UQ figures done" ) &
wait
echo "$(date +%H:%M) all figures done"
