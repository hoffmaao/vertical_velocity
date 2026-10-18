#!/bin/bash
# Bilaplacian-prior (delta 4e-4, ELL 3 km) rc chain for one model:
#   stage fwd: forward.py --gradient (ISMIP7 CESM2-WACCM ssp585 from 2015, no a_ref,
#              melt slope frozen off the tape, forward.py's default)
#   stage eig: eigendec.py K=1500 (replays results/eigendec_cache/matvecs_<fp>.h5)
#   stage uq:  after fwd + eig: uq_vaf_pcg, uq_vaf_pcg --misfit-scale chi2, uq_vaf --K 1500
# usage: launch_bilap_chain.sh ssa|hybrid fwd|eig|uq
M=$1; STAGE=$2
D=/media/andrew/wd1/projects/vertical_velocity
N=sweep_${M}_rc_bilap_d4e-04_L3km_thwaites_fixed_tau5km
PY=$HOME/venv-firedrake/bin/python
ENV="-E VV_MESH=thwaites_fixed -E VV_TAU_SMOOTH=5000 -E OMP_NUM_THREADS=1 -E FIREDRAKE_DISABLE_OPTIONS_LEFT=1"
run() { u=$1; l=$2; shift 2
  systemd-run --user --unit=$u --working-directory=$D $ENV \
    -p StandardOutput=truncate:$l -p StandardError=truncate:$l "$@"; }
case $STAGE in
fwd) run vv-fwdb-$M $D/output/logs/forward/forward_${N}_50yr.log \
       $PY -u scripts/forward.py --model $M --friction rc --map $N --T 50 --gradient \
       --gradient-times 5 10 15 20 25 30 35 40 45 50 ;;
eig) run vv-eigb-$M $D/output/logs/eigendec/eigendec_${N}_K1500.log \
       $PY -u scripts/eigendec.py --model $M --friction rc --map $N --K 1500 ;;
uq)  run vv-uqb-$M $D/output/logs/uq/uq_${N}.log \
       bash -c "while systemctl --user is-active -q vv-fwdb-$M || systemctl --user is-active -q vv-eigb-$M; do sleep 600; done; \
         test -f results/forward_${N}_50yr.h5 || { echo 'no forward output'; exit 1; }; \
         test -f results/eigendec_${N}_K1500.json || { echo 'no eigendec output'; exit 1; }; \
         $PY -u scripts/uq_vaf_pcg.py --model $M --friction rc --map $N --T 50 --K 1500 && \
         $PY -u scripts/uq_vaf_pcg.py --model $M --friction rc --map $N --T 50 --K 1500 --misfit-scale chi2 && \
         $PY -u scripts/uq_vaf.py --map $N --T 50 --K 1500" ;;
*) echo "usage: $0 ssa|hybrid fwd|eig|uq" >&2; exit 2 ;;
esac
