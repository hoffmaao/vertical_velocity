# Thwaites Glacier: ApRES strain-rate assimilation

Ice-flow simulations for Hoffman et al., *Basal properties drive changes in englacial
deformation on Thwaites Glacier* (in prep., GRL).

We invert for ice fluidity and basal friction on Thwaites Glacier with
[icepack](https://icepack.github.io), assimilating satellite surface velocity (MEaSUREs 450 m)
and 91 ApRES/pRES vertical strain-rate profiles. We then run posterior ensembles of 50-yr
projections to compare mass loss between the hybrid (higher-order) model and the shallow-shelf
approximation (SSA, plug flow).

## Layout

| Path          | Contents                                                                                        |
|---------------|-------------------------------------------------------------------------------------------------|
| `scripts/`    | All code: ApRES processing (`apres/`), mesh, inversion, L-curve, eigendecomposition, UQ, MISMIP+ synthetic tests |
| `figures/`    | Generated figures (`strain_pipeline/`: the GRL manuscript's current Figs. 1-5, see below)          |
| `output/logs/`| Run logs, grouped by stage (`prior`, `inversion`, `lcurve`, `eigendec`, `forward`, `mismip`)    |
| `manuscript/` | Methods text and bibliography                                                                   |
| `docs/`       | Geometry-consistency rule (`geometry_consistency.md`); notes on tlm_adjoint / pyadjoint Hessian issues |
| `data/`, `mesh/`, `results/` | Inputs, meshes/checkpoints and large numerical outputs (not tracked, ~57 GB)      |

## Pipeline

```
source ~/venv-firedrake/bin/activate && export OMP_NUM_THREADS=1

scripts/prepare_apres.py                      ApRES -> Legendre-fit eps_zz (data/apres_legendre_fits.h5)
scripts/lcurve_hybrid_apres.py <L_km>         prior correlation-length L-curve
scripts/physical_prior.py                     prior means: balance friction C_w0, thermo fluidity A_prior
scripts/invert.py --model {ssa,hybrid} --friction {weertman,rc} [--no-apres]
                                              MAP (prior-whitened L-BFGS from the prior mean)
scripts/run_eigendec.py --method gn           Gauss-Newton prior-preconditioned Hessian (hybrid)
scripts/run_eigendec_ssa.py                   same for SSA
scripts/uq_sample_posterior.py                Laplace posterior samples
scripts/uq_sample_forward{,_ssa}.py --idx N   50-yr forward per sample
scripts/uq_sample_aggregate.py                ensemble statistics
scripts/uq_ridgeline_compare.py               hybrid vs SSA mass-loss figure
```

Conventions (prior form, misfit normalization, file naming) are documented in the script headers.

## Paper chain (rc friction, bilaplacian prior)

The figures and numbers in the paper come from the MAPs
`sweep_{ssa,hybrid}_rc_bilap_d4e-04_L3km_thwaites_fixed_tau5km` (rc friction, bilaplacian
prior delta 4e-4, ELL 3 km, `VV_MESH=thwaites_fixed VV_TAU_SMOOTH=5000`), carried through:

```
scripts/sweep_regularization.py               MAP inversions over the (delta, ELL) prior surface
scripts/forward.py --gradient                 taped 50-yr ISMIP7 forward + dVAF/dtheta
scripts/eigendec.py --K 1500                  GN Hessian eigendecomposition at the MAP
scripts/uq_vaf_pcg.py                         exact posterior sigma of VAF(t) by PCG
scripts/uq_vaf.py --K 1500                    K-mode (low-rank) posterior sigma
scripts/paper_extract.py --model M            fields behind the paper figures (Firedrake)
scripts/paper_extract_profiles.py --model M   modelled ApRES eps_zz profiles (Firedrake)
scripts/paper_figures.py                      every paper figure + the retreat-rate table
```

`output/launch/launch_bilap_chain.sh` runs the forward, eigendecomposition and UQ stages.
`paper_figures.py` is plain matplotlib on archived results; `figstyle.py` holds the
figure style (Morrow et al. 2026). It writes `figures/paper/`, which is copied to the
GRL manuscript repository, hoffmaao/paper-thwaites-nonlinear-deformation.

## GRL manuscript figures

The GRL manuscript ("Basal properties drive changes in englacial deformation on Thwaites
Glacier", on Overleaf) uses the same style:

```
scripts/manuscript_prepare.py    raw ApRES bursts, strain profiles, MEaSUREs and BedMachine
                                 -> results/manuscript/{apres,maps}.npz (run once)
scripts/manuscript_figures.py    Figs. 1-7 -> figures/manuscript/fig<NN>.{png,pdf}, numbers.json
```

`apres_dat.py` reads the ApRES `.dat` bursts for the return-power section. The manuscript
no longer uses these seven; its figures are listed below.

### Figures currently in the manuscript (ApRES strain pipeline)

The manuscript in hoffmaao/paper-thwaites-nonlinear-deformation still uses the five
figures in `figures/strain_pipeline/` (`fig01_overview` ... `fig05_legendre`), not the
output of `manuscript_figures.py`. They come from a separate pipeline that processes the
raw repeat bursts itself, with the `apres/` package, instead of reading
`data/apres_legendre_fits.h5`. Its Legendre fit (`fit_legendre.py`, `apres/legendre.py`)
is not the same as the one in `manuscript_prepare.py`, so check which pipeline a number in
the text comes from until the manuscript switches to one set of figures. The scripts'
numbers are not the paper's, so figure labels avoid numbers and the captions supply them:

| Paper | Script | Output |
|---|---|---|
| Fig. 1 | `scripts/figures/fig01_overview.py` | `figures/strain_pipeline/fig01_overview` |
| Fig. 2 | `scripts/figures/fig02_section.py` | `figures/strain_pipeline/fig02_section` |
| Fig. 3 | `scripts/figures/fig05_legendre.py` | `figures/strain_pipeline/fig05_legendre` |
| Fig. 4 | `scripts/figures/fig04_map.py` | `figures/strain_pipeline/fig04_map` |
| Fig. 5 | `scripts/paper_figures.py` (F_apres) | `figures/paper/F_apres` |
| Fig. 6 | `scripts/paper_figures.py` (F_retreat) | `figures/paper/F_retreat_rates` |
| Fig. 7 | `scripts/plot_dissipation.py` | `figures/paper/F_dissipation` |
| Figs. S1-S8 | `scripts/paper_figures.py` (S1-S8) | `figures/paper/S*` |
| Fig. S9 | `scripts/figures/fig03_strain_regression.py` | `figures/strain_pipeline/fig03_strain_regression` |
| Fig. S10 | `scripts/figures/figS10_bed_link.py` | `figures/strain_pipeline/figS10_bed_link`, `results/vertical_velocity/bed_link.json`; needs `results/paper/extract_ssa.npz` (`paper_extract.py`) and `results/paper/dissipation_ssa.npz` (`dissipation_partition.py`), Firedrake scripts outside `run_pipeline.sh` |

```
scripts/run_pipeline.sh            all stages below, in order (env: PY_ANALYSIS, PY_FIGURES, JOBS)
scripts/run_pipeline.sh figures    figures only, from existing results/

scripts/build_site_catalog.py          data/{2022-2023,2023-2024} bursts -> results/sites/{site_catalog,repeat_pairs}.csv
scripts/compute_vertical_velocities.py repeat-pair cross-correlation -> results/vertical_velocity/{profiles.nc,site_summary.csv} (~30 min, 4 cores)
scripts/compute_strain_rates.py        -> results/vertical_velocity/strain_profiles.nc
scripts/compute_surface_strain.py      ASE annual velocity mosaics -> results/sites/surface_strain.csv
scripts/fit_legendre.py                -> results/vertical_velocity/legendre_fits.csv
scripts/figures/fig0{1..5}_*.py        -> figures/strain_pipeline/fig0N_*.{png,pdf} (PyGMT for Figs. 1 and 4)
scripts/figures/figS10_bed_link.py     -> figures/strain_pipeline/figS10_bed_link.{png,pdf}, results/vertical_velocity/bed_link.json
```

These figures follow the paper's figure style (`scripts/figstyle.py`, after Morrow et
al. 2026) through `scripts/figures/style.py`, which also carries it over to PyGMT; its
module docstring lists the conventions and where these figures depart from them.

External datasets (BedMachine, MEaSUREs, ASE velocity mosaics, REMA, MOA, coastlines,
IceLines, grounding lines, the vibroseismic bed attributes) are located through
`scripts/paths.py`, with an environment variable override for each. Fig. 1 also outlines
the model domain from the mesh extract written by `paper_extract.py` (`MODEL_EXTRACT`,
default `results/paper/extract_ssa.npz`); Fig. S10 reads it too, with the basal
frictional power from `dissipation_partition.py` (`MODEL_DISSIPATION`, default
`results/paper/dissipation_ssa.npz`).
