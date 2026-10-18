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
| `figures/`    | Generated figures                                                                               |
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
paper repository, hoffmaao/paper-thwaites-vertical-velocity.
