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
| `output/logs/`| Run logs, grouped by stage (`inversion`, `lcurve`, `eigendec`, `forward`, `mismip`)             |
| `manuscript/` | Methods text and bibliography                                                                   |
| `docs/`       | Notes on tlm_adjoint / pyadjoint Hessian issues                                                 |
| `data/`, `mesh/`, `results/` | Inputs, meshes/checkpoints and large numerical outputs (not tracked, ~57 GB)      |

## Pipeline

```
source ~/venv-firedrake/bin/activate && export OMP_NUM_THREADS=1

scripts/prepare_apres.py                      ApRES -> Legendre-fit eps_zz (data/apres_legendre_fits.h5)
scripts/lcurve_hybrid_apres.py <L_km>         prior correlation-length L-curve
scripts/inversion_hires_apres.py              hybrid MAP (velocity + ApRES)
scripts/inversion_ssa_recinos.py              SSA MAP (velocity only; Coulomb or Weertman)
scripts/run_eigendec.py --method gn           Gauss-Newton prior-preconditioned Hessian (hybrid)
scripts/run_eigendec_ssa.py                   same for SSA
scripts/uq_sample_posterior.py                Laplace posterior samples
scripts/uq_sample_forward{,_ssa}.py --idx N   50-yr forward per sample
scripts/uq_sample_aggregate.py                ensemble statistics
scripts/uq_ridgeline_compare.py               hybrid vs SSA mass-loss figure
```

Conventions (prior form, misfit normalization, file naming) are documented in the script headers.
