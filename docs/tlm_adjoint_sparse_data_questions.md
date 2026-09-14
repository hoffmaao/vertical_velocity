# Questions for icepack / firedrake / tlm_adjoint developers

## Context

We are doing Bayesian uncertainty quantification on an icepack `HybridModel`
inversion that assimilates two types of data:

1. **Surface velocity** (MEaSUREs 450 m), interpolated onto a
   `VertexOnlyMesh` at the surface (ζ=1) of an `ExtrudedMesh` and compared
   to model `velocity`.
2. **ApRES vertical strain rates** at ~91 GHOST stations, each as a
   short depth profile, interpolated onto a 3D `VertexOnlyMesh` and
   compared to the model `horizontal_strain_rate` (via
   `icepack.models.hybrid.horizontal_strain_rate`).

Mesh: hires Thwaites trimmed mesh, 67,600 vertices, extruded with
`layers=1`. Controls are
`θ_A = log(fluidity/A₀)` and `θ_C = log(friction/C₀)` on
`FunctionSpace(mesh, "CG", 1, vfamily="DG", vdegree=0)`.

## What works

`tlm_adjoint.firedrake.compute_gradient(J, (θ_A, θ_C))` returns the correct
adjoint gradient of the misfit through the entire chain:

```
θ_A, θ_C
  → interpolate(A₀·exp(θ_A))          # UFL pointwise
  → solver.diagnostic_solve(...)       # icepack HybridModel
  → Function(Δ_vel).interpolate(u[0])  # VertexOnlyMesh interpolate
  → icepack.models.hybrid.horizontal_strain_rate(...)
  → Function(Δ_apres).interpolate(...) # VertexOnlyMesh interpolate
  → 0.5/N * ((δu/σ)² + ...) * dx       # assemble per-mesh
```

We verified this against finite-difference of J. `|grad|` is consistent
across our inversion runs.

A taped 50-yr prognostic forward (600 monthly steps of
`solver.diagnostic_solve` + `solver.prognostic_solve`) followed by
`compute_gradient(J=VAF, (θ_A, θ_C))` *also* works — we get sensible
`||∂VAF/∂θ||` that scales correctly with the time horizon.

So **first-order adjoint** through every operation we care about is fine.

## What does not work

**`tlm_adjoint.firedrake.CachedHessian.action([θ_A, θ_C], [v_A, v_C])`
returns ≈ 0** on the same misfit. Specifically, for a random unit
direction (v_A, v_C):

| method | ‖Hv_A‖ | ‖Hv_C‖ |
|---|---|---|
| finite-difference of `compute_gradient` (EPS_FD=1e-5) | 1.79e+1 | 4.69e+1 |
| `CachedHessian.action` then `riesz_representation("L2")` | 5.5e−5 | 1.9e−4 |

The TLM result is six orders of magnitude smaller than FD — consistent with
the perturbation **silently failing to propagate forward** through some
operation along the tape.

We also tried **pyadjoint** as a cross-check:

```python
from firedrake.adjoint import continue_annotation, ReducedFunctional, Control
# ... same forward, with continue_annotation() ...
J_red = ReducedFunctional(J, [Control(θ_A), Control(θ_C)])
dJ = J_red.derivative()      # gradient via pyadjoint
Hv = J_red.hessian([v_A, v_C])  # Hessian-vector via pyadjoint
```

The pyadjoint **gradient itself** comes out as 1.88e−7 (vs 1.90e−2 from
`tlm_adjoint.compute_gradient` on the same forward). So pyadjoint's tape
isn't even capturing the first-order dependency through our forward —
its 14-block tape is much shorter than the equivalent operations the
adjoint walk visits via `tlm_adjoint`.

This means **neither framework gives us a working Hessian action** for
this setup, despite `tlm_adjoint` adjoint working.

We tried the obvious mitigations:
- `from tlm_adjoint.firedrake import Function as TLMFunction` and
  `TLMFunction(Q, name=..., static=True)` for the controls
- `configure_checkpointing("memory", {"drop_references": False})` before
  `start_manager()`
- Both at once

None changed the TLM result.

## Why this matters

We do Hessian-eigendecomposition-based UQ (Isaac et al. 2015 / fenics_ice
convention): the prior-preconditioned misfit Hessian's eigenpairs
let us compute σ_post for a QoI via

```
σ²(Q) = ∇Q^T Γ_prior ∇Q − Σᵢ (λᵢ / (λᵢ+1)) (∇Q · wᵢ)²
```

We currently compute the Hessian-vector products in our eigensolve via
finite differences of the adjoint gradient (`scripts/run_eigendec.py`),
which is the only path open to us. FD noise pollutes the small-λ tail of
the spectrum: at K=1500 modes computed, 202 are spurious negative, and
many "informative" modes (λ ~ 5–14) we suspect are FD-noise-perturbed
true modes. Our UQ result for VAF(50yr) shows a 0.16% variance
reduction, dominated by mode indices 302–455 of intermediate eigenvalue —
suspicious if true eigenvectors would project differently against the
VAF gradient.

`CachedHessian.action` would let us replace FD with exact second-order
adjoint and clean this up.

## Minimal reproducers

Two test scripts already isolate the failure, both in
`scripts/`:

- `test_cached_hessian.py` — `tlm_adjoint` version, gives `||Hv|| ≈ 1e−4`
  vs FD's `~10`.
- `test_pyadjoint_hessian.py` — pyadjoint version, gives `|grad| ≈ 1e−7`
  for the gradient itself (the Hessian then trivially fails).

Both can run with `--component vel` to drop the ApRES branch (same
failure), confirming the issue is in the shared
`diagnostic_solve` + `VertexOnlyMesh.interpolate` path, not in the
ApRES-specific code.

## Comparison: where this same UQ pipeline works

UQ_forward (the same author's earlier project,
`/media/andrew/wd1/projects/UQ_forward/venable/`) uses the **same**
Isaac-et-al. machinery and **the same** `CachedHessian.action` + `eigsh`
pattern, and it works. Differences from our project:

| | UQ_forward Venable | this project (Thwaites + ApRES) |
|---|---|---|
| icepack model | `IceStream` (2D depth-averaged) | `HybridModel` (3D extruded) |
| misfit measure | integral over domain | sparse via `VertexOnlyMesh` |
| `depth_average` on the tape | no | indirectly (only in setup) |
| `horizontal_strain_rate` | no | yes |

So **the combination "HybridModel diagnostic_solve + VertexOnlyMesh
interpolate"** is what tips the second-order path over.

## Questions, by team

### To the firedrake team

1. **Is `VertexOnlyMesh.interpolate` annotated for TLM (forward-mode)
   in `tlm_adjoint`?** It supports reverse-mode adjoint (we get a
   gradient from `compute_gradient` through it), but our experiments
   suggest forward-mode is missing or no-ops the perturbation.
   `firedrake/adjoint_utils/blocks/solving.py` has
   `evaluate_tlm_component` for variational solves, but we have not
   located the equivalent for VOM-target `Function.interpolate(expr)`.

2. **Why does pyadjoint's
   `ReducedFunctional.derivative()` return ~0 on a forward that
   includes `VertexOnlyMesh.interpolate(u_component)`?** We see only
   14 blocks recorded on the pyadjoint tape, which seems short for the
   operations between the controls and the assembled misfit. Is VOM
   interpolation skipped in the pyadjoint annotation path under some
   condition (e.g., when the target VOM Function is created inside the
   taped region, or when its function space is `DG0` on the VOM)?

3. **What is the recommended pattern for sparse-data assimilation
   that is fully second-order-adjoint compatible?** Several inverse
   problems beyond ours need this (point GPS, point altimetry,
   IceBridge, ApRES, dense in-situ networks). If `VertexOnlyMesh` is
   not the supported route, what is?

### To the icepack team

4. **Is `solver.diagnostic_solve(...)` on `HybridModel` annotated for
   TLM under `tlm_adjoint`?** The `IceStream` version is clearly fine
   in UQ_forward; we don't know whether `HybridModel`'s extruded-mesh
   variational solve has any annotation gap. (The comment at
   `icepack/src/icepack/solvers/flow_solver.py:407` shows the team
   considered `tlm_adjoint` when wiring BCs — is there a known list of
   adjoint/TLM-supported solver entries?)

5. **`solver.prognostic_solve` over 600 monthly steps with
   `compute_gradient` works for us.** Do you have any known issues
   with the `LaxWendroff` / `ImplicitEuler` prognostic schemes when
   recorded under a long `tlm_adjoint` tape with the
   `compute_gradient` walk-back (e.g., re-use of `_thickness_old`
   across steps)? We saw no problem at 50 yr; we'd like to confirm.

6. **Is `icepack.models.hybrid.horizontal_strain_rate` purely UFL?**
   Looking at the code suggests yes, but we'd like to confirm there's
   no hidden non-differentiable branch that would silently zero out
   the TLM pathway. Same question for `icepack.compute_surface` and
   `icepack.utilities.lift3d`.

### To the tlm_adjoint team

7. **Is there a published list of firedrake operations that are
   adjoint-annotated but not TLM-annotated?** Knowing what to avoid
   would let us refactor a forward to stay within the supported
   subset. Our experiments suggest at least `VertexOnlyMesh` →
   `Function.interpolate(expr)` falls in this category for
   `tlm_adjoint`.

8. **When `CachedHessian.action` is called on a tape that contains an
   operation lacking TLM annotation, does it silently return zero in
   that subspace (as we observe), or is there a way to ask it to
   raise/warn?** A diagnostic mode would have saved us a day of
   debugging.

9. **Is there a recommended approach for second-order adjoints
   through `VertexOnlyMesh` interpolation specifically?** We can
   provide patches if there is a documented design we should follow.

## What we already have

- `scripts/run_eigendec.py` — current FD-based eigensolve, K=1500
  modes with HDF5 matvec checkpointing for restart.
- `scripts/test_cached_hessian.py` — tlm_adjoint reproducer.
- `scripts/test_pyadjoint_hessian.py` — pyadjoint reproducer.
- `scripts/run_forward.py` — taped 50-yr prognostic with
  `compute_gradient` for ∂VAF(T)/∂θ.
- `scripts/uq_vaf.py` — Isaac et al. propagation given gradient +
  eigenpairs.
- 1298 positive eigenpairs at `results/eigenvectors.npz` (FD-based,
  K=1500 ARPACK with `which="LM"`).
- UQ_forward project at `/media/andrew/wd1/projects/UQ_forward/` for
  comparison (same author).

## What we are willing to invest

We can:
- Provide a minimal MWE without the icepack dependency if a firedrake
  developer wants one (a manufactured PDE + VOM-target interpolate +
  CachedHessian).
- Patch icepack (`HybridModel`) to use `tlm_adjoint`'s
  `EquationSolver` instead of `firedrake.NonlinearVariationalSolver`
  if that is the recommended path.
- Implement the TLM annotation for `VertexOnlyMesh.interpolate`
  ourselves if a tlm_adjoint developer can point us at the pattern.
