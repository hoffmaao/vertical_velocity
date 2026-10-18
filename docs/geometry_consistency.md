# Geometry consistency: never smooth the surface

## The rule

`h`, `s` and `b` must satisfy the model's own flotation relation

    s = max(b + h, (1 - rho_I/rho_W) h)        (icepack.compute_surface)

everywhere, at every point of the pipeline. Filter the *inputs* (`h`, `b`) if
you must, then derive `s` from the filtered pair. Never filter `s` itself, and
re-derive `s` after anything that touches `h` or `b` - including a thickness
clamp.

`scripts/check_geometry.py` audits a mesh file against this.

## What went wrong (2026-09-23)

`build_mesh*.py` derived `s` correctly and then Helmholtz-smoothed it over 2 km.
On `mesh/thwaites.h5` that left:

| | stored (smoothed) surface | consistent surface |
|---|---|---|
| \|s - compute_surface(h,b)\| | up to 362 m; > 10 m at 21376/67600 nodes (31.6%) | 0 |
| nodes with flotation factor < -0.01 | 1894 | 0 |
| \|tau_d\| median / p99, smoothed over 2 km | 0.0812 / 0.2038 MPa | 0.0850 / 0.2655 MPa |

Two consequences:

1. **Negative basal drag.** The friction laws scale the coefficient by
   `phi = 1 - rho_W g max(0, h - s)/(rho_I g h)`. A smoothed surface puts ice
   below its hydrostatic height, `phi` goes negative, and the basal stress
   becomes a momentum *source*. 1894 nodes were in that state in every inversion
   run on this mesh.
2. **The forward cannot start.** The first prognostic step calls
   `compute_surface` and applies the whole 362 m difference at once. Peak speed
   went 4081 -> 16824 m/yr in one step and the run diverged within three
   (`output/logs/forward/smoke_ismip7.log`). This is not a solver problem: the
   old `lu` options do the same, and the trust-region rescue and dt subcycling
   both fire and both fail. Imposing the consistent surface at t = 0 instead
   makes stepping stable, but then the MAP velocity primes at 19971 m/yr,
   because the MAP was fitted at a surface the forward does not use.

The smoothing bought nothing: `physical_prior.py` already Helmholtz-smooths the
driving stress when it builds the `C_w0` anchor, and after that smoothing the two
surfaces give the same statistics (table above). That is why the anchor survives
the fix nearly unchanged.

## What the other codebases do

None of icepack, icepack2, icepack_tools, icepack2_tools or gia-icepack ever
smooths a geometry field. gia-icepack applies the *same* 2 km Helmholtz operator
(`scripts/ase_model.py:82-91`) to the grounded mask, the melt rate and the
apparent mass balance - never to `h`, `s` or `b`. ismip7 cell-averages the
BedMachine *inputs* onto the mesh and derives `s` from the averaged pair
(`icepack2_tools/geometry.py:106-177`, `antarctica/scripts/simulation.py:348-361`),
then funnels every later thickness change through one routine that re-derives
`s`, `_lift_h` (`simulation.py:1463-1494`).

The ancestor of our smoother is real but was never in a forward path: ismip7's
`antarctica/scripts/diagnostic_solve.py:105-117` and two `greenland/` scripts do
derive-then-smooth with the same `alpha = 2e3`. They are standalone snapshot
scripts. We put those lines in the mesh builders, where everything inherits them.

These tools actively depend on the consistency: `icepack2_tools/dual_friction.py:78-89`
uses the model surface "so `N` vanishes EXACTLY at flotation - the property that
gives exact-zero Coulomb drag on shelves", and `icepack_tools/test/density_test.py`
exists to catch a surface that disagrees with `(h, b)` by *a few metres*.

## The fix

- `build_mesh.py`, `build_mesh_hires.py`, `build_mesh_coarse.py`: derive `s`,
  do not smooth it.
- `VV_MESH` selects the mesh (`physical_prior.py`), so a corrected mesh can be
  built and inverted while another is still in use:

      VV_MESH=thwaites_consistent python scripts/build_mesh_hires.py
      VV_MESH=thwaites_consistent python scripts/physical_prior.py
      VV_MESH=thwaites_consistent python scripts/invert.py --model hybrid \
          --friction weertman --delta 1e-5 --ell 2000.0 --name map_..._consistent

- `forward.py` funnels every geometry change through `lift_geometry()`, which
  clamps `h` then re-derives `s`, mirroring ismip7's `_lift_h`.

If you ever do want a smoothed surface, derive the thickness from it instead -
`icepack_tools/adapt_mesh.py:721-732` (`surface_route_thickness`,
`h = min(s - b, s rho_w/(rho_w - rho_i))`). Move one of `(h, s)` and derive the
other; never both independently.

## Still open

- `mesh/thwaites.h5` and every inversion on it, including the regularization
  sweep, carry the inconsistency. The sweep's purpose is to choose `(delta, ELL)`
  and that choice is unlikely to hinge on this, but any MAP from it is unusable
  as a forward initial state.
- The apparent mass balance in `forward.py` cancels the initial flux divergence
  (+-1500 m/yr on this state) by solving the transport scheme's own
  source-response operator. A single zero-accumulation step is not enough under
  Lax-Wendroff, whose response to the source is not the identity; ismip7 makes
  the same point at `simulation.py:1194-1212`.
- The structural cure for the flux-divergence spike is a dH/dt-constrained
  inversion, so the initial state is flux-consistent with the observed geometry
  rather than corrected after the fact (`icepack2_tools/obs_dhdt.py:1-8`).
