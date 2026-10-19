r"""Solver presets for the prognostic forward runs, ported from the ismip7 stack.

**There is no custom preconditioner to port.** Every momentum and transport solve
in /media/andrew/wd1/projects/ismip7 is a direct MUMPS LU factorisation: no
fieldsplit, no multigrid on the velocity operator, no Schur complement, no
matrix-free. The project says so itself (antarctica/scripts/batch_runners/
readme.md:288-290): "There is no iterative solver path in the model today: every
solve is a direct MUMPS factorisation, which is what sets the memory. If the
768 GB nodes turn out to be scarce, the fix is a field split that eliminates the
cell-wise stress and traction blocks and puts multigrid on the velocity
operator." That fix is named as future work and is not implemented.

What ismip7 *does* have, and what is worth having here, is a robustness stack
around the direct solve, tuned against real grounding-line retreat failures:

  * ``nleqerr`` line search, no divergence test and no step-size test, so only
    the residual tolerances and the iteration cap can stop Newton
    (simulation.py:663-692);
  * MUMPS null-pivot handling, which is what survives h -> 0 at the calving
    front: icntl_14 (working-memory headroom), icntl_24 = 1 (detect null
    pivots), cntl_3 = 1e-12 (null-pivot threshold);
  * a dt subcycling ladder for steps the diagnostic solve cannot get through;
  * an escalation ladder when a step fails: retry with a trust-region Newton,
    then retry the step at dt/4 and dt/16 (simulation.py:1405-1461, 1720-1747).

Two ismip7 choices deliberately do NOT transfer:

  * ``quadrature_degree = 4``: classic icepack derives the quadrature degree
    from the model (icepack/solvers/flow_solver.py:349, :426), so this is not a
    knob here.
  * implicit-Euler transport: ismip7's is a hand-written upwind DG0 form whose
    M-matrix structure is what keeps the update monotone
    (simulation.py:1570-1610). icepack's ``ImplicitEuler`` is a different
    scheme, which icepack itself documents as "included for backward
    compatibility only ... the Lax-Wendroff scheme is preferable by far"
    (icepack/solvers/flow_solver.py:465-472), and which omits the
    quadrature-degree fix its Lax-Wendroff sibling applies: measured at ~10
    minutes per step on our mesh, against seconds for Lax-Wendroff. The presets
    therefore keep icepack's Lax-Wendroff default; ``--transport
    implicit-euler`` is there if you want to compare.

ismip7 runs icepack2's dual (velocity / membrane stress / basal traction)
formulation, whose residual-level conditioning devices - the composite-rheology
regulariser, the ocean drag, the RC identity block - have no counterpart in the
classic icepack model we use here, so only the PETSc options and the ladder
transfer. The ``greenland`` preset is the one ismip7 configuration that *was*
written against classic icepack (greenland/diagnostic_solves.py:34-83), and is
kept here for comparison; note that its ``pc_hypre_type`` key is inert next to
``pc_type: bjacobi`` and is dropped below rather than copied.

Two ismip7 results worth knowing before running anything long:
  * their RC (regularised Coulomb) forward was prognostically unstable - the
    diagnostic velocity roughly doubled per step, independent of the transport
    scheme and not curable by shelf drag - and they pivoted the forward to Budd.
    Our sweep includes rc points, so expect trouble there and check early.
  * reducing dt does not always help: at their blow-up, dt = 0.01 diverged
    *faster* than dt = 0.1. The dt ladder is for local stiffness, not for a
    forward that is already unstable.
"""

# MUMPS knobs that make the direct factorisation survive h -> 0 at the front.
# ismip7 antarctica/scripts/simulation.py:663-692.
_MUMPS = {
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_14": 400,      # +400% working-memory headroom
    "mat_mumps_icntl_24": 1,        # detect null pivots
    "mat_mumps_cntl_3": 1e-12,      # null-pivot threshold
}

# The project baseline: what run_forward.py and the inversions solve with.
_LU_DIAGNOSTIC = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_max_it": 100, "snes_rtol": 1e-8,
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
}
_LU_PROGNOSTIC = {
    "snes_type": "ksponly",
    "ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
}

# ismip7 production diagnostic options. snes_divergence_tolerance = -1 disables
# the divergence test and snes_stol = 0 the step-size test, so a slowly
# descending Newton is not killed off; ksp_type is gmres over an exact LU, which
# converges in one iteration (kept verbatim for comparability).
_ISMIP7_DIAGNOSTIC = {
    "snes_type": "newtonls", "snes_linesearch_type": "nleqerr",
    "snes_max_it": 200,
    "snes_divergence_tolerance": -1,
    "snes_stol": 0.0,
    "ksp_type": "gmres",
    **_MUMPS,
}
# The rescue attempt: same operator, trust-region Newton, patient. ismip7
# escalates to this on a failed step (simulation.py:1405-1461) and it is what
# got them through geometries where nleqerr hit a wall.
_ISMIP7_RESCUE = {
    **_ISMIP7_DIAGNOSTIC,
    "snes_type": "newtontr",
    "snes_max_it": 600,
}
# ismip7's transport solve: implicit upwind DG0, an M-matrix system, solved
# directly (simulation.py:1591-1610).
_ISMIP7_PROGNOSTIC = dict(_LU_PROGNOSTIC)
# Their one iterative solve, used for the transport step inside the
# time-dependent inversion (inversion_icepack2.py:958-968). Cheap on memory,
# which is the binding constraint of the direct path.
_ITER_PROGNOSTIC = {
    "snes_type": "ksponly",
    "ksp_type": "gmres", "pc_type": "bjacobi", "sub_pc_type": "ilu",
    "ksp_rtol": 1e-10,
}
# The one ismip7 configuration written against classic icepack
# (greenland/diagnostic_solves.py:34-83): a damped line search to a rough
# answer, then trust region to finish. The inert pc_hypre_type key is dropped.
_GREENLAND_DIAGNOSTIC = {
    "snes_type": "newtonls", "snes_linesearch_type": "bt",
    "snes_linesearch_order": 2, "snes_linesearch_max_it": 2500,
    "snes_linesearch_damping": 0.05,
    "snes_max_it": 5000, "snes_stol": 1e-6, "snes_rtol": 1e-5,
    "ksp_type": "bcgs", "ksp_max_it": 2500, "ksp_rtol": 1e-8, "ksp_atol": 1e-4,
    "ksp_converged_maxits": True,
    "pc_type": "bjacobi", "pc_factor_mat_solver_type": "mumps",
    "pc_factor_shift_amount": 1e-10,
}
_GREENLAND_RESCUE = {
    "snes_type": "newtontr", "snes_tr_delta0": 1e5, "snes_tr_fallback_type": "dogleg",
    "snes_max_it": 5000, "snes_stol": 1e-8, "snes_rtol": 1e-8,
    "ksp_type": "bcgs", "ksp_max_it": 100000, "ksp_rtol": 1e-16, "ksp_atol": 1e-16,
    "pc_type": "bjacobi", "pc_factor_mat_solver_type": "mumps",
    "pc_factor_shift_amount": 1e-10,
}

SOLVERS = {
    "lu": {
        "diagnostic": _LU_DIAGNOSTIC,
        "prognostic": _LU_PROGNOSTIC,
        "rescue": None,
        "transport": None,
        "doc": "project baseline: bt line search, direct MUMPS, icepack's "
               "default Lax-Wendroff transport, no rescue",
    },
    "ismip7": {
        "diagnostic": _ISMIP7_DIAGNOSTIC,
        "prognostic": _ISMIP7_PROGNOSTIC,
        "rescue": _ISMIP7_RESCUE,
        "transport": None,
        "doc": "ismip7 production: nleqerr + MUMPS null-pivot handling, "
               "trust-region rescue, dt subcycling",
    },
    "ismip7-iter-transport": {
        "diagnostic": _ISMIP7_DIAGNOSTIC,
        "prognostic": _ITER_PROGNOSTIC,
        "rescue": _ISMIP7_RESCUE,
        "transport": None,
        "doc": "ismip7 diagnostic, transport by GMRES/bjacobi-ILU (lower memory)",
    },
    "greenland": {
        "diagnostic": _GREENLAND_DIAGNOSTIC,
        "prognostic": _LU_PROGNOSTIC,
        "rescue": _GREENLAND_RESCUE,
        "transport": None,
        "doc": "ismip7's classic-icepack Greenland pair: damped bt, then "
               "trust-region dogleg",
    },
}

# dt ladder for a step the diagnostic solve cannot get through: retry the step
# as m substeps. ismip7 uses 1,4,16 by default and 1,4,16,64 in the core matrix
# (simulation.py:1686-1698, run_core_matrix.sh:94).
SUBCYCLES = (1, 4, 16)


def describe_solver(name):
    return SOLVERS[name]["doc"]
