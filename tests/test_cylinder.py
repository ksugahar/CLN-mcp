import math
import json
import pytest
from cln import TOmegaCLN, ladder_impedance, wire_internal_impedance, wire_cln_s0_zero, cylinder_mesh

A, H, SIG, MU = 0.01, 0.01, 1e6, 4e-7*math.pi


@pytest.fixture(scope="module")
def mesh():
    return cylinder_mesh(A, H, maxh=A/2, maxh_conductor=A/10)


@pytest.mark.parametrize("bc", ["Neumann", "Dirichlet"])
@pytest.mark.fe
def test_s0_zero_matches_exact_elements(mesh, bc, record_property):
    solver = TOmegaCLN(mesh, SIG, MU, A, H, s0=0.0, bc=bc, order=2)
    R, L, _ = solver.run(nmax=4)
    _record_solver(record_property, solver, R, L)
    assert all(d["converged"] and d["relative_residual"] <= solver.tol for d in solver.solve_diagnostics)
    for d in solver.solve_diagnostics:
        assert d["requested_solver"] == "sparsesolv" and d["backend"] == "sparsesolv", d
    for n in range(5):
        v = R[n] if n % 2 == 0 else L[n]
        assert v == pytest.approx(wire_cln_s0_zero(n, A, SIG, MU), rel=5e-3), (bc, n)


@pytest.mark.parametrize("bc", ["Neumann", "Dirichlet"])
@pytest.mark.fe
def test_expansion_point_impedance(mesh, bc, record_property):
    s0 = 2*math.pi*1e5
    solver = TOmegaCLN(mesh, SIG, MU, A, H, s0=s0, bc=bc, order=2, solver="direct")
    R, L, _ = solver.run(nmax=4)
    _record_solver(record_property, solver, R, L)
    assert all(d["requested_solver"] == "direct" and d["backend"] == "sparsecholesky"
               for d in solver.solve_diagnostics), solver.solve_diagnostics
    for f, tol in ((1e5, 2e-3),(5e4, 1e-2), (2e5, 1e-2)):
        s = 2*math.pi*f
        ze = wire_internal_impedance(s, A, SIG, MU).real
        assert ladder_impedance(R, L, s, s0) == pytest.approx(ze, rel=tol), (bc, f)


def test_ladder_reduces_to_R0_at_s0():
    assert ladder_impedance({0: 2.0, 2: 5.0}, {1: 1.0, 3: 1.0}, 3.0, 3.0 + 1e-14) == pytest.approx(2.0)


def _record_solver(record_property, solver, resistance=None, inductance=None):
    """Machine-readable observations for an admitted qualification collector."""
    record_property("cln_solve", json.dumps({
        "s0": solver.s0, "bc": solver.bc, "solver": solver.solver,
        "outer_tolerance": solver.tol, "inner_tolerance": solver._iccg_tolerance,
        "diagnostics": solver.solve_diagnostics, "iterations": solver.iterations,
        "residual_over_tol": [d["relative_residual"] / solver.tol for d in solver.solve_diagnostics],
        "R": resistance, "L": inductance}, allow_nan=False))


@pytest.mark.parametrize("s0", [0.0, 2*math.pi*1e5])
@pytest.mark.fe
def test_explicit_direct_backend(mesh, record_property, s0):
    """Guards: direct remains explicitly selectable. Assumes: admitted circular wire."""
    solver = TOmegaCLN(mesh, SIG, MU, A, H, solver="direct", order=2, s0=s0)
    R, L, _ = solver.run(nmax=2)
    _record_solver(record_property, solver, R, L)
    for d in solver.solve_diagnostics:
        assert d["requested_solver"] == "direct", d
        assert d["backend"] == ("sparsecholesky" if s0 > 0 else "direct") and d["converged"], d
        assert d["relative_residual"] <= solver.tol, d


@pytest.mark.fe
def test_iccg_unattainable_tolerance_raises(mesh, record_property):
    """Guards: no coefficients after unattained true residual. Assumes: admitted FE."""
    solver = TOmegaCLN(mesh, SIG, MU, A, H, solver="sparsesolv", tol=1e-30,
                       maxiter=1, order=2)
    with pytest.raises(RuntimeError, match="did not converge"):
        solver.run(nmax=0)
    _record_solver(record_property, solver)
    assert solver.solve_diagnostics and not solver.solve_diagnostics[-1]["converged"], solver.solve_diagnostics


@pytest.mark.fe
def test_bad_iccg_iterate_cannot_switch_backend(mesh, monkeypatch, record_property):
    """Guards: true-residual failure and no fallback. Assumes: FE fault injection."""
    from types import SimpleNamespace
    solver = TOmegaCLN(mesh, SIG, MU, A, H, solver="sparsesolv", maxiter=1, order=2)
    class ZeroStep:
        last_result = SimpleNamespace(iterations=0)
        def __mul__(self, rhs):
            value = rhs.CreateVector()
            value[:] = 0
            return value
    def forbidden_direct():
        raise AssertionError("ICCG failure attempted an implicit direct solve")
    solver._inv = ZeroStep()
    monkeypatch.setattr(solver, "_regularized_inverse", forbidden_direct)
    with pytest.raises(RuntimeError, match="did not converge"):
        solver.run(nmax=0)
    _record_solver(record_property, solver)
    assert solver.solve_diagnostics[-1]["backend"] == "sparsesolv", solver.solve_diagnostics
