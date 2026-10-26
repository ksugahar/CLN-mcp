"""Retained fields must not follow later mutations of solver work buffers."""
import math
import numpy as np
import pytest
from ngsolve import GridFunction, CF, x, y, Integrate
from cln import TOmegaCLN, cylinder_mesh


@pytest.fixture(scope='module')
def mesh():
    return cylinder_mesh(.01, .01, maxh=.005, maxh_conductor=.003)


@pytest.mark.parametrize('s0', [0., 2*math.pi*1e5])
@pytest.mark.parametrize('bc', ['Neumann', 'Dirichlet'])
def test_stage_expressions_do_not_alias_solve_buffers(mesh, s0, bc):
    solver = TOmegaCLN(mesh, 1e6, 4e-7*math.pi, .01, .01,
                       s0=s0, bc=bc, order=2, solver='direct')
    work = []
    solve = solver._solve
    def capture(rhs, gf):
        result = solve(rhs, gf)
        work.append(result)
        return result
    solver._solve = capture
    t0, j0 = solver.initial()
    r0 = 1/(solver.Pj(j0)+s0*solver.Wm(t0))
    h1 = r0*t0
    t2, j2 = solver.step(t0, j0, h1, solver.Wm(h1))
    fields = (t0, j0, h1, t2, j2)
    norms = [Integrate(f*f*solver._dx, mesh) for f in fields]
    assert all(v > 0 for v in norms)
    # This is the destructive scratch reuse that a refactoring/notebook can do.
    for gf in work:
        gf.vec[:] = 0
    assert [Integrate(f*f*solver._dx, mesh) for f in fields] == pytest.approx(norms, rel=1e-13)
    # A returned GridFunction remains user-editable, but editing H must not
    # silently overwrite the independently retained J expression.
    if hasattr(t0, 'vec'):
        t0.vec[:] = 0
        assert Integrate(j0*j0*solver._dx, mesh) == pytest.approx(norms[1], rel=1e-13)


@pytest.mark.parametrize('s0', [0., 2*math.pi*1e5])
def test_gauge_owns_input_without_reinterpolation(mesh, s0):
    solver = TOmegaCLN(mesh, 1e6, 4e-7*math.pi, .01, .01,
                       s0=s0, order=2, solver='direct')
    original = GridFunction(solver.fes)
    original.Set(CF((-y, x, 0)))
    before = original.vec.FV().NumPy().copy()
    gauged = solver._gauge(original)
    assert np.array_equal(original.vec.FV().NumPy(), before)
    norm = solver.Wm(gauged)
    original.vec[:] = 0
    assert solver.Wm(gauged) == pytest.approx(norm, rel=1e-13)
