import numpy as np
import pytest
from scipy.linalg import expm
from netgen.geom2d import unit_square
from ngsolve import Mesh, GridFunction, Integrate, BilinearForm, grad, dx, x, y
from cln.diffusion import DiffusionCLN


@pytest.fixture(scope='module')
def thermal():
    mesh = Mesh(unit_square.GenerateMesh(maxh=.25))
    solver = DiffusionCLN(mesh, 1+x, 2+y, 1+x, order=3, length=2.5)
    return solver, solver.energy_model(3)


def test_native_projection_in_span_and_outside(thermal):
    solver, model = thermal
    known = np.array([.3, -.02, .004])
    projection = model.project_initial(model.reconstruct(known))
    assert np.allclose(projection.x0, known, rtol=1e-10, atol=1e-13)
    field = GridFunction(solver.fes)
    field.Set(x*(1-x)*y*(1-y))
    # Enforce exact homogeneous constrained coefficients, including roundoff.
    fixed = ~np.array(list(solver.fes.FreeDofs()), dtype=bool)
    field.vec.FV().NumPy()[fixed] = 0
    p = model.project_initial(field)
    error = field-model.reconstruct(p.x0)
    for e in model.field_modes:
        assert abs(Integrate((2+y)*e*error, solver.mesh, order=20)) < 1e-12
    full = 1.25*Integrate((2+y)*field*field, solver.mesh, order=20)
    assert p.diagnostics['full_quadratic_storage'] == pytest.approx(full, rel=1e-11)
    assert p.diagnostics['residual_quadratic_storage'] > 0
    assert model.to_lti(initial_field=field).initial_state == pytest.approx(p.x0)


def test_native_free_response_against_independent_fe_galerkin(thermal):
    solver, model = thermal
    # Assemble independent higher-quadrature original FE matrices, then reduce.
    u, v = solver.fes.TnT()
    k, m = BilinearForm(solver.fes), BilinearForm(solver.fes)
    k += (1+x)*grad(u)*grad(v)*dx(bonus_intorder=12)
    m += (2+y)*u*v*dx(bonus_intorder=12)
    k.Assemble(); m.Assemble()
    E = np.column_stack([e.vec.FV().NumPy() for e in model.field_modes])
    from scipy.sparse import coo_matrix
    def dense(a):
        i, j, values = a.mat.COO()
        return coo_matrix((values, (i, j)), shape=(solver.fes.ndof,)*2).toarray()
    K, M = dense(k), dense(m)
    ag = -np.linalg.solve(E.T @ M @ E, E.T @ K @ E)
    u0 = model.reconstruct([.3, -.02, .004])
    lti = model.to_lti(initial_field=u0)
    for t in (0., .01, .1, .5):
        assert np.allclose(E @ expm(lti.A*t) @ lti.initial_state,
                           E @ expm(ag*t) @ lti.initial_state, rtol=1e-9, atol=1e-12)


def test_native_rejects_boundary_mismatch(thermal):
    solver, model = thermal
    field = GridFunction(solver.fes)
    field.Set(1)
    with pytest.raises(ValueError, match='boundary'):
        model.project_initial(field)
    with pytest.raises(ValueError, match='GridFunction'):
        model.project_initial(1)
