import math
import numpy as np
import pytest
from scipy.linalg import expm
from ngsolve import Integrate, CF, x, y
from cln import TOmegaCLN, cylinder_mesh


@pytest.fixture(scope='module')
def mesh():
    return cylinder_mesh(.01, .01, maxh=.005, maxh_conductor=.003)


@pytest.mark.parametrize('bc', ['Neumann', 'Dirichlet'])
@pytest.mark.parametrize('s0', [0., 2*math.pi*1e5])
def test_initial_magnetic_modes_energy_and_free_response(mesh, bc, s0):
    solver = TOmegaCLN(mesh, 1e6, 4e-7*math.pi, .01, .01,
                       s0=s0, bc=bc, order=2, solver='direct')
    model = solver.field_model(2)
    known = np.array([.3, -.02])
    h0 = model.reconstruct(known)
    p = model.project_initial(h0)
    assert p.x0 == pytest.approx(known, rel=1e-9)
    assert p.diagnostics['relative_field_error'] < 1e-9
    lti = model.to_lti(initial_field=h0)
    assert np.all(np.linalg.eigvals(lti.A).real < 0)
    # Independent high-quadrature integrals; do not infer loss from shifted R.
    mass = np.array([[Integrate(solver.mu*a*b, mesh, order=16)
                       for b in model.field_modes] for a in model.field_modes])
    loss = np.array([[Integrate(a*b/solver.sigma, mesh, order=16)
                       for b in model.current_modes] for a in model.current_modes])
    ag = -np.linalg.solve(mass, loss)
    port = np.array([Integrate(j[2], mesh, order=16)/solver.length for j in model.current_modes])
    assert (model.metric @ lti.B).ravel() == pytest.approx(port, rel=1e-7)
    assert lti.C.ravel() == pytest.approx(port, rel=1e-7)
    for t in (0., 1e-7, 1e-6, 1e-5):
        assert expm(lti.A*t) @ p.x0 == pytest.approx(expm(ag*t) @ known, rel=1e-7, abs=1e-12)
    assert .5*known @ mass @ known == pytest.approx(p.diagnostics['full_quadratic_storage'], rel=1e-8)
    outside = CF((-y, x, 0))*(1-(x*x+y*y)/.01**2)
    out = model.project_initial(outside)
    assert out.diagnostics['residual_quadratic_storage'] > 0
    assert out.diagnostics['pythagorean_relative_residual'] < 1e-8
    # Physical DC cylinder field at unit current, with correct lateral trace.
    dc = CF((-y, x, 0))/(2*math.pi*.01**2)
    dc_projection = model.project_initial(dc)
    dc_error = dc-model.reconstruct(dc_projection.x0)
    measured = math.sqrt(Integrate(solver.mu*dc_error*dc_error, mesh, order=16)/
                         Integrate(solver.mu*dc*dc, mesh, order=16))
    assert dc_projection.diagnostics['relative_field_error'] == pytest.approx(measured, rel=1e-7, abs=1e-10)
    assert 0 <= measured <= 1
    # Two modes centered at high frequency need not resolve a DC initial field.
    if s0:
        assert measured > .25
    # Forced voltage step compared to independently integrated FE forms.
    bg = np.linalg.solve(mass, port)
    for t in (1e-7, 1e-6):
        reduced = np.linalg.solve(lti.A, (expm(lti.A*t)-np.eye(2)) @ lti.B).ravel()
        galerkin = np.linalg.solve(ag, (expm(ag*t)-np.eye(2)) @ bg)
        assert reduced == pytest.approx(galerkin, rel=2e-7, abs=1e-10)
    with pytest.raises(ValueError, match='three-component'):
        model.project_initial(CF(1))
