"""Regressions from reciprocal review of the expansion-point NGSolve solver."""
import math
import pytest
from ngsolve import Integrate
from cln import TOmegaCLN, cylinder_mesh, reference_impedance, ladder_impedance, rl_lti

A, H, SIG, MU = .01, .01, 1e6, 4e-7*math.pi


@pytest.fixture(scope="module")
def mesh():
    return cylinder_mesh(A, H, maxh=A/2, maxh_conductor=A/5)


def test_unconverged_refinement_cannot_return_coefficients(mesh):
    solver = TOmegaCLN(mesh, SIG, MU, A, H, solver="direct", eps_rel=1e3, maxiter=1, order=2)
    with pytest.raises(RuntimeError, match="did not converge"):
        solver.run(nmax=0)
    assert not solver.solve_diagnostics[-1]["converged"]
    assert solver.solve_diagnostics[-1]["relative_residual"] > .1


@pytest.mark.parametrize("bc", ["Neumann", "Dirichlet"])
def test_shifted_ladder_complex_response_and_first_moment(mesh, bc):
    s0 = 2*math.pi*1e5
    solver = TOmegaCLN(mesh, SIG, MU, A, H, s0=s0, bc=bc, order=2, solver="direct")
    r, ell, _ = solver.run(nmax=6)
    assert all(item["converged"] for item in solver.solve_diagnostics)
    ref = lambda s: reference_impedance(mesh, SIG, MU, A, H, s, bc=bc, order=2)
    assert r[0] == pytest.approx(ref(s0), rel=1e-10)
    delta = s0*1e-3
    assert (ref(s0+delta)-ref(s0-delta))/(2*delta) == pytest.approx(ell[1], rel=2e-6, abs=0)
    model = rl_lti(r, ell, s0=s0)
    for f in (1e5, 2e5):
        s = 2j*math.pi*f
        full = ref(s)
        assert model.response(s) == pytest.approx(ladder_impedance(r, ell, s, s0), rel=1e-10)
        assert abs(model.response(s)/full-1) < .01
    # Expansion-point accuracy is local: not an all-band/low-frequency guarantee.
    slow = 2j*math.pi*1e3
    assert abs(model.response(slow)/ref(slow)-1) < .1


@pytest.mark.parametrize("args", [{"s0": -1}, {"s0": float("nan")}, {"maxiter": 0},
                                   {"solver": "typo"}, {"eps_rel": 0}, {"tol": 0}, {"order": True}])
def test_invalid_parameters_fail_before_mesh_access(args):
    with pytest.raises(ValueError):
        TOmegaCLN(None, SIG, MU, A, H, **args)


@pytest.mark.parametrize("bc", ["Neumann", "Dirichlet"])
@pytest.mark.parametrize("s0", [0., 2*math.pi*1e5])
def test_modal_orthogonality_for_both_spaces(mesh, bc, s0):
    solver = TOmegaCLN(mesh, SIG, MU, A, H, s0=s0, bc=bc, order=2, solver="direct")
    even, currents, odd, resistance, inductance = [], [], [], [], []
    t, j = solver.initial()
    hprev = None
    for stage in range(3):
        if stage:
            t, j = solver.step(t, j, hprev, inductance[-1])
        r = 1/(solver.Pj(j)+s0*solver.Wm(t))
        hprev = r*t if hprev is None else hprev+r*t
        even.append(t); currents.append(j); odd.append(hprev)
        resistance.append(r); inductance.append(solver.Wm(hprev))
    for i in range(3):
        for k in range(i):
            rk = Integrate((currents[i]*currents[k]/SIG+s0*MU*even[i]*even[k])*solver._dx, mesh)/H
            lm = Integrate(MU*odd[i]*odd[k]*solver._dx, mesh)/H
            assert abs(rk)*(resistance[i]*resistance[k])**.5 < 1e-7
            assert abs(lm)/(inductance[i]*inductance[k])**.5 < 1e-7
