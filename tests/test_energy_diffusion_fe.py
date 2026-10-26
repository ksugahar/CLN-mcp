import numpy as np
import pytest
from netgen.geom2d import unit_square
from ngsolve import (Mesh, H1, BilinearForm, LinearForm, GridFunction, Integrate,
                     grad, dx, x, y)
from cln.diffusion import DiffusionCLN


@pytest.fixture(scope="module")
def example():
    mesh = Mesh(unit_square.GenerateMesh(maxh=.2))
    solver = DiffusionCLN(mesh, 1+x, 2+y, 1+x, order=4, length=2.5)
    return solver, solver.energy_model(3)


def test_variable_materials_certify_and_reconstruct_energy(example):
    solver, model = example
    v = np.array([.3, -.02, .004])
    u = model.reconstruct(v)
    q = model.reconstruct_flux(v)
    d = solver.length
    # Independent higher-order integration, not the certificate's integration helper.
    W = .5*d*Integrate((2+y)*u*u, solver.mesh, order=20)
    D = d*Integrate((1+x)*grad(u)*grad(u), solver.mesh, order=20)
    U = d*Integrate((2+y)*u, solver.mesh, order=20)
    residual = q + (1+x)*grad(u)
    assert Integrate(residual*residual, solver.mesh, order=20) < 1e-25
    b = model.balance(v, .7)
    assert b["quadratic_storage"] == pytest.approx(W, rel=1e-10)
    assert b["dissipation"] == pytest.approx(D, rel=1e-10)
    assert b["linear_storage_content"] == pytest.approx(U, rel=1e-10)
    assert max(model.certificate["residuals"].values()) < 1e-10


@pytest.mark.parametrize("s", [1., 1j, 4j])
def test_variable_material_reduction_matches_independent_full_fe(example, s):
    solver, model = example
    mesh, d = solver.mesh, solver.length
    fes = H1(mesh, order=4, dirichlet=".*", complex=True)
    u, v = fes.TnT()
    measure = dx(bonus_intorder=14)
    a = BilinearForm(fes, symmetric=True)
    a += ((1+x)*grad(u)*grad(v) + s*(2+y)*u*v)*measure
    rhs = LinearForm(fes)
    rhs += solver.f*v/d*measure
    a.Assemble(); rhs.Assemble()
    full = GridFunction(fes)
    full.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky")*rhs.vec
    exact = Integrate(solver.f*full, mesh, order=20)
    assert model.impedance(s) == pytest.approx(exact, rel=2e-6)


def test_pure_neumann_is_not_silently_regularized():
    mesh = Mesh(unit_square.GenerateMesh(maxh=.5))
    with pytest.raises(ValueError, match="Neumann"):
        DiffusionCLN(mesh, 1, 1, 1, dirichlet="")
