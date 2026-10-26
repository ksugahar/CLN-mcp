import math
import pytest
from cln import (TOmegaCLN, cylinder_mesh, ladder_impedance, reference_impedance,
                 reference_sweep, wire_internal_impedance)

A, H, SIG, MU = 0.01, 0.01, 1e6, 4e-7*math.pi


@pytest.fixture(scope="module")
def mesh():
    return cylinder_mesh(A, H, maxh=A/2, maxh_conductor=A/10)


@pytest.mark.parametrize("bc", ["Neumann", "Dirichlet"])
@pytest.mark.parametrize("jomega", [False, True])
def test_reference_matches_bessel(mesh, bc, jomega):
    for f in (1e3, 1e5):
        s = (2j if jomega else 2)*math.pi*f
        z = reference_impedance(mesh, SIG, MU, A, H, s, bc=bc, order=2)
        ze = complex(wire_internal_impedance(s, A, SIG, MU))
        # The low-frequency check detects biased order-2 Dirichlet lifting;
        # at high frequency this mesh has a separate skin-depth error.
        tolerance = 1e-4 if f == 1e3 else 3e-3
        assert abs(z - ze) <= tolerance*abs(ze), (bc, jomega, f, z, ze)


def test_sweep_shape_and_type(mesh):
    z = reference_sweep(mesh, SIG, MU, A, H, [1e3, 1e4], order=2)
    assert z.shape == (2,) and z.dtype.kind == "c"


def test_ladder_converges_to_reference_near_s0(mesh):
    """Same mesh/order: the expansion-point ladder reproduces the direct solve at s0."""
    s0 = 2*math.pi*1e5
    R, L, _ = TOmegaCLN(mesh, SIG, MU, A, H, s0=s0, bc="Neumann", order=2, solver="direct").run(nmax=4)
    for f in (1e5, 1.5e5):
        s = 2*math.pi*f
        zr = reference_impedance(mesh, SIG, MU, A, H, s, bc="Neumann", order=2)
        assert ladder_impedance(R, L, s, s0) == pytest.approx(zr, rel=2e-4), f
