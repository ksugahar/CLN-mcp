import math
import pytest
from netgen.occ import Circle, OCCGeometry
from ngsolve import Mesh, CF
from cln.diffusion import DiffusionCLN, rc_ladder_impedance
from scipy.special import iv

A, K, RHOC = 0.05, 500.0, 8960.0*385.0      # disk of Tanimoto et al. (SA-25-014)


def r_exact(n):
    return 1/(8*(n + 1)**5*math.pi*K)


def c_exact(n):
    return (n + 1)**2*(n + 2)**2*math.pi*A**2*RHOC/(2*n + 3)


def z_exact(s):
    g = math.sqrt(s*RHOC/K)
    qs = 1/(math.pi*A**2)
    return qs**2/(RHOC*s)*(math.pi*A**2 - 2*math.pi*A*iv(1, g*A)/(g*iv(0, g*A)))


@pytest.fixture(scope="module")
def stages():
    geo = Circle((0, 0), A).Face()
    geo.edges.name = "outer"
    mesh = Mesh(OCCGeometry(geo, dim=2).GenerateMesh(maxh=A/12)).Curve(4)
    return DiffusionCLN(mesh, K, RHOC, CF(1.0), dirichlet="outer", order=6).run(nstages=4)


def test_elements_match_closed_form(stages):
    for n in range(4):
        assert stages["R"][n] == pytest.approx(r_exact(n), rel=1e-6), ("R", n)
        assert stages["C"][n] == pytest.approx(c_exact(n), rel=1e-6), ("C", n)


def test_ladder_reproduces_exact_impedance(stages):
    tau = RHOC*A**2/K
    for x, tol in ((0.1, 1e-8), (1.0, 1e-6), (10.0, 1e-4)):
        s = x/tau
        assert rc_ladder_impedance(stages["R"], stages["C"], s) == pytest.approx(z_exact(s), rel=tol), x
