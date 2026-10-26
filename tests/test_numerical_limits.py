import numpy as np
import pytest
from ngsolve import CF
from cln.analytic import wire_internal_impedance
from cln.diffusion import rc_ladder_impedance

A, SIG, MU = .01, 1e6, 4e-7*np.pi

def test_wire_dc_and_small_s():
    rdc=1/(np.pi*A*A*SIG)
    s=np.array([0,1e-10,1e-6,1j*1e-6])
    got=wire_internal_impedance(s,A,SIG,MU)
    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(got,rdc+MU/(8*np.pi)*s,rtol=1e-12)
    assert wire_internal_impedance(0,A,SIG,MU)==pytest.approx(rdc)

def test_wire_high_frequency_skin_limit():
    s=2j*np.pi*1e12
    got=wire_internal_impedance(s,A,SIG,MU)
    skin=np.sqrt(s*MU/SIG)/(2*np.pi*A)
    assert np.isfinite(got)
    assert got==pytest.approx(skin,rel=1e-4)

def test_rc_dc_and_complex_circuit():
    assert rc_ladder_impedance([2.,3.],[4.],0.)==2.
    s=np.array([0,1e-8,1j,10+2j])
    # Two-node admittance elimination of the series capacitor.
    expected=2*(1+12*s)/(1+20*s)
    np.testing.assert_allclose(rc_ladder_impedance([2.,3.],[4.],s),expected)

@pytest.mark.parametrize('n',[0,-1,3,1.5,True])
def test_rc_reject_bad_counts(n):
    with pytest.raises(ValueError):
        rc_ladder_impedance([2.,3.],[4.],1.,nstages=n)

@pytest.mark.parametrize('R,C,n', [([], [], None), ([2., 3.], [], None)])
def test_rc_rejects_incomplete_ladders(R, C, n):
    with pytest.raises(ValueError):
        rc_ladder_impedance(R, C, 1., nstages=n)

@pytest.fixture(scope='module')
def diffusion_mesh():
    from netgen.geom2d import unit_square
    from ngsolve import Mesh
    return Mesh(unit_square.GenerateMesh(maxh=.3))

@pytest.mark.parametrize('length',[0.,-1.,float('nan'),float('inf')])
def test_diffusion_rejects_invalid_depth(diffusion_mesh,length):
    from cln.diffusion import DiffusionCLN
    with pytest.raises(ValueError,match='length'):
        DiffusionCLN(diffusion_mesh,1.,1.,CF(1.),length=length)

def test_diffusion_rejects_zero_source(diffusion_mesh):
    from cln.diffusion import DiffusionCLN
    with pytest.raises(ValueError,match='source'):
        DiffusionCLN(diffusion_mesh,1.,1.,CF(0.))

@pytest.mark.parametrize('n',[0,-1,1.5,True])
def test_diffusion_rejects_invalid_stage_count(diffusion_mesh,n):
    from cln.diffusion import DiffusionCLN
    solver=DiffusionCLN(diffusion_mesh,1.,1.,CF(1.))
    with pytest.raises(ValueError,match='nstages'):
        solver.run(nstages=n)

def test_diffusion_extrusion_scaling(diffusion_mesh):
    from cln.diffusion import DiffusionCLN
    unit=DiffusionCLN(diffusion_mesh,1.,1.,CF(1.)).run(nstages=2)
    deep=DiffusionCLN(diffusion_mesh,1.,1.,CF(1.),length=2.5).run(nstages=2)
    np.testing.assert_allclose(deep['R'],np.array(unit['R'])/2.5)
    np.testing.assert_allclose(deep['C'],np.array(unit['C'])*2.5)
    for s in [0.,1.,1j]:
        assert rc_ladder_impedance(deep['R'],deep['C'],s)==pytest.approx(
            rc_ladder_impedance(unit['R'],unit['C'],s)/2.5)

def test_diffusion_rejects_zero_capacity(diffusion_mesh):
    from cln.diffusion import DiffusionCLN
    solver = DiffusionCLN(diffusion_mesh, 1., 0., CF(1.))
    with pytest.raises(ValueError, match='capacity'):
        solver.run(nstages=1)
