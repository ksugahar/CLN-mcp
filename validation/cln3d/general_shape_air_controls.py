"""Native geometry, quadrature and stage-A operator controls for the air model."""
import numpy as np
import scipy.sparse.linalg as sla
from general_shape_air import Model,recursion,SIGMA,MU,csr

def stage_a_control(maxh=.006,airh=.008,order=1):
    # Same conformal mesh, different magnetic boundary; no air electric dofs.
    full=Model(maxh,order,airh=airh,geometry="notched")
    interior=Model(maxh,order,airh=airh,geometry="notched",interior_control=True)
    assert full.metadata['mesh_sha256']==interior.metadata['mesh_sha256']
    indices=np.flatnonzero(interior.fa)[np.isin(np.flatnonzero(interior.fa),np.flatnonzero(full.fa))]
    lookup={d:i for i,d in enumerate(np.flatnonzero(full.fa))}
    idx=np.array([lookup[d] for d in indices])
    def relative(a,b):return float(sla.norm(a-b)/max(sla.norm(a),sla.norm(b),1e-300))
    defects={'K_full_clamped_vs_D':relative(full.K[idx][:,idx],interior.K_D),
        'K_air_clamped':float(sla.norm(full.K_air[idx][:,idx])/sla.norm(interior.K_D)),
        'M':relative(full.M[idx][:,idx],interior.M),'S':relative(full.S,interior.S),
        'C':relative(full.C[:,idx],interior.C),
        'f':float(np.linalg.norm(full.f[idx]-interior.f)/np.linalg.norm(interior.f))}
    # Fresh independent assembly of the published stage-A conductor-only K form.
    ng=interior.ng;a,v=interior.va.TnT()
    k=ng.BilinearForm(interior.va);k+=ng.curl(a)*ng.curl(v)/MU*interior.dxD;k.Assemble()
    defects['independent_interior_K']=relative(csr(k)[interior.fa][:,interior.fa],interior.K)
    assert max(defects.values())<1e-10,defects
    runs=[recursion(interior,s,4) for s in [0.,2*np.pi*1e4]]
    airmax=max(abs(v) for r in runs for e in r['energy'] for v in [e['magnetic_region_integrals'][1],e['cumulative_region_integrals'][1]])
    assert airmax==0
    return dict(scope="Same-mesh interface clamp removes the decoupled air block and recovers stage-A conductor-only operators, not a rectangular-box geometric limit",operator_defects=defects,air_energy_max=airmax,
        mesh_sha256=interior.metadata['mesh_sha256'],L1=[r['L'][0] for r in runs],R0=[r['Rhat'][0] for r in runs])
