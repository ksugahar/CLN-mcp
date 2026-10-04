"""Common-model 3D A-phi diffusion experiment (Codex self-review).

A notched conductor with axial voltage lifting gives a genuinely 3D field.
Magnetic potential has tangential homogeneous boundary conditions; electric
potential is prescribed on the end faces and natural on other faces. This is
an internal-field benchmark, not a complete open-boundary device model.
The scalar potential is eliminated from the conductivity matrix; the curl
kernel is removed explicitly. All reduced models use this same pencil.
Direct reference uses sparse LU and dense symmetric eigensolves, independently
of the production ICCG. High-frequency errors against this *discrete* reference
do not establish continuum accuracy when the skin depth is unresolved.
"""
import math
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh, cho_factor, cho_solve
from netgen.occ import Box, OCCGeometry, Z
from ngsolve import Mesh, H1, HCurl, GridFunction, BilinearForm, LinearForm, grad, curl, dx, ds, Integrate, BoundaryFromVolumeCF, CoefficientFunction, x, y, z, exp, TaskManager

MU=4e-7*math.pi; SIGMA=1e6; A=.01; HEIGHT=.015; TAU=MU*SIGMA*A*A


def csr(bf):
    return sp.csr_matrix(bf.mat.CSR()).copy()


def build(maxh):
    shape=Box((-A,-A,0),(A,A,HEIGHT))-Box((.002,.002,HEIGHT/2),(.012,.012,HEIGHT+.002))
    shape.faces.name='surface'; shape.faces.Min(Z).name='in'; shape.faces.Max(Z).name='out'
    mesh=Mesh(OCCGeometry(shape).GenerateMesh(maxh=maxh))
    va=HCurl(mesh,order=1,nograds=False,dirichlet='surface|in|out')
    vp=H1(mesh,order=2,dirichlet='in|out')
    freea=np.fromiter(va.FreeDofs(),bool,va.ndof); freep=np.fromiter(vp.FreeDofs(),bool,vp.ndof)
    pa,qa=va.TnT(); pp,qp=vp.TnT()
    kp=BilinearForm(vp); kp+=SIGMA*grad(pp)*grad(qp)*dx; kp.Assemble()
    g=GridFunction(vp); g.Set(HEIGHT,definedon=mesh.Boundaries('in'))
    kp0=csr(kp)[freep][:,freep].tocsc(); lu=spla.splu(kp0)
    lift=g.vec.CreateVector(); lift.data=kp.mat*g.vec
    rhs=-lift.FV().NumPy()[freep].copy()
    phi=lu.solve(rhs)
    assert np.linalg.norm(kp0@phi-rhs)/np.linalg.norm(rhs)<1e-10
    g.vec.FV().NumPy()[freep]=phi
    e0=-grad(g)
    ka=BilinearForm(va); ka+=curl(pa)*curl(qa)/MU*dx; ka.Assemble()
    ma=BilinearForm(va); ma+=SIGMA*pa*qa*dx; ma.Assemble()
    lf=LinearForm(va); lf+=SIGMA*e0*qa*dx; lf.Assemble()
    k=csr(ka)[freea][:,freea].toarray(); m=csr(ma)[freea][:,freea].toarray()
    f=lf.vec.FV().NumPy()[freea].copy()
    mixed=va*vp; (ua,up),(wa,wp)=mixed.TnT()
    cross=BilinearForm(mixed); cross+=SIGMA*ua*grad(wp)*dx; cross.Assemble()
    coupling=csr(cross)[va.ndof:,:va.ndof][freep][:,freea].toarray()
    # A-phi conductivity: eliminate the correction potential with homogeneous end values.
    m-=coupling.T@lu.solve(coupling)
    m=(m+m.T)/2; k=(k+k.T)/2
    lam,q=eigh(k)
    keep=lam>lam.max()*1e-10
    kernel_load=float(np.linalg.norm(q[:,~keep].T@f)/np.linalg.norm(f))
    assert kernel_load<1e-8
    q=q[:,keep]; kr=q.T@k@q; mr=q.T@m@q; fr=q.T@f
    surface=None
    # Virtual magnetic trace ports: boundary functionals, not new physical excitation.
    specifications=[]
    for axis in range(3):
        for sign in [-1,1]:
            for tangent in range(3):
                if tangent==axis: continue
                center=[0.,0.,HEIGHT/2]
                center[axis]=sign*A if axis<2 else (0. if sign<0 else HEIGHT)
                specifications.append((center,tangent))
    ga=GridFunction(va)
    components=[]
    field=BoundaryFromVolumeCF(curl(ga))
    for center,tangent in specifications:
        weight=exp(-((x-center[0])**2+(y-center[1])**2+(z-center[2])**2)/(.007**2))
        direction=[0.,0.,0.];direction[tangent]=1.
        components.append(weight*CoefficientFunction(tuple(direction))*field)
    integrand=CoefficientFunction(tuple(components))
    surface=np.zeros((q.shape[1],len(specifications)))
    for j in range(q.shape[1]):
        ga.vec.FV().NumPy()[:]=0.
        ga.vec.FV().NumPy()[freea]=q[:,j]
        surface[j,:]=Integrate(integrand,mesh,definedon=mesh.Boundaries('surface|in|out'))
    assert np.linalg.norm(surface)>0

    kr=(kr+kr.T)/2; mr=(mr+mr.T)/2
    assert eigh(mr,eigvals_only=True)[0]>0
    l0=float(fr@np.linalg.solve(kr,fr))
    # H(u)=H(s)/H(0), u=s*tau. Z(u)=u*H(u). Exact static slope is 1.
    mr=mr/TAU; fr=fr/math.sqrt(l0)
    poles,v=eigh(kr,mr); residues=(fr@v)**2
    assert np.all(poles>0) and np.all(residues>=0)
    assert abs(np.sum(residues/poles)-1)<1e-8
    return kr,mr,fr,poles,residues,{'maxh':maxh,'ne':mesh.ne,'free_A':int(freea.sum()),
            'physical_dimension':len(poles),'gradient_dimension':int((~keep).sum()),
            'kernel_load_fraction':kernel_load,'static_H':l0},surface
