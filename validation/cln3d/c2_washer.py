"""Coil-driven washer: physical A-phi Type1 field recursion.

The caller owns TaskManager. Only the conductor has conductivity; the PEC
enclosure fixes tangential A. This is a same-mesh validation example, not a
claim of continuum accuracy at skin-depth-limited frequencies.
"""
import scipy.linalg as la
import sys,math,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import ngsolve as ng
from netgen.occ import Cylinder,Z,Glue,OCCGeometry
from surface_hybrid.ladder_eval import z_type1
SIGMA=1e6;MU=4e-7*math.pi


def csr(a):return sp.csr_matrix(a.mat.CSR(),shape=(a.mat.height,a.mat.width)).copy()


def run(h=.006,return_state=False,coil_h=None,a_order=0, filled=False):
 start=time.time();washer=Cylinder((0,0,.0085),Z,r=.014,h=.003)
 if not filled:washer-=Cylinder((0,0,.0085),Z,r=.008,h=.003)
 washer.mat('conductor');washer.faces.name='surface';washer.solids.maxh=h;washer.faces.maxh=h;washer.edges.maxh=h
 coil=Cylinder((0,0,.005),Z,r=.021,h=.002)-Cylinder((0,0,.005),Z,r=.018,h=.002)
 coil.mat('coil');coil.faces.name='coil_interface';coil.solids.maxh=coil_h or h;coil.faces.maxh=coil_h or h;coil.edges.maxh=coil_h or h
 enclosure=Cylinder((0,0,0),Z,r=.025,h=.020);enclosure.faces.name='outer'
 air=enclosure-washer-coil;air.mat('air');air.solids.maxh=h;air.faces.maxh=h
 mesh=ng.Mesh(OCCGeometry(Glue([washer,coil,air])).GenerateMesh(maxh=h,curvaturesafety=2));mesh.Curve(3)
 dx=ng.dx(bonus_intorder=16);D=ng.dx(definedon=mesh.Materials('conductor'),bonus_intorder=16);coil_dx=ng.dx(definedon=mesh.Materials('coil'),bonus_intorder=20)
 A=ng.HCurl(mesh,order=a_order,nograds=False,dirichlet='outer');phi=ng.H1(mesh,order=a_order+1,definedon=mesh.Materials('conductor'))
 fa=np.array(list(A.FreeDofs()),bool);fp=np.array(list(phi.FreeDofs()),bool);fp[np.flatnonzero(fp)[0]]=False
 a,v=A.TnT();kf=ng.BilinearForm(A);kf+=ng.curl(a)*ng.curl(v)/MU*dx;kf.Assemble();K=csr(kf)[fa][:,fa].tocsc()
 mf=ng.BilinearForm(A);mf+=SIGMA*a*v*D;mf.Assemble();M=csr(mf)[fa][:,fa].tocsc()
 p,q=phi.TnT();sf=ng.BilinearForm(phi);sf+=SIGMA*ng.grad(p)*ng.grad(q)*D;sf.Assemble();S=csr(sf)[fp][:,fp].tocsc()
 pair=A*phi;(a,p),(v,q)=pair.TnT();cf=ng.BilinearForm(pair);cf+=SIGMA*a*ng.grad(q)*D;cf.Assemble();C=csr(cf)[A.ndof:,:A.ndof][fp][:,fa].tocsc()
 gm,scalar=A.CreateGradient();fh=np.array(list(scalar.FreeDofs()),bool);G=sp.csr_matrix(gm.CSR(),shape=(gm.height,gm.width)).copy()[fa][:,fh].tocsc()
 gg=G@G.T;scale=max(abs(K.diagonal()))/max(gg.diagonal());Kg=K+scale*gg
 radius=ng.sqrt(ng.x**2+ng.y**2);wr=.5*(1-ng.cos(2*math.pi*(radius-.018)/.003));wz=.5*(1-ng.cos(2*math.pi*(ng.z-.005)/.002))
 profile=wr*wz;normalization=ng.Integrate(profile/(2*math.pi*radius)*coil_dx,mesh)
 impressed=profile/normalization*ng.CoefficientFunction((-ng.y/radius,ng.x/radius,0))
 load=ng.LinearForm(A);load+=impressed*A.TestFunction()*coil_dx;load.Assemble();f=load.vec.FV().NumPy().copy()[fa]
 gram=(G.T@G).tocsc();projection=G@sla.spsolve(gram,G.T@f);fraction=np.linalg.norm(projection)/np.linalg.norm(f)
 print('mesh',mesh.ne,len(f),'source gradient fraction',fraction,flush=True)
 assert fraction<1e-8,fraction
 f-=projection
 active=np.flatnonzero(np.asarray(abs(M).sum(axis=0)).ravel()>0);sub=C[:,active].toarray()
 condensed=M[active][:,active].toarray()-sub.T@sla.splu(S).solve(sub)
 rows,columns=np.meshgrid(active,active,indexing='ij')
 m=sp.coo_matrix((condensed.ravel(),(rows.ravel(),columns.ravel())),shape=M.shape).tocsc();m.eliminate_zeros();assert sla.norm(m@G)/(sla.norm(m)*sla.norm(G))<1e-10
 frequencies=[1e3,1e4,1e5,1e6];runs=[]
 for s0 in [0.,2*math.pi*1e4]:
  km=Kg+s0*m;factor=sla.splu(km.tocsc())
  def solve(rhs):
   result=factor.solve(rhs)
   for _ in range(3):result+=factor.solve(rhs-km@result)
   return result
  a=solve(f);electric=np.zeros(len(f));magnetic=[];els=[];elements=[]
  for _ in range(4):
   ell=a@((K+s0*m)@a);electric+=a/ell;r=1/(electric@(m@electric))
   assert min(ell,r)>0;elements.extend([float(ell),float(r)]);magnetic.append(a.copy());els.append(electric.copy())
   a-=r*solve(m@electric)
  Q=np.column_stack(magnetic);kp=Q.T@K@Q;mp=Q.T@m@Q;bp=Q.T@f
  def orth(a):return float(np.max(abs(a-np.diag(a.diagonal()))/np.sqrt(np.outer(a.diagonal(),a.diagonal()))))
  defect=max(orth(kp+s0*mp),orth(np.array(els)@m@np.array(els).T));assert defect<1e-8,defect
  # Taylor coefficients of flux Z(s)/s about s0: full physical pencil versus ladder Galerkin pencil.
  full_v=solve(f); reduced_factor=la.cho_factor(kp+s0*mp); small_v=la.cho_solve(reduced_factor,bp)
  tq=[]; rq=[]
  for n in range(9):
   tq.append(float(f@full_v));rq.append(float(bp@small_v))
   full_v=-solve(m@full_v);small_v=-la.cho_solve(reduced_factor,mp@small_v)
  relative=[abs(x-y)/max(abs(x),1e-300) for x,y in zip(tq,rq)]
  assert max(relative[:8])<1e-7,relative
  rows=[]
  for hz in frequencies:
   s=2j*math.pi*hz;af=sla.spsolve((Kg+s*m).tocsc(),f);z=s*(f@af);zg=s*(bp@np.linalg.solve(kp+s*mp,bp));zl=z_type1(elements,s0,s)
   assert abs(zl/zg-1)<1e-8
   residual=np.linalg.norm((K+s*m)@af-f)/np.linalg.norm(f);assert residual<1e-8
   power=float(np.real(np.vdot(-s*af,m@(-s*af))));assert abs(power/z.real-1)<1e-8
   rows.append(dict(hz=hz,Z=[z.real,z.imag],ladder=[zl.real,zl.imag],reduction_error=abs(zl/z-1),original_residual=residual,joule=power,eddy_increment=[(z-s*(f@sla.spsolve(Kg,f))).real,(z-s*(f@sla.spsolve(Kg,f))).imag]))
  runs.append(dict(s0=s0,elements=elements,orthogonality=defect,frequency=rows,flux_taylor_full=tq,flux_taylor_reduced=rq,flux_taylor_relative=relative,magnetic_modes=Q.tolist(),electric_modes=np.array(els).T.tolist(),K_reduced=kp.tolist(),M_reduced=mp.tolist(),f_reduced=bp.tolist()))
 result=dict(complete=True,ne=mesh.ne,dofs=len(f),source_projection_fraction=fraction,seconds=time.time()-start,h=h,coil_h=coil_h or h,a_order=a_order,filled=filled,self_inductance=float(f@sla.spsolve(Kg,f)),runs=runs)
 if return_state:return result,locals()
 return result
