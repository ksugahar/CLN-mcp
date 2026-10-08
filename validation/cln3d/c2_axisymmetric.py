"""Independent C2 independent axisymmetric A_theta/r and analytic solid-cylinder oracle."""
import math,json,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import ngsolve as ng
from netgen.geom2d import SplineGeometry
import mpmath as mp
SIGMA=1e6;MU=4e-7*math.pi;RADIUS=.025;HEIGHT=.020


def solve(maxh=.0005,oracle=False):
 start=time.time();geo=SplineGeometry()
 geo.AddRectangle((0,0),(RADIUS,HEIGHT),bcs=['bottom','outer','top','axis'],leftdomain=1,rightdomain=0)
 geo.SetMaterial(1,'conductor' if oracle else 'air')
 if not oracle:
  geo.AddRectangle((.008,.0085),(.014,.0115),leftdomain=2,rightdomain=1);geo.SetMaterial(2,'conductor')
  geo.AddRectangle((.018,.005),(.021,.007),leftdomain=3,rightdomain=1);geo.SetMaterial(3,'coil')
 mesh=ng.Mesh(geo.GenerateMesh(maxh=maxh));space=ng.H1(mesh,order=2,dirichlet='bottom|outer|top');free=np.array(list(space.FreeDofs()),bool)
 r=ng.x;u,v=space.TnT();dx=ng.dx(bonus_intorder=12);D=ng.dx(definedon=mesh.Materials('conductor'),bonus_intorder=12)
 k=ng.BilinearForm(space);k+=2*math.pi*r/MU*((r*ng.grad(u)[0]+2*u)*(r*ng.grad(v)[0]+2*v)+r*r*ng.grad(u)[1]*ng.grad(v)[1])*dx;k.Assemble()
 m=ng.BilinearForm(space);m+=2*math.pi*SIGMA*r**3*u*v*D;m.Assemble()
 if oracle:
  j0=math.pi/(RADIUS**2*HEIGHT);current=j0*r*ng.sin(math.pi*ng.y/HEIGHT);source_dx=dx
 else:
  profile=.25*(1-ng.cos(2*math.pi*(r-.018)/.003))*(1-ng.cos(2*math.pi*(ng.y-.005)/.002))
  source_dx=ng.dx(definedon=mesh.Materials('coil'),bonus_intorder=20);norm=ng.Integrate(profile*source_dx,mesh);current=profile/norm
 f=ng.LinearForm(space);f+=2*math.pi*r*r*current*space.TestFunction()*source_dx;f.Assemble()
 K=sp.csr_matrix(k.mat.CSR(),shape=(k.mat.height,k.mat.width)).copy()[free][:,free].tocsc();M=sp.csr_matrix(m.mat.CSR(),shape=(m.mat.height,m.mat.width)).copy()[free][:,free].tocsc();b=f.vec.FV().NumPy().copy()[free]
 rows=[]
 for hz in [1e3,1e4,1e5,1e6]:
  s=2j*math.pi*hz;a=sla.spsolve(K+s*M,b);z=s*(b@a);row=dict(hz=hz,Z=[z.real,z.imag])
  if oracle:
   mp.mp.dps=50;kappa=mp.sqrt((mp.pi/HEIGHT)**2+MU*SIGMA*s);kr=kappa*RADIUS
   phi=mp.pi*MU*j0**2*HEIGHT/kappa**2*(RADIUS**4/4-RADIUS**3*mp.besseli(2,kr)/(kappa*mp.besseli(1,kr)))
   exact=complex(s*phi);row['analytic_Z']=[exact.real,exact.imag];row['relative_error']=abs(z/exact-1)
   assert row['relative_error']<.002,row
  rows.append(row)
 return dict(oracle=oracle,maxh=maxh,order=2,triangles=mesh.ne,dofs=len(b),seconds=time.time()-start,self_inductance=float(b@sla.spsolve(K,b)),frequency=rows)
