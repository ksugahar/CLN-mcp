"""Curl-constrained T--Omega current pencil with natural enclosure boundary.

Prototype construction: exterior H is closed, retaining its absolute H1
period; global scalar gradients are eliminated by magnetic-energy minimization.
No tangential H condition is imposed on the perfect enclosure.
"""
import numpy as np
import scipy.linalg as la
import ngsolve as ng
from general_shape_air import csr,SIGMA,MU


def orthogonal_range(a,tolerance=1e-11):
 u,s,_=la.svd(a,full_matrices=False)
 return u[:,s>tolerance*s[0]]


class TOmega:
 def __init__(self,base,order=1):
  self.base=base
  self.space=ng.HCurl(base.mesh,order=order,nograds=False)
  u,v=self.space.TnT()
  resistance=ng.BilinearForm(self.space);resistance+=ng.curl(u)*ng.curl(v)/SIGMA*base.dxD;resistance.Assemble()
  aircurl=ng.BilinearForm(self.space);aircurl+=ng.curl(u)*ng.curl(v)*base.dxAir;aircurl.Assemble()
  magnetic=ng.BilinearForm(self.space);magnetic+=MU*u*v*base.dx;magnetic.Assemble()
  load=ng.LinearForm(self.space);load+=base.e0*ng.curl(v)*base.dxD;load.Assemble()
  R=csr(resistance).toarray();A=csr(aircurl).toarray();L=csr(magnetic).toarray();b=load.vec.FV().NumPy().copy()
  ev,vec=la.eigh(A);scale=max(ev[-1],1.)
  assert ev[0]>-1e-10*scale
  closed=vec[:,ev<1e-10*scale]
  gradient,scalar=self.space.CreateGradient()
  from scipy.sparse import csr_matrix
  self.raw_gradient=csr_matrix(gradient.CSR(),shape=(gradient.height,gradient.width)).copy()
  G=orthogonal_range(self.raw_gradient.toarray())
  gram=G.T@L@G
  closed-=G@la.solve(gram,G.T@L@closed,assume_a='pos')
  self.gradient=G
  self.gradient_gram=gram
  self.Q=orthogonal_range(closed)
  self.R=self.Q.T@R@self.Q;self.L=self.Q.T@L@self.Q;self.b=self.Q.T@b
  self.raw_R,self.raw_L,self.raw_aircurl,self.raw_b=R,L,A,b
  self.closure=float(la.norm(A@self.Q)/(la.norm(A)*la.norm(self.Q)))
  self.scalar_stationarity=float(la.norm(G.T@L@self.Q)/(la.norm(G)*la.norm(L)*la.norm(self.Q)))
  assert self.closure<1e-10 and self.scalar_stationarity<1e-10
  assert la.eigvalsh(self.R)[0]>0 and la.eigvalsh(self.L)[0]>0

 def impedance(self,s):
  j=la.solve(self.R+s*self.L,self.b,assume_a='sym')
  return 1/(self.b@j)

 def field(self,j):
  gf=ng.GridFunction(self.space);gf.vec.FV().NumPy()[:]=np.real(self.Q@j);return gf

 def topology(self):
  from air_subcomplex import transfer_generator
  first=transfer_generator(self.base,self.space)
  second=transfer_generator(self.base,self.space,alternate=True)
  lifts=[];records=[]
  for data in [first,second]:
   h,p=data['h'],data['period']
   orientation=1 if (p@self.Q)@self.b>0 else -1
   h=orientation*h;p=orientation*p
   assert abs(p@h-1)<1e-10
   period_error=la.norm(p@self.Q-self.b)/la.norm(self.b)
   assert period_error<1e-8,period_error
   natural=h-self.gradient@la.solve(self.gradient_gram,self.gradient.T@self.raw_L@h,assume_a='pos')
   assert la.norm(natural-self.Q@(self.Q.T@natural))<1e-8*la.norm(natural)
   # An exact-gradient representative change must cancel through Omega.
   eta=np.sin(np.arange(self.raw_gradient.shape[1])*.37)
   delta=self.raw_gradient@eta
   delta*=la.norm(h)/max(la.norm(delta),1e-300)
   changed=h+delta
   changed-=self.gradient@la.solve(self.gradient_gram,self.gradient.T@self.raw_L@changed,assume_a='pos')
   representative_error=float(la.norm(changed-natural)/la.norm(natural))
   assert representative_error<1e-8,representative_error
   lift=self.Q.T@natural;lifts.append(lift)
   records.append(dict(b1=data['b1'],air_chain_closure=data['closure'],period_port_relative=float(period_error),
    gradient_representative_relative=representative_error,period=p.tolist(),generator=h.tolist(),module_sha256=data['module_sha256'],air_vertices=data['vertices'],
    air_edges=data['edges'],cochain=data['cochains'].tolist(),cycle=data['cycle'].tolist()))
  zero=la.null_space(self.b[None,:])
  transforms=[np.column_stack([lift,zero]) for lift in lifts]
  return records,transforms
