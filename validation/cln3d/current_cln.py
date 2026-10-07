"""Energy-normalized Type2 recursion of the physical current R/L pencil."""
import math
import numpy as np
import scipy.linalg as la
from general_shape_air import TAU,FREQUENCIES,ladder,complex_pair
from impedance_taylor import reduced


def physical_recursion(R,L,b,s0,modes=4):
 matrix=R+s0*L;scale=1/np.sqrt(matrix.diagonal());factor=la.cho_factor(scale[:,None]*matrix*scale[None,:])
 def solve(rhs):
  out=scale*la.cho_solve(factor,scale*rhs)
  for _ in range(3):out+=scale*la.cho_solve(factor,scale*(rhs-matrix@out))
  return out
 j=solve(b);cumulative=np.zeros_like(j);vectors=[];cumulatives=[];rh=[];ells=[]
 for stage in range(modes):
  value=1/(j@matrix@j);cumulative+=value*j;ell=cumulative@L@cumulative
  assert value>0 and ell>0
  rh.append(float(value));ells.append(float(ell));vectors.append(j.copy());cumulatives.append(cumulative.copy())
  if stage+1<modes:j-=solve(L@cumulative)/ell
 v=np.column_stack(vectors);x=np.column_stack(cumulatives);rp=v.T@R@v;lp=v.T@L@v;bp=v.T@b
 def orth(a):return float(np.max(abs(a-np.diag(a.diagonal()))/np.sqrt(np.outer(a.diagonal(),a.diagonal()))))
 electric=orth(rp+s0*lp);magnetic=orth(x.T@L@x)
 assert max(electric,magnetic)<1e-8,(electric,magnetic)
 exact=reduced(R,L,b,s0,2*modes+1,TAU);projected=reduced(rp,lp,bp,s0,2*modes+1,TAU)
 difference=abs(exact[:2*modes]-projected[:2*modes])/np.maximum(abs(exact[:2*modes]),1e-14*np.max(abs(exact[:2*modes])))
 assert max(difference)<1e-8,difference
 frequency=[]
 for hz in FREQUENCIES:
  s=2j*math.pi*hz;state=la.solve(R+s*L,b,assume_a='sym');current=b@state;z=1/current
  zlad=ladder(rh,ells,s,s0);zgal=1/(bp@la.solve(rp+s*lp,bp,assume_a='sym'))
  assert abs(zlad/zgal-1)<1e-8
  loss=float(np.real(state.conj()@R@state));mag=float(np.real(state.conj()@L@state))
  power=abs((loss+s*mag)/np.conj(current)-1);assert power<1e-8
  frequency.append(dict(hz=hz,Z=complex_pair(z),ladder=complex_pair(zlad),reduction_error=float(abs(zlad/z-1)),
    ladder_galerkin_error=float(abs(zlad/zgal-1)),power_defect=float(power),loss=loss,magnetic_energy=mag))
 return dict(s0=s0,modes=modes,Rhat=rh,L=ells,R_phys=rp.tolist(),L_phys=lp.tolist(),port=bp.tolist(),
  electric_orth=electric,magnetic_orth=magnetic,current_modes=[v.tolist() for v in vectors],cumulative_modes=[v.tolist() for v in cumulatives],
  frequency=frequency,full_Z_coefficients=exact.tolist(),galerkin_Z_coefficients=projected.tolist(),contact_per_coefficient=difference.tolist())
