"""Physical CLN on a nested magnetic trace subspace; full conductor currents."""
import math
from types import SimpleNamespace
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from general_shape_air import TAU,FREQUENCIES,ladder,complex_pair
from impedance_taylor import mixed,reduced


class TraceModel:
    def __init__(self,base,basis):
        self.base,self.Q=base,basis
        self.K=basis.T@base.K@basis
        self.M=basis.T@base.M@basis
        self.C=np.asarray(base.C@basis)
        self.f=basis.T@base.f
        self.chol=la.cho_factor(self.K)
        self.lus={}
        self.residuals=[]

    def magnetic(self,e):
        load=self.base.load(e);return self.Q@la.cho_solve(self.chol,self.Q.T@load)

    def response(self,e,s):
        b=self.base;_,v,p=b.split(e)
        load=self.Q.T@b.load(e);phi=b.C@v-b.S@p
        if s==0:
            out=e.copy();out[1+b.na:]+=la.solve(b.S.toarray(),phi,assume_a='pos')
            return out,self.magnetic(out)
        if s not in self.lus:
            mat=np.block([[self.K+s*self.M,self.C.T],[self.C,b.S.toarray()/s]])
            scale=1/np.sqrt(abs(mat.diagonal()))
            self.lus[s]=(la.lu_factor(scale[:,None]*mat*scale[None,:]),scale,mat)
        lu,scale,mat=self.lus[s]
        rhs=np.r_[load,phi/s];x=scale*la.lu_solve(lu,scale*rhs)
        for _ in range(3):x+=scale*la.lu_solve(lu,scale*(rhs-mat@x))
        self.residuals.append(float(la.norm(mat@x-rhs)/max(la.norm(rhs),1e-300)))
        a=self.Q@x[:self.Q.shape[1]]
        out=e.astype(x.dtype).copy();out[1:1+b.na]-=s*a;out[1+b.na:]+=x[self.Q.shape[1]:]
        return out,a

    def taylor_proxy(self):
        return SimpleNamespace(na=len(self.f),np=self.base.np,Kg=sp.csc_matrix(self.K),M=sp.csc_matrix(self.M),
           C=sp.csc_matrix(self.C),S=self.base.S,f=self.f,G0=self.base.G0)


def run(model,s0,modes=4):
    b=model.base;initial=np.zeros(1+b.na+b.np);initial[0]=1
    e,a=model.response(initial,s0);x=np.zeros_like(e);ax=np.zeros_like(a)
    es=[];aa=[];xx=[];R=[];L=[];parts=[]
    for i in range(modes):
        loss=b.r(e,e);mag=a@(b.K@a);rh=1/(loss+s0*mag)
        x+=rh*e;ax+=rh*a;ell=ax@(b.K@ax)
        assert rh>0 and ell>0
        R.append(float(rh));L.append(float(ell));es.append(e.copy());aa.append(a.copy());xx.append(ax.copy())
        parts.append([float(ax@(k@ax)) for k in [b.K_D,b.K_air]])
        if i+1<modes:
            u=np.zeros_like(e);u[1:1+b.na]=ax
            correction,ca=model.response(u,s0);e-=correction/ell;a-=ca/ell
    rp=np.array([[b.r(u,v) for v in es] for u in es]);lp=np.array([[u@(b.K@v) for v in aa] for u in aa]);bp=np.array([b.port(u) for u in es])
    def orth(v):return float(np.max(abs(v-np.diag(np.diag(v)))/np.sqrt(np.outer(v.diagonal(),v.diagonal()))))
    electric=orth(rp+s0*lp);magnetic=orth(np.array([[u@(b.K@v) for v in xx] for u in xx]))
    assert max(electric,magnetic)<1e-8,(electric,magnetic)
    full=mixed(model.taylor_proxy(),s0,2*modes+1,TAU);projected=reduced(rp,lp,bp,s0,2*modes+1,TAU)
    coefficient=np.abs(full[:2*modes]-projected[:2*modes])/np.maximum(abs(full[:2*modes]),1e-300)
    assert max(coefficient)<1e-8,coefficient
    rows=[]
    for hz in FREQUENCIES:
        s=2j*math.pi*hz;ef,af=model.response(initial,s)
        z=1/b.port(ef);zlad=ladder(R,L,s,s0);zgal=1/(bp@la.solve(rp+s*lp,bp))
        assert abs(zlad/zgal-1)<1e-8
        loss=float(np.real(b.r(ef.conj(),ef)));regions=[float(np.real(af.conj()@(k@af))) for k in [b.K_D,b.K_air]]
        power=abs((loss+s*sum(regions))/np.conj(b.port(ef))-1)
        assert power<1e-8
        rows.append(dict(hz=hz,Z=complex_pair(z),ladder=complex_pair(zlad),reduction_error=float(abs(zlad/z-1)),
          ladder_galerkin_error=float(abs(zlad/zgal-1)),power_defect=float(power),magnetic_regions=regions,loss=loss))
    return dict(s0=s0,Rhat=R,L=L,R_phys=rp.tolist(),L_phys=lp.tolist(),port=bp.tolist(),
      electric_orth=electric,magnetic_orth=magnetic,energy_parts=parts,frequency=rows,
      full_Z_coefficients=full.tolist(),galerkin_Z_coefficients=projected.tolist(),contact_per_coefficient=coefficient.tolist(),
      electric_modes=[v.tolist() for v in es],magnetic_potentials=[v.tolist() for v in aa])
