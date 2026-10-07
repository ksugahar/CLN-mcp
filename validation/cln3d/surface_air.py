"""Physical air Schur complement and nested Steklov trace spaces.

No air conductivity or epsilon inverse. Dense quotient operations deliberately
limit this verification implementation to small matched meshes.
"""
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla


def relative(a, b):
    return float(la.norm(a-b)/max(la.norm(a),la.norm(b),1e-300))


def range_basis(a, tolerance=1e-11):
    u,s,_=la.svd(a,full_matrices=False)
    return u[:,s>tolerance*s[0]] if len(s) and s[0]>0 else np.empty((a.shape[0],0))


class AirTrace:
    def __init__(self, base, surface_mass):
        self.base=base
        ng=base.ng
        inside=np.asarray(list(base.va.GetDofs(base.mesh.Materials('conductor'))),bool)[base.fa]
        outside=np.asarray(list(base.va.GetDofs(base.mesh.Materials('air'))),bool)[base.fa]
        self.d=np.flatnonzero(inside & ~outside)
        self.gamma=np.flatnonzero(inside & outside)
        self.air=np.flatnonzero(~inside & outside)
        assert len(self.d)+len(self.gamma)+len(self.air)==base.na
        ka=base.K_air;k=base.K;g=base.G
        aa=ka[self.air][:,self.air].toarray();ag=ka[self.air][:,self.gamma].toarray()
        ev,u=la.eigh(aa);threshold=1e-11*max(ev[-1],1.)
        assert ev[0]>-threshold
        positive=ev>threshold;self.interior_null=u[:,~positive]
        self.extension=-(u[:,positive]/ev[positive])@(u[:,positive].T@ag)
        self.range_defect=float(la.norm(self.interior_null.T@ag)/max(la.norm(ag),1e-300))
        assert self.range_defect<1e-10
        self.S=ka[self.gamma][:,self.gamma].toarray()+ag.T@self.extension
        self.S=(self.S+self.S.T)/2
        self.W=surface_mass[np.ix_(self.gamma,self.gamma)]
        # Explicit trace nullspace is retained, not assigned an epsilon energy.
        ev,u=la.eigh(self.S);threshold=1e-10*max(ev[-1],1.)
        assert ev[0]>-threshold
        self.trace_null=u[:,ev<=threshold];physical=u[:,ev>threshold]
        w=physical.T@self.W@physical
        assert np.min(la.eigvalsh(w))>0
        self.eigenvalues,z=la.eigh(physical.T@self.S@physical,w)
        self.steklov=physical@z
        self.global_gradient=g
        self.gradient_lu=sla.splu((g.T@g).tocsc())
        initial=np.zeros(1+base.na+base.np);initial[0]=1
        dc=base.magnetic(initial)
        self.port_trace=dc[self.gamma,None]
        self.port_trace-=self.trace_null@(self.trace_null.T@self.port_trace)
        self.port_trace=range_basis(self.port_trace)
        self.K=k
        self.harmonic_defect=relative(aa@self.extension,-ag)
        assert self.harmonic_defect<1e-9

    def basis(self,count,protect=True):
        trace=np.column_stack([self.trace_null,self.port_trace if protect else np.empty((len(self.gamma),0)),self.steklov[:,:count]])
        trace=range_basis(trace)
        q=np.zeros((self.base.na,len(self.d)+trace.shape[1]))
        q[self.d,np.arange(len(self.d))]=1
        q[self.gamma,len(self.d):]=trace
        q[self.air,len(self.d):]=self.extension@trace
        # Canonical global gradient quotient; change is curl-free only.
        q-=self.global_gradient@self.gradient_lu.solve(self.global_gradient.T@q)
        q=range_basis(q)
        assert np.min(la.eigvalsh(q.T@self.K@q))>0
        return q

    def export(self):
        return dict(conductor_indices=self.d.tolist(),interface_indices=self.gamma.tolist(),air_indices=self.air.tolist(),
          extension=self.extension.tolist(),interior_gradient_null=self.interior_null.tolist(),trace_null=self.trace_null.tolist(),
          S=self.S.tolist(),W=self.W.tolist(),eigenvalues=self.eigenvalues.tolist(),steklov=self.steklov.tolist(),
          port_trace=self.port_trace.tolist(),range_defect=self.range_defect,harmonic_defect=self.harmonic_defect)
