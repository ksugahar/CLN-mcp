"""Positive RL Foster fitting with explicit static constraints.

Z(u)=D + L_tail*u + sum r_j*u/(u+p_j), positive p_j,r_j,D,L_tail.
The DC-constrained variant fixes D=0 and L_tail+sum r_j/p_j=1.
Only fitted input/output behavior is asserted; fitted poles need not be the
physical eigenvalues, and a field reconstruction is not inferred from a fit.
"""
import math,time
import numpy as np
from scipy.optimize import least_squares
from scipy.special import softmax

def response(poles,residues,u):
    return u*np.sum(residues[None,:]/(u[:,None]+poles),axis=1)


def foster_fit(u,target,n,poles,dc,seed=None):
    p0=np.geomspace(max(poles[0],abs(u[0])/5),min(poles[-1],abs(u[-1])*5),n)
    if seed is not None:
        p0=np.asarray(seed['poles'])
    if dc:
        if seed is None:
            theta0=np.r_[np.log(p0),np.zeros(n)]
        else:
            w0=np.asarray(seed['residues'])/p0
            tail=max(1-float(w0.sum()),1e-9)
            w0=w0*(1-tail)/w0.sum()
            theta0=np.r_[np.log(p0),np.log(w0/tail)]
        def model(theta,points):
            p=np.exp(theta[:n]); w=softmax(np.r_[theta[n:],0.])
            return response(p,w[:n]*p,points)+w[-1]*points
    else:
        r0=np.asarray(seed['residues']) if seed is not None else np.maximum(p0/(n*(1+p0)),1e-12)
        theta0=np.r_[np.log(p0),np.log(r0),math.log(max(seed.get('DC_offset',1e-13),1e-13) if seed is not None else 1e-9),math.log(max(seed.get('L_tail',1e-13),1e-13) if seed is not None else 1e-9)]
        def model(theta,points):
            return response(np.exp(theta[:n]),np.exp(theta[n:2*n]),points)+np.exp(theta[-2])+np.exp(theta[-1])*points
    def residual(theta):
        e=(model(theta,u)-target)/abs(target)
        return np.r_[e.real,e.imag]
    t=time.perf_counter()
    fit=least_squares(residual,theta0,bounds=(-30,30),max_nfev=5000,x_scale="jac",ftol=1e-10,gtol=1e-9,xtol=1e-10)
    if not fit.success:
        raise RuntimeError(f'Foster fit failed: {fit.message}; max training error={np.max(abs(residual(fit.x))):.3e}')
    params={'poles':np.exp(fit.x[:n]).tolist(),'seconds':time.perf_counter()-t,'nfev':int(fit.nfev),'optimizer_converged':bool(fit.success), 'training_relative_rms':float(np.sqrt(np.mean(residual(fit.x)**2)))}
    if dc:
        w=softmax(np.r_[fit.x[n:],0.]); params.update(residues=(w[:n]*np.exp(fit.x[:n])).tolist(),L_tail=float(w[-1]),DC_offset=0.,DC_slope_error=abs(float(w.sum())-1))
    else:
        r=np.exp(fit.x[n:2*n]); params.update(residues=r.tolist(),L_tail=float(np.exp(fit.x[-1])),DC_offset=float(np.exp(fit.x[-2])),DC_slope_error=abs(float(np.sum(r/np.exp(fit.x[:n]))+np.exp(fit.x[-1]))-1))
    return lambda points:model(fit.x,points),params
