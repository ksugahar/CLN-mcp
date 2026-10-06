"""Direct impedance Taylor series with a work-conjugate unit-current constraint.

Avoid inversion of a voltage-driven admittance series when external series
inductance makes its high-order inverse coefficients cancellation-sensitive.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

def taylor(matrix, derivative, rhs, count, tau):
    from general_shape_air import refined_solve
    a=matrix.tocsc();rows=np.ones(a.shape[0]);cols=rows.copy()
    for _ in range(5):
        dr=1/np.sqrt(np.asarray(abs(a).max(axis=1).toarray()).ravel())
        a=sp.diags(dr)@a;rows*=dr
        dc=1/np.sqrt(np.asarray(abs(a).max(axis=0).toarray()).ravel())
        a=a@sp.diags(dc);cols*=dc
    a=a.tocsc();lu=sla.splu(a)
    x=cols*refined_solve(lu,a,rows*rhs);out=[float(x[-1])]
    for _ in range(1,count):
        forcing=-derivative@x/tau
        x=cols*refined_solve(lu,a,rows*forcing);out.append(float(x[-1]))
    return np.array(out)

def mixed(model,s0,count,tau):
    na,np_=model.na,model.np;zero=sp.csc_matrix((np_,1));f=sp.csc_matrix(model.f[:,None])
    h=sp.bmat([[model.Kg+s0*model.M,model.C.T,-f],[s0*model.C,model.S,zero],[-s0*f.T,zero.T,sp.csc_matrix([[model.G0]])]],format='csc')
    d=sp.bmat([[model.M,sp.csc_matrix((na,np_)),sp.csc_matrix((na,1))],[model.C,sp.csc_matrix((np_,np_)),zero],[-f.T,zero.T,sp.csc_matrix((1,1))]],format='csc')
    rhs=np.zeros(na+np_+1);rhs[-1]=1
    return taylor(h,d,rhs,count,tau)

def current(model,s0,count,tau):
    n=len(model.b);na=model.base.na;b=sp.csc_matrix(model.b[:,None]);zero=sp.csc_matrix((na,1))
    h=sp.bmat([[model.Rg,s0*model.D.T,-b],[-model.D,model.base.Kg,zero],[b.T,zero.T,sp.csc_matrix((1,1))]],format='csc')
    d=sp.bmat([[sp.csc_matrix((n,n)),model.D.T,sp.csc_matrix((n,1))],[sp.csc_matrix((na,n)),sp.csc_matrix((na,na)),zero],[sp.csc_matrix((1,n)),zero.T,sp.csc_matrix((1,1))]],format='csc')
    rhs=np.zeros(n+na+1);rhs[-1]=1
    return taylor(h,d,rhs,count,tau)

def reduced(r,l,b,s0,count,tau):
    b=sp.csc_matrix(b[:,None]);n=r.shape[0]
    h=sp.bmat([[sp.csc_matrix(r+s0*l),-b],[b.T,sp.csc_matrix((1,1))]],format='csc')
    d=sp.block_diag([sp.csc_matrix(l),sp.csc_matrix((1,1))],format='csc');rhs=np.zeros(n+1);rhs[-1]=1
    return taylor(h,d,rhs,count,tau)
