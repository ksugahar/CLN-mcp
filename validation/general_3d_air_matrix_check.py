"""Independent dense enclosure CLN checker; numpy/stdlib only.

Recurrence is checked from original matrices. Direct unit-current constrained
Taylor coefficients avoid inverse-admittance cancellation from series air
inductance. Refined native cohorts remain stored evidence, not CI FE reruns.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def dense(item):
    a = np.zeros(item["shape"])
    np.add.at(a,(item["row"],item["col"]),item["data"])
    return a


def inv_series(y):
    z=[1/y[0]]
    for n in range(1,len(y)):
        z.append(-sum(y[k]*z[n-k] for k in range(1,n+1))/y[0])
    return np.array(z)


def constrained_coefficients(h, derivative, rhs, count, tau):
    # Independent dense construction; no inverse admittance series.
    rows=np.ones(len(h));cols=rows.copy();a=h.copy()
    for _ in range(5):
        dr=1/np.sqrt(np.max(abs(a),axis=1));a=dr[:,None]*a;rows*=dr
        dc=1/np.sqrt(np.max(abs(a),axis=0));a=a*dc;cols*=dc
    inverse=np.linalg.inv(a)
    v=cols*(inverse@(rows*rhs));out=[v[-1]]
    for _ in range(1,count):
        v=cols*(inverse@(rows*(-derivative@v/tau)));out.append(v[-1])
    return np.array(out)


def series(R,L,b,s0,count,tau):
    n=len(b)
    h=np.block([[R+s0*L,-b[:,None]],[b[None,:],np.zeros((1,1))]])
    derivative=np.zeros_like(h);derivative[:n,:n]=L
    rhs=np.zeros(n+1);rhs[-1]=1
    return constrained_coefficients(h,derivative,rhs,count,tau)


def check_coefficients(actual, reference):
    """Check each dimensionless Taylor coefficient, including the small tail."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape
    scale = np.maximum(abs(reference), 1e-14*np.max(abs(reference)))
    assert np.max(abs(actual-reference)/scale) < 1e-7


def check_run(run, independent_full=None):
    n=run["modes"]
    assert n>=2 and len(run["Rhat"])==len(run["L"])==n
    assert abs(run["Rhat"][0]/run["moments"]["full_Z_coefficients"][0]-1)<1e-8
    assert min(run["Rhat"]+run["L"])>0
    r,l,b=map(np.array,[run["R_phys"],run["L_phys"],run["port"]])
    h=r+run["s0"]*l
    assert np.max(abs(h-np.diag(1/np.array(run["Rhat"]))))/np.max(abs(h))<1e-8
    assert np.linalg.eigvalsh(r).min()>0
    assert np.linalg.eigvalsh(l).min()>-1e-10*np.linalg.norm(l)
    assert abs(b[0]*run["Rhat"][0]-1)<1e-8
    assert np.max(abs(b[1:]))/abs(b[0])<1e-8
    assert max(run["electric_orth"],run["magnetic_orth"])<=1e-8
    for rh,energy in zip(run["Rhat"],run["energy"]):
        assert abs(rh*(energy["joule"]+run["s0"]*energy["magnetic"])-1)<1e-8
        assert max(abs(np.array(energy["field_integrals"])/[energy["joule"],energy["magnetic"]]-1))<1e-9
    m=run["moments"]
    assert m["contact_order"]==2*n
    regenerated=series(r,l,b,run["s0"],2*n+1,m["TAU"])
    full=np.array(m["full_Z_coefficients"]) if independent_full is None else independent_full
    assert np.max(abs(regenerated[:-1]-full[:-1]))/np.max(abs(full[:-1]))<1e-8
    coefficient_scale=np.maximum(abs(full[:-1]),1e-14*np.max(abs(full[:-1])))
    assert np.max(abs(regenerated[:-1]-full[:-1])/coefficient_scale)<1e-7
    assert np.max(abs(regenerated-np.array(m["galerkin_Z_coefficients"])))<1e-8*np.max(abs(regenerated))
    for row in run["frequency"]:
        assert row["ladder_galerkin_error"]<1e-10
        assert row["power_balance_error"]<1e-8
        zfull,zlad=complex(*row["full_Z"]),complex(*row["ladder_Z"])
        assert abs(abs(zlad/zfull-1)-row["reduction_error"])<1e-10
        for retained in range(1,n+1):
            q=2j*np.pi*row["hz"]-run["s0"]
            ladder=q*run["L"][retained-1]
            for index in range(retained-1,-1,-1):
                ladder+=run["Rhat"][index]
                if index:
                    branch=q*run["L"][index-1]
                    ladder=branch*ladder/(branch+ladder)
            galerkin=1/(b[:retained]@np.linalg.solve(r[:retained,:retained]+2j*np.pi*row["hz"]*l[:retained,:retained],b[:retained]))
            assert abs(ladder/galerkin-1)<1e-10
    for retained in range(1,n+1):
        prefix=series(r[:retained,:retained],l[:retained,:retained],b[:retained],run["s0"],2*retained+1,m["TAU"])
        assert np.max(abs(prefix[:-1]-full[:2*retained]))/np.max(abs(full[:2*retained]))<1e-8
        coefficient_scale=np.maximum(abs(full[:2*retained]),1e-14*np.max(abs(full[:2*retained])))
        assert np.max(abs(prefix[:-1]-full[:2*retained])/coefficient_scale)<1e-7


def check_small(data):
    k,m,s,c,g=map(lambda key:dense(data[key]),["K","M","S","C","G"])
    f,g0=np.array(data["f"]),data["G0"]
    na,np_=len(f),len(s)
    kg=k+data["gauge_scale"]*(g@g.T)
    ki,si=np.linalg.inv(kg),np.linalg.inv(s)
    assert np.linalg.matrix_rank(g)==g.shape[1]
    eigen=np.linalg.eigvalsh(k)
    assert np.sum(abs(eigen)<1e-10*np.max(abs(eigen)))==g.shape[1]
    h=np.block([[np.array([[g0]]),f[None,:],np.zeros((1,np_))],
                [f[:,None],m,-c.T],[np.zeros((np_,1)),-c,s]])
    load=np.column_stack([f,m,-c.T])
    lp=load.T@ki@load
    port=np.r_[g0,f,np.zeros(np_)]
    base=np.zeros(1+na+np_);base[0]=1
    def response(v,s0):
        d=c@v[1:1+na]-s@v[1+na:]
        a=np.linalg.solve(kg+s0*(m-c.T@si@c),load@v-c.T@si@d)
        chi=si@(d-s0*c@a)
        out=v.copy();out[1:1+na]-=s0*a;out[1+na:]+=chi
        return out,a
    for run in data["runs"]:
        check_run(run)
        e,a=response(base,run["s0"])
        x=np.zeros_like(e)
        for index in range(run["modes"]):
            stored=np.array(run["electric_modes"][index])
            assert np.linalg.norm(e-stored)/np.linalg.norm(stored)<1e-8
            assert np.linalg.norm(a-np.array(run["magnetic_potentials"][index]))/np.linalg.norm(a)<1e-8
            rh=1/(e@(h+run["s0"]*lp)@e)
            assert abs(rh/run["Rhat"][index]-1)<1e-8
            x+=rh*e
            ax=ki@(load@x)
            lk=ax@k@ax
            assert abs(lk/run["L"][index]-1)<1e-8
            assert np.linalg.norm(c@e[1:1+na]-s@e[1+na:])<1e-8*np.linalg.norm(load@e)
            u=np.zeros_like(e);u[1:1+na]=ax
            correction,ac=response(u,run["s0"])
            e-=correction/lk;a-=ac/lk
        # Full mixed Taylor coefficients derived independently from the export.
        mbar=m-c.T@si@c
        s0=run['s0']
        terminal=np.block([[kg+s0*mbar,-f[:,None]],[-s0*f[None,:],np.array([[g0]])]])
        derivative=np.block([[mbar,np.zeros((na,1))],[-f[None,:],np.zeros((1,1))]])
        rhs=np.zeros(na+1);rhs[-1]=1
        full=constrained_coefficients(terminal,derivative,rhs,2*run['modes']+1,run['moments']['TAU'])
        saved=np.array(run["moments"]["full_Z_coefficients"])
        check_coefficients(saved, full)
        check_run(run, independent_full=full)
    for name,matrices in data["cross_formulations"].items():
        r,rg,b=map(lambda v:np.array(v),[dense(matrices["R"]),dense(matrices["Rg"]),matrices["b"]])
        if name=="A-T":
            d=dense(matrices["D"]);l=d.T@ki@d
        else:
            u,w,so=map(lambda key:dense(matrices[key]),["U","W","So"])
            l=u-w.T@np.linalg.solve(so,w)
        for run in data["cross_formulation_runs"][name]:
            check_run(run)
            a=np.linalg.inv(rg+run["s0"]*l)
            j=a@b;x=np.zeros_like(j)
            for index in range(run["modes"]):
                stored=np.array(run["current_mode_coefficients"][index])
                assert np.linalg.norm(j-stored)/np.linalg.norm(stored)<1e-7
                rh=1/(j@(r+run["s0"]*l)@j);x+=rh*j;lk=x@l@x
                assert abs(rh/run["Rhat"][index]-1)<1e-8
                assert abs(lk/run["L"][index]-1)<1e-8
                j-=a@(l@x)/lk
            full=series(rg,l,b,run["s0"],2*run["modes"]+1,run["moments"]["TAU"])
            saved=np.array(run["moments"]["full_Z_coefficients"])
            check_coefficients(saved, full)
            check_run(run, independent_full=full)
