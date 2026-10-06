"""Check stored general-3D physical CLN evidence using numpy and stdlib only.

The small assembled example is recomputed here. Refined native FEM runs are
stored evidence, not rerun by this validator. No analytic 3D oracle exists.
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


def series(R,L,b,s0,count,tau):
    h=R+s0*L
    v=np.linalg.solve(h,b)
    coefficients=[b@v]
    for _ in range(1,count):
        v=-np.linalg.solve(h,L@v)/tau
        coefficients.append(b@v)
    return inv_series(coefficients)


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
        ee,aa=response(base,run["s0"])
        y=[port@ee]
        tau=run["moments"]["TAU"]
        for _ in range(2*run["modes"]):
            u=np.zeros_like(base);u[1:1+na]=-aa/tau
            ee,aa=response(u,run["s0"]);y.append(port@ee)
        full=inv_series(y)
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


def main():
    files=[ROOT/"docs/data"/name for name in ["general_3d_cln_p1.json","general_3d_cln_p2.json","general_3d_cln_matrices.json"]]
    for path in files:
        data=json.loads(path.read_text())
        assert data["complete"]
        for name,identity in data["source_sha256"].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==identity,name
        if "cases" in data:
            for case in data["cases"]:
                assert case["seconds"]>0 and len(case["mesh"]["mesh_sha256"])==64
                assert case["kernel_defect"]<1e-10 and case["kernel_fraction_max"]<=1e-8
                assert case["original_residual_max"]<=1e-8
                assert case["mesh"]["phi_departure_l2_relative"]>1e-3
                assert case["mesh"]["dc_transverse_current_fraction"]>1e-3
                for run in case["runs"]:
                    assert run["modes"]==4
                    check_run(run)
                    assert run["gauge_port_invariance_max"]<1e-8
                    for row in run["frequency"]:
                        assert row["full_field_energy_defect"]<1e-9
                        assert row["terminal_reactions"]["relative_defect"]<1e-8
                        if row["hz"] in (1e5,1e6): assert not row["mesh_resolution_screen"]
                for runs in case["cross_formulations"].values():
                    for run in runs:
                        assert run["modes"]==4
                        check_run(run);assert run["original_residual_max"]<1e-8
        else:
            assert all(run["modes"]==4 for run in data["runs"])
            check_small(data)
    checks=json.loads((ROOT/"docs/data/general_3d_cln_checks.json").read_text())
    assert checks["complete"]
    for name,identity in checks["source_sha256_lf"].items():
        assert hashlib.sha256((ROOT/name).read_text(encoding="utf-8-sig").encode()).hexdigest()==identity
    assert max(value for rows in checks["quadrature_element_relative_changes"].values() for value in rows)<1e-8
    assert max(checks["iccg_LU_dc_field_relative"],checks["iccg_LU_magnetic_field_relative"])<1e-8
    for info in [checks["iccg_phi"],checks["iccg_A"]]:
        assert info["converged"] and info["original_rhs_residual"]<1e-8
    print("PASS: physical general-3D CLN, independent small-matrix recursion, contact order 2m, stored refinement evidence")


if __name__=="__main__":
    main()
