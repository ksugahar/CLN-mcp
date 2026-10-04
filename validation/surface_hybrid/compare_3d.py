"""Common-model 3D A-phi diffusion experiment (Codex self-review).

A notched conductor with axial voltage lifting gives a genuinely 3D field.
Magnetic potential has tangential homogeneous boundary conditions; electric
potential is prescribed on the end faces and natural on other faces. This is
an internal-field benchmark, not a complete open-boundary device model.
The scalar potential is eliminated from the conductivity matrix; the curl
kernel is removed explicitly. All reduced models use this same pencil.
Direct reference uses sparse LU and dense symmetric eigensolves, independently
of the production ICCG. High-frequency errors against this *discrete* reference
do not establish continuum accuracy when the skin depth is unresolved.
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh, cho_factor, cho_solve
from positive_foster import response, foster_fit
from ladder_eval import z_type1
from netgen.occ import Box, OCCGeometry, Z
from ngsolve import Mesh, H1, HCurl, GridFunction, BilinearForm, LinearForm, grad, curl, dx, TaskManager

MU=4e-7*math.pi; SIGMA=1e6; A=.01; HEIGHT=.015; TAU=MU*SIGMA*A*A


def csr(bf):
    return sp.csr_matrix(bf.mat.CSR()).copy()


def build(maxh):
    shape=Box((-A,-A,0),(A,A,HEIGHT))-Box((.002,.002,HEIGHT/2),(.012,.012,HEIGHT+.002))
    shape.faces.name='surface'; shape.faces.Min(Z).name='in'; shape.faces.Max(Z).name='out'
    mesh=Mesh(OCCGeometry(shape).GenerateMesh(maxh=maxh))
    va=HCurl(mesh,order=1,nograds=False,dirichlet='surface|in|out')
    vp=H1(mesh,order=2,dirichlet='in|out')
    freea=np.fromiter(va.FreeDofs(),bool,va.ndof); freep=np.fromiter(vp.FreeDofs(),bool,vp.ndof)
    pa,qa=va.TnT(); pp,qp=vp.TnT()
    kp=BilinearForm(vp); kp+=SIGMA*grad(pp)*grad(qp)*dx; kp.Assemble()
    g=GridFunction(vp); g.Set(HEIGHT,definedon=mesh.Boundaries('in'))
    kp0=csr(kp)[freep][:,freep].tocsc(); lu=spla.splu(kp0)
    lift=g.vec.CreateVector(); lift.data=kp.mat*g.vec
    rhs=-lift.FV().NumPy()[freep].copy()
    phi=lu.solve(rhs)
    assert np.linalg.norm(kp0@phi-rhs)/np.linalg.norm(rhs)<1e-10
    g.vec.FV().NumPy()[freep]=phi
    e0=-grad(g)
    ka=BilinearForm(va); ka+=curl(pa)*curl(qa)/MU*dx; ka.Assemble()
    ma=BilinearForm(va); ma+=SIGMA*pa*qa*dx; ma.Assemble()
    lf=LinearForm(va); lf+=SIGMA*e0*qa*dx; lf.Assemble()
    k=csr(ka)[freea][:,freea].toarray(); m=csr(ma)[freea][:,freea].toarray()
    f=lf.vec.FV().NumPy()[freea].copy()
    mixed=va*vp; (ua,up),(wa,wp)=mixed.TnT()
    cross=BilinearForm(mixed); cross+=SIGMA*ua*grad(wp)*dx; cross.Assemble()
    coupling=csr(cross)[va.ndof:,:va.ndof][freep][:,freea].toarray()
    # A-phi conductivity: eliminate the correction potential with homogeneous end values.
    m-=coupling.T@lu.solve(coupling)
    m=(m+m.T)/2; k=(k+k.T)/2
    lam,q=eigh(k)
    keep=lam>lam.max()*1e-10
    kernel_load=float(np.linalg.norm(q[:,~keep].T@f)/np.linalg.norm(f))
    assert kernel_load<1e-8
    q=q[:,keep]; kr=q.T@k@q; mr=q.T@m@q; fr=q.T@f
    kr=(kr+kr.T)/2; mr=(mr+mr.T)/2
    assert eigh(mr,eigvals_only=True)[0]>0
    l0=float(fr@np.linalg.solve(kr,fr))
    # H(u)=H(s)/H(0), u=s*tau. Z(u)=u*H(u). Exact static slope is 1.
    mr=mr/TAU; fr=fr/math.sqrt(l0)
    poles,v=eigh(kr,mr); residues=(fr@v)**2
    assert np.all(poles>0) and np.all(residues>=0)
    assert abs(np.sum(residues/poles)-1)<1e-8
    return kr,mr,fr,poles,residues,{'maxh':maxh,'ne':mesh.ne,'free_A':int(freea.sum()),
            'physical_dimension':len(poles),'gradient_dimension':int((~keep).sum()),
            'kernel_load_fraction':kernel_load,'static_H':l0}




def independent_basis(columns,energy,tol=1e-11):
    """Rank reveal after normalizing each field in a positive energy norm."""
    c=np.column_stack(columns)
    c=c/np.sqrt(np.einsum('ij,ij->j',c,energy@c))
    chol=np.linalg.cholesky(energy)
    u,s,_=np.linalg.svd(chol.T@c,full_matrices=False)
    rank=int(np.sum(s>s[0]*tol))
    return np.linalg.solve(chol.T,u[:,:rank]),rank,(s/s[0]).tolist()


def cln_columns(k,m,f,s0,n,return_elements=False):
    """Type1 physical magnetic modes, port-shorted correction solves."""
    energy=k+s0*m; fac=cho_factor(energy)
    x=cho_solve(fac,f); mag=x; l=float(x@energy@x)
    cols=[x]; elements=[l]; previous_e=None
    for j in range(n-1):
        electric=mag/(s0*l)+(previous_e if previous_e is not None else 0)
        r=1/(s0*s0*float(electric@m@electric))
        elements.append(r)
        correction=cho_solve(fac,-s0*r*(m@electric))
        correction-=float(f@correction)/float(f@x)*x
        nxt=correction+(mag if j>0 else 0)
        ln=float(nxt@energy@nxt)
        if not np.isfinite(ln) or ln<=0:
            break
        cols.append(nxt); elements.append(ln)
        previous_e=electric; mag=nxt; l=ln
    if return_elements:
        electric=mag/(s0*l)+(previous_e if previous_e is not None else 0)
        elements.append(1/(s0*s0*float(electric@m@electric)))
        return cols,elements
    return cols


def krylov_columns(k,m,f,s0,n):
    """Independent twice-reorthogonalized span check for the physical CLN modes."""
    energy=k+s0*m; factor=cho_factor(energy)
    w=cho_solve(factor,f); cols=[]
    for j in range(n):
        before=float(w@energy@w)
        for _ in range(2):
            for v in cols:
                w-=v*float(v@energy@w)
        after=float(w@energy@w)
        if after<before*1e-26:
            break
        w=w/math.sqrt(after); cols.append(w)
        w=cho_solve(factor,m@w)
    return cols


def ritz(k,m,f,columns,u):
    q,rank,svals=independent_basis(columns,k+m)
    kq=q.T@k@q; mq=q.T@m@q; fq=q.T@f
    kq=(kq+kq.T)/2; mq=(mq+mq.T)/2
    p,v=eigh(kq,mq); r=(fq@v)**2
    z=response(p,r,u)
    return z,{'rank':rank,'relative_singular_values':svals,
              'DC_slope_error':abs(float(np.sum(r/p))-1),'poles':p.tolist(),'residues':r.tolist()}




def experiment(maxh):
    t=time.perf_counter()
    with TaskManager():
        k,m,f,p,r,meta=build(maxh)
    meta['reference_build_seconds']=time.perf_counter()-t
    print('3D reference:',meta,flush=True)
    freq=np.geomspace(100,1e6,301); u=2j*np.pi*freq*TAU
    exact=response(p,r,u); delta=np.sqrt(2/abs(u)); transition=(delta>=.3)&(delta<=3); high=freq>=1e3
    train_low=np.geomspace(.1,3e4,90); train_high=np.geomspace(1e3,1e6,90)
    u_low=2j*np.pi*train_low*TAU; u_high=2j*np.pi*train_high*TAU
    out={'scope':'3D notched conductor; same discrete A-phi pencil, normalized reaction transfer',
         'mesh':meta,'frequency_hz':freq.tolist(),'tau':TAU,'transition_delta_over_a':[.3,3],
         'reference_real':exact.real.tolist(),'reference_imag':exact.imag.tolist(),'runs':{}}
    def save(label,z,info):
        e=abs(z/exact-1)
        assert np.all(np.isfinite(e))
        assert np.min(z.real)>-1e-10
        info.update(relative_error=e.tolist(),transition_max=float(e[transition].max()),high_band_max=float(e[high].max()),at_1MHz=float(e[-1]))
        out['runs'][label]=info
        print(label,{x:info[x] for x in ['transition_max','high_band_max','DC_slope_error']},flush=True)
    for n in [4,8,12]:
        s0=2*np.pi*1e4*TAU
        t=time.perf_counter(); cols,elements=cln_columns(k,m,f,s0,n,True)
        z,info=ritz(k,m,f,cols,u)
        circuit=np.asarray([z_type1(elements,s0,point) for point in u])
        circuit_difference=float(np.max(abs(circuit/z-1)))
        if circuit_difference>1e-6:
            raise RuntimeError(f'Energy ladder/Ritz mismatch at rank {n}: {circuit_difference}')
        assert all(value>0 for value in elements)
        info.update(circuit_elements=elements,circuit_elements_count=len(elements),
                    circuit_vs_ritz=circuit_difference,termination='Type1 ending on R')
        check_cols=krylov_columns(k,m,f,s0,n)
        check_z,check_info=ritz(k,m,f,check_cols,u)
        diff=float(np.max(abs(z/check_z-1)))
        if diff>1e-6:
            raise RuntimeError(f'CLN/Krylov span check failed at rank {n}: {diff}')
        info.update(requested_rank=n,seconds=time.perf_counter()-t,s0=s0,independent_span_response_difference=diff)
        save(f'CLN single point rank {n}',z,info)
        t=time.perf_counter(); cols=cln_columns(k,m,f,s0,n-1)+[np.linalg.solve(k,f)]
        z,info=ritz(k,m,f,cols,u); info.update(requested_rank=n,seconds=time.perf_counter()-t)
        save(f'CLN single+DC rank {n}',z,info)
        shifts=np.geomspace(abs(u_high[0]),abs(u_high[-1]),min(4,n))
        for dc in [False,True]:
            count=n-int(dc); allocation=[count//len(shifts)+(j<count%len(shifts)) for j in range(len(shifts))]
            t=time.perf_counter(); cols=[]
            for shift,num in zip(shifts,allocation):
                if num: cols.extend(cln_columns(k,m,f,float(shift),int(num)))
            if dc: cols.append(np.linalg.solve(k,f))
            z,info=ritz(k,m,f,cols,u); info.update(requested_rank=n,seconds=time.perf_counter()-t,shifts=shifts.tolist())
            save(f'CLN multipoint {"DC" if dc else "band"} rank {n}',z,info)
        z=response(p[:n],r[:n],u)
        save(f'Foster slow modes rank {n}',z,{'rank':n,'DC_slope_error':abs(float(np.sum(r[:n]/p[:n]))-1)})
        ltail=max(0.,1-float(np.sum(r[:n]/p[:n])))
        save(f'Foster slow+DC-tail rank {n}',z+ltail*u,{'rank':n,'DC_slope_error':0.,'L_tail':ltail})
        if n == 4:
            for dc,high_target,label in [(True,False,'DC'),(True,True,'high+DC'),(False,False,'low-free-DC'),(False,True,'band')]:
                utr=u_high if high_target else u_low
                seed_label=f'CLN multipoint {"DC" if dc else "band"} rank {n}' if high_target else f'CLN single+DC rank {n}'
                if not dc:
                    seed_label=f'Foster fitted {"high+DC" if high_target else "DC"} rank {n}'
                seed=out['runs'][seed_label]
                fitted,info=foster_fit(utr,response(p,r,utr),n,p,dc,seed)
                training=train_high if high_target else train_low
                info.update(rank=n,train_band_hz=[float(training[0]),float(training[-1])],seed=seed_label,seed_seconds=seed['seconds'],static_constraints='Z(0)=0 and static slope=1' if dc else 'DC value and slope unconstrained')
                save(f'Foster fitted {label} rank {n}',fitted(u),info)
    # Direct dense solves at holdout points independently verify the modal reference.
    defects=[]
    for i in [0,37,91,155,217,300]:
        sol=np.linalg.solve(k+u[i]*m,f)
        defects.append(abs(u[i]*(f@sol)/exact[i]-1))
    assert max(defects)<1e-8
    out['direct_reference_relative_difference']=float(max(defects))
    # Inspect attainable physical mode ranks beyond the comparison orders.
    out['depth_probe']=[]
    for n in [16,24,32]:
        cols=cln_columns(k,m,f,s0,n)
        _,rank,sv=independent_basis(cols,k+m)
        out['depth_probe'].append({'requested':n,'constructed_columns':len(cols),'independent_rank':rank,'relative_singular_values':sv})
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maxh',type=float,default=.004)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    data=experiment(args.maxh)
    Path(args.output).write_text(json.dumps(data,indent=2),encoding='utf-8')
    print('PASS: common-model reference and positivity checks',flush=True)

if __name__=='__main__': main()
