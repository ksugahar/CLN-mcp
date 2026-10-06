"""Historical virtual-port experiment, superseded for common-excitation comparisons.
Surface-port enrichment versus field POD on a common 3D diffusion pencil.

No physical surface-enriched Cauer ladder is inferred. The coupled Galerkin
system is compared with its *terminal-equivalent* positive Foster diagonalization.
Boundary ports are smooth magnetic trace functionals on the conductor surface;
they do not add analytic skin information or an exterior-air model.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve, qr
from surface_3d_model import build, TAU
from compare_3d import cln_columns, independent_basis
from positive_foster import response


def run(maxh):
    start=time.perf_counter()
    k,m,f,p,r,meta,ports=build(maxh)
    meta['assembly_reference_and_surface_ports_seconds']=time.perf_counter()-start
    print('mesh',meta,flush=True)
    freq=np.geomspace(100,1e6,301); u=2j*np.pi*freq*TAU
    s0=2*np.pi*1e4*TAU; energy=k+s0*m; chol=np.linalg.cholesky(energy)
    # Full reference fields from generalized eigenvectors, not reduced-model fits.
    p,vec=eigh(k,m); coupling=vec.T@f
    full=vec@(coupling[:,None]/(p[:,None]+u))
    exact=u*(f@full)
    ref_diss=np.real(np.sum(full.conj()*(m@full),axis=0))
    ref_mag=np.real(np.sum(full.conj()*(k@full),axis=0))
    assert np.max(abs(exact.real-(abs(u)**2)*ref_diss)/abs(exact))<1e-8
    direct_defects=[]
    for idx in [0,37,91,155,217,300]:
        direct=np.linalg.solve(k+u[idx]*m,f)
        direct_defects.append(np.linalg.norm(direct-full[:,idx])/np.linalg.norm(direct))
    assert max(direct_defects)<1e-8
    out={'status':'historical; virtual-port excitation; superseded for same-source comparisons by same_excitation_3d_*','scope':'Linear 3D notched internal conductor; same discrete pencil; boundary-trace virtual ports; no analytic skin/exterior correction',
         'mesh':meta,'frequency_hz':freq.tolist(),'reference_real':exact.real.tolist(),
         'reference_imag':exact.imag.tolist(),'tau':TAU,'bulk_states':2,
         'bulk_meaning':'two magnetic field states / four Type1 R-L energy elements, not two surface-coupled circuit sections',
         'direct_field_reference_difference':max(direct_defects),'runs':{}}
    t=time.perf_counter()
    bulkcols=cln_columns(k,m,f,s0,2)
    qb,br,_=independent_basis(bulkcols,energy); assert br==2
    wb=chol.T@qb
    # 12 smooth virtual surface ports, four real shifts. Each candidate is
    # normalized in the SAME fixed energy metric before basis selection.
    shifts=2*np.pi*np.geomspace(1e3,1e6,4)*TAU
    portnorm=np.sqrt(np.einsum('ij,ij->j',ports,cho_solve(cho_factor(energy),ports)))
    assert np.all(portnorm>0)
    port_vectors=[]
    for shift in shifts:
        lifts=cho_solve(cho_factor(k+shift*m),ports/portnorm)
        for j in range(lifts.shape[1]):port_vectors.append(lifts[:,j])
    candidates=np.column_stack(port_vectors)
    whitened=chol.T@candidates
    whitened/=np.linalg.norm(whitened,axis=0)
    residual=whitened-wb@(wb.T@whitened)
    left,sv,_=np.linalg.svd(residual,full_matrices=False)
    qsurf,_,pivot=qr(residual,mode='economic',pivoting=True)
    surface_precompute=time.perf_counter()-t
    out['surface_basis']={'virtual_ports':ports.shape[1],'candidate_columns':candidates.shape[1],
       'shifts':shifts.tolist(),'relative_singular_values':(sv/sv[0]).tolist(),
       'qr_pivots':pivot.tolist(),'offline_seconds':surface_precompute,
       'selection':'QR picks lifting columns; SVD truncation is POD of virtual-port lifting snapshots; no training physical-port solutions used'}
    # Separate actual-port POD, trained only on 31 high-band frequency fields.
    t=time.perf_counter(); trainfreq=np.geomspace(1e3,1e6,31)
    trainu=2j*np.pi*trainfreq*TAU
    trainfields=vec@(coupling[:,None]/(p[:,None]+trainu))
    trainwhite=chol.T@trainfields
    trainwhite/=np.sqrt(np.sum(abs(trainwhite)**2,axis=0))
    snapshots=np.column_stack([trainwhite.real,trainwhite.imag])
    pod,svpod,_=np.linalg.svd(snapshots,full_matrices=False)
    res=snapshots-wb@(wb.T@snapshots)
    hybridpod,svhyb,_=np.linalg.svd(res,full_matrices=False)
    out['field_pod']={'training_frequency_hz':trainfreq.tolist(),'snapshot_columns':snapshots.shape[1],
      'weighting':'complex field normalized in K+s0*M energy, then real/imaginary columns with equal frequency weights',
      'relative_singular_values':(svpod/svpod[0]).tolist(),'offline_seconds':time.perf_counter()-t,
      'cost_caveat':'training uses full-system spectral solves and a full eigensystem; this benchmark is not a sparse industrial cost estimate'}
    high=freq>=1e3; holdout=np.all(abs(freq[:,None]/trainfreq[None,:]-1)>1e-9,axis=1); out['holdout_points']=int(holdout.sum()); delta=np.sqrt(2/abs(u)); transition=(delta>=.3)&(delta<=3)
    def evaluate(label,columns,construction_seconds):
        if isinstance(columns,np.ndarray): columns=[columns[:,j] for j in range(columns.shape[1])]
        c=np.column_stack(columns)
        c=c/np.sqrt(np.einsum('ij,ij->j',c,energy@c))
        kc=c.T@k@c; mc=c.T@m@c
        coupling_ratio=(np.linalg.norm(kc[:2,2:])+s0*np.linalg.norm(mc[:2,2:]))/(np.linalg.norm(kc)+s0*np.linalg.norm(mc)) if label.startswith('bulk2 +') else None
        q,rank,svals=independent_basis(columns,energy)
        kq=q.T@k@q; mq=q.T@m@q; fq=q.T@f
        poles,v=eigh((kq+kq.T)/2,(mq+mq.T)/2); residues=(fq@v)**2
        assert np.all(poles>0) and np.all(residues>=0)
        y=np.column_stack([np.linalg.solve(kq+point*mq,fq) for point in u])
        field=q@y; z=u*(fq@y); foster=response(poles,residues,u)
        equiv=float(np.max(abs(foster-z)/abs(z))); assert equiv<1e-8
        error=abs(z/exact-1)
        diff=field-full
        field_error=np.sqrt(np.maximum(0,np.real(np.sum(diff.conj()*(energy@diff),axis=0)))/np.real(np.sum(full.conj()*(energy@full),axis=0)))
        diss=np.real(np.sum(y.conj()*(mq@y),axis=0))
        mag=np.real(np.sum(y.conj()*(kq@y),axis=0))
        # Common passivity/energy identity; this is not independent accuracy proof.
        assert np.max(abs(z.real-(abs(u)**2)*diss)/abs(z))<1e-8
        out['runs'][label]={'rank':rank,'construction_seconds':construction_seconds,'poles':poles.tolist(),
          'residues':residues.tolist(),'relative_error':error.tolist(),'field_energy_relative_error':field_error.tolist(),
          'joule_relative_error':abs(diss/ref_diss-1).tolist(),'magnetic_energy_relative_error':abs(mag/ref_mag-1).tolist(),
          'high_band_max':float(error[high].max()),'holdout_high_band_max':float(error[high & holdout].max()),'transition_max':float(error[transition].max()),
          'field_high_band_max':float(field_error[high].max()),'joule_high_band_max':float(abs(diss[high]/ref_diss[high]-1).max()),
          'DC_slope_error':abs(float(np.sum(residues/poles))-1),'coupled_vs_foster':equiv,
          'bulk_surface_coupling_ratio':None if coupling_ratio is None else float(coupling_ratio),
          'coupling_basis_caveat':'computed before rank SVD in the energy-normalized bulk/surface input basis'}
        print(label,'rank',rank,'high',error[high].max(),'field',field_error[high].max(),flush=True)
    evaluate('bulk CLN 2',bulkcols,0.)
    for n in [4,8,12]:
        t=time.perf_counter(); cols=cln_columns(k,m,f,s0,n); elapsed=time.perf_counter()-t
        evaluate(f'ordinary CLN {n}',cols,elapsed)
        for method,extra in [('surface QR',qsurf),('surface POD',left),('physical-port residual POD',hybridpod)]:
            cols=np.column_stack([qb,np.linalg.solve(chol.T,extra[:,:n-2])])
            evaluate(f'bulk2 + {method} {n}',cols,surface_precompute if 'surface' in method else out['field_pod']['offline_seconds'])
        evaluate(f'physical-port POD {n}',np.linalg.solve(chol.T,pod[:,:n]),out['field_pod']['offline_seconds'])
    out['review']='Codex self-review; independent Claude review pending'
    out['total_seconds']=time.perf_counter()-start
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maxh',type=float,default=.004);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(run(args.maxh),indent=2),encoding='utf-8')
    print('PASS: reference fields, positivity and coupled/Foster equivalence',flush=True)
