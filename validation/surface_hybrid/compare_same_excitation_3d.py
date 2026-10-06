"""CLN and POD driven by the SAME physical source on a common 3D pencil.

No physical surface-enriched Cauer ladder is inferred. The coupled Galerkin
system is compared with its *terminal-equivalent* positive Foster diagonalization.
Boundary functionals OBSERVE physical-source snapshots; they never drive new solves.
They are not exterior Schur/Steklov modes. All CLN/POD bases use the same f.
The boundary observations add no analytic skin information or exterior air.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve
from surface_3d_model import build, TAU
from compare_3d import cln_columns, independent_basis
from positive_foster import response


def run(maxh):
    start=time.perf_counter()
    k,m,f,p,r,meta,ports=build(maxh)
    meta['assembly_reference_and_boundary_observation_seconds']=time.perf_counter()-start
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
    out={'scope':'Linear 3D notched internal conductor; same discrete pencil; SAME physical source f for CLN and snapshots; boundary observation only; no Steklov/analytic skin/exterior correction',
         'mesh':meta,'frequency_hz':freq.tolist(),'reference_real':exact.real.tolist(),
         'reference_imag':exact.imag.tolist(),'tau':TAU,'bulk_states':2,
         'bulk_meaning':'two magnetic field states / four Type1 R-L energy elements, not two surface-coupled circuit sections',
         'direct_field_reference_difference':max(direct_defects),'runs':{}}
    t=time.perf_counter()
    bulkcols=cln_columns(k,m,f,s0,2)
    qb,br,_=independent_basis(bulkcols,energy); assert br==2
    wb=chol.T@qb
    # Frequency snapshots driven by EXACTLY the same f as the CLN recurrence.
    t=time.perf_counter(); trainfreq=np.geomspace(1e3,1e6,31)
    trainu=2j*np.pi*trainfreq*TAU
    trainfields=vec@(coupling[:,None]/(p[:,None]+trainu))
    source_residual=float(np.max(np.linalg.norm(k@trainfields+(m@trainfields)*trainu-f[:,None],axis=0)/np.linalg.norm(f)))
    initial_source_residual=float(np.linalg.norm((k+s0*m)@bulkcols[0]-f)/np.linalg.norm(f))
    assert source_residual<1e-8 and initial_source_residual<1e-8
    out['common_source_checks']={'snapshot_equation_residual':source_residual,'CLN_initial_equation_residual':initial_source_residual}
    trainwhite=chol.T@trainfields
    trainwhite/=np.sqrt(np.sum(abs(trainwhite)**2,axis=0))
    snapshots=np.column_stack([trainwhite.real,trainwhite.imag])
    pod,svpod,_=np.linalg.svd(snapshots,full_matrices=False)
    res=snapshots-wb@(wb.T@snapshots)
    hybridpod,svhyb,_=np.linalg.svd(res,full_matrices=False)
    out['field_pod']={'training_frequency_hz':trainfreq.tolist(),'snapshot_columns':snapshots.shape[1],
      'weighting':'complex field normalized in K+u0*M energy, then real/imaginary columns with equal frequency weights',
      'relative_singular_values':(svpod/svpod[0]).tolist(),'offline_seconds':time.perf_counter()-t,
      'cost_caveat':'offline_seconds is partial and excludes the full eigensystem; total POD construction cost is unavailable. No offline-cost ranking.'}
    # Observe actual-source residual snapshots at the boundary. No virtual
    # port-response snapshots are generated (static dual norms use solves).
    # Boundary-selected directions are full-field
    # combinations of the same snapshots, NOT independent Steklov modes.
    t=time.perf_counter()
    trace_norm=np.sqrt(np.einsum('ij,ij->j',ports,cho_solve(cho_factor(energy),ports)))
    observation=(ports/trace_norm).T
    boundary_snapshots=observation@np.linalg.solve(chol.T,res)
    _,trace_sv,trace_right=np.linalg.svd(boundary_snapshots,full_matrices=False)
    trace_selected=res@trace_right.T
    out['boundary_selection']={'observation_functionals':ports.shape[1],
      'training_source':'same physical f as CLN; no virtual forcing',
      'relative_singular_values':(trace_sv/trace_sv[0]).tolist(),
      'offline_seconds':time.perf_counter()-t,
      'scope':'POD of boundary observations; snapshot coefficient vectors lifted to full residual fields. Not exterior Steklov modes or localized surface fields.'}
    out['source_contract']={'CLN':'A(u0)^-1 f then physical Type1 recurrences',
      'multipoint_CLN':'A(real shifts)^-1 f and physical recurrences at each shift',
      'POD':'A(i*omega*tau)^-1 f at training frequencies',
      'boundary_selected_POD':'observations of the SAME f-driven frequency snapshots',
      'no_virtual_port_excitation':True,'comparison_caveat':'same source and same state counts, different frequency information and offline solve budgets'}
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
        out['runs'][label]={'rank':rank,'construction_seconds':None if 'POD' in label else construction_seconds,'poles':poles.tolist(),
          'residues':residues.tolist(),'relative_error':error.tolist(),'field_energy_relative_error':field_error.tolist(),
          'joule_relative_error':abs(diss/ref_diss-1).tolist(),'magnetic_energy_relative_error':abs(mag/ref_mag-1).tolist(),
          'high_band_max':float(error[high].max()),'holdout_high_band_max':float(error[high & holdout].max()),'transition_max':float(error[transition].max()),
          'field_high_band_max':float(field_error[high].max()),'holdout_field_high_band_max':float(field_error[high & holdout].max()),'holdout_joule_high_band_max':float(abs(diss[high & holdout]/ref_diss[high & holdout]-1).max()),'joule_high_band_max':float(abs(diss[high]/ref_diss[high]-1).max()),
          'DC_slope_error':abs(float(np.sum(residues/poles))-1),'coupled_vs_foster':equiv,'positive_energy_pencil':True,
          'bulk_enrichment_coupling_ratio':None if coupling_ratio is None else float(coupling_ratio),
          'coupling_basis_caveat':'computed before rank SVD in the energy-normalized bulk/enrichment input basis'}
        if 'POD' in label:
            out['runs'][label]['basis_processing_seconds_partial']=construction_seconds
            out['runs'][label]['timing_scope']='POD total construction cost unavailable: full eigensystem cost excluded'
        print(label,'rank',rank,'high',error[high].max(),'field',field_error[high].max(),flush=True)
    evaluate('bulk CLN 2',bulkcols,None)
    for n in [4,8,12]:
        t=time.perf_counter(); cols=cln_columns(k,m,f,s0,n); elapsed=time.perf_counter()-t
        evaluate(f'ordinary CLN {n}',cols,elapsed)
        t=time.perf_counter()
        shifts=2*np.pi*np.geomspace(1e3,1e6,min(4,n))*TAU
        allocation=[n//len(shifts)+(j<n%len(shifts)) for j in range(len(shifts))]
        cols=[]
        for shift,count in zip(shifts,allocation):
            cols.extend(cln_columns(k,m,f,float(shift),int(count)))
        elapsed=time.perf_counter()-t
        evaluate(f'multipoint CLN {n}',cols,elapsed)
        out['runs'][f'multipoint CLN {n}']['shifts']=shifts.tolist()
        for method,extra in [('boundary-response POD',trace_selected),('field-residual POD',hybridpod)]:
            cols=np.column_stack([qb,np.linalg.solve(chol.T,extra[:,:n-2])])
            evaluate(f'bulk2 + {method} {n}',cols,out['field_pod']['offline_seconds']+out['boundary_selection']['offline_seconds'] if method=='boundary-response POD' else out['field_pod']['offline_seconds'])
        evaluate(f'field POD {n}',np.linalg.solve(chol.T,pod[:,:n]),out['field_pod']['offline_seconds'])
    out['review']='Codex self-review; independent Claude review pending'
    out['total_seconds']=time.perf_counter()-start
    return out

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maxh',type=float,default=.004);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(run(args.maxh),indent=2),encoding='utf-8')
    print('PASS: reference fields, positivity and coupled/Foster equivalence',flush=True)
