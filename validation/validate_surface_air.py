"""Independent surface CLN checks using original matrices; numpy/stdlib only."""
import json,hashlib,math
from pathlib import Path
import numpy as np
from general_3d_air_matrix_check import dense,constrained_coefficients
ROOT=Path(__file__).resolve().parents[1]


def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(a),np.linalg.norm(b),1e-300))


def check_original(export):
 original=export['original'];surface=export['surface']
 k,kd,ka,m,s,c,g=map(lambda key:dense(original[key]),['K','K_D','K_air','M','S','C','G'])
 f=np.array(original['f']);g0=original['G0'];gamma=surface['interface_indices'];air=surface['air_indices'];interior=surface['conductor_indices']
 assert sorted(gamma+air+interior)==list(range(len(k)))
 ext=np.array(surface['extension']);S=np.array(surface['S']);W=np.array(surface['W']);v=np.array(surface['steklov']);values=np.array(surface['eigenvalues']);null=np.array(surface['trace_null'])
 assert relative(k,kd+ka)<1e-10
 assert relative(ka[np.ix_(air,air)]@ext,-ka[np.ix_(air,gamma)])<1e-9
 rebuilt=ka[np.ix_(gamma,gamma)]+ka[np.ix_(gamma,air)]@ext
 assert relative(S,rebuilt)<1e-10 and relative(S,S.T)<1e-10
 assert np.linalg.eigvalsh(S)[0]>-1e-10*np.linalg.norm(S,2)
 assert np.linalg.norm(S@null)<1e-9*np.linalg.norm(S)
 assert relative(v.T@W@v,np.eye(len(values)))<1e-9
 assert relative(v.T@S@v,np.diag(values))<1e-9
 previous=None;worst=0.
 for case in export['cases']:
  q=np.array(case['basis']);assert np.linalg.norm(g.T@q)<1e-9*max(np.linalg.norm(g)*np.linalg.norm(q),1e-300)
  assert relative(q.T@q,np.eye(q.shape[1]))<1e-9
  if previous is not None:assert np.linalg.norm(previous-q@(q.T@previous))<1e-8*np.linalg.norm(previous)
  previous=q;K=q.T@k@q;M=q.T@m@q;C=c@q;F=q.T@f
  assert np.linalg.eigvalsh(K)[0]>0
  for run in case['runs']:
   shift=run['s0'];r=np.array(run['R_phys']);l=np.array(run['L_phys']);b=np.array(run['port'])
   assert np.linalg.eigvalsh(r).min()>0 and np.linalg.eigvalsh(l).min()>0
   assert run['electric_orth']<1e-8 and run['magnetic_orth']<1e-8
   assert min(run['Rhat']+run['L'])>0
   es=np.array(run['electric_modes']);aas=np.array(run['magnetic_potentials'])
   def loss(e,v):
    a,evec,ep=e[0],e[1:1+len(k)],e[1+len(k):]
    b,vvec,vp=v[0],v[1:1+len(k)],v[1+len(k):]
    return a*b*g0+a*(f@vvec)+b*(f@evec)+evec@m@vvec-evec@c.T@vp-ep@c@vvec+ep@s@vp
   def response(e):
    v=e[1:1+len(k)];p=e[1+len(k):]
    forcing=e[0]*f+m@v-c.T@p;scalar=c@v-s@p
    out=e.copy()
    if shift==0:
     out[1+len(k):]+=np.linalg.solve(s,scalar)
     vout=out[1:1+len(k)];pout=out[1+len(k):]
     magnetic=q@np.linalg.solve(K,q.T@(out[0]*f+m@vout-c.T@pout))
    else:
     system=np.block([[K+shift*M,C.T],[C,s/shift]])
     x=np.linalg.solve(system,np.r_[q.T@forcing,scalar/shift]);magnetic=q@x[:len(F)]
     out[1:1+len(k)]-=shift*magnetic;out[1+len(k):]+=x[len(F):]
    return out,magnetic
   initial=np.zeros(1+len(k)+len(s));initial[0]=1
   predicted,magnetic=response(initial);cumulative=np.zeros(len(k))
   for stage,(electric,a) in enumerate(zip(es,aas)):
    # Compare physical load/field, independent of scalar-gradient coordinates.
    force=lambda e:e[0]*f+m@e[1:1+len(k)]-c.T@e[1+len(k):]
    assert relative(force(predicted),force(electric))<1e-7
    assert relative(k@magnetic,k@a)<1e-7
    rh=1/(loss(electric,electric)+shift*(a@k@a))
    assert abs(rh/run['Rhat'][stage]-1)<1e-8
    cumulative+=rh*a;ell=cumulative@k@cumulative
    assert abs(ell/run['L'][stage]-1)<1e-8
    parts=[cumulative@kd@cumulative,cumulative@ka@cumulative]
    assert relative(np.array(parts),np.array(run['energy_parts'][stage]))<1e-8
    if stage+1<len(es):
     forcing=np.zeros_like(initial);forcing[1:1+len(k)]=cumulative
     correction,ac=response(forcing)
     predicted=electric-correction/ell;magnetic=a-ac/ell
   # Independent unit-current constrained original pencil, no producer helper.
   n=len(F);p=len(s);zero=np.zeros((p,1));h=np.block([[K+shift*M,C.T,-F[:,None]],[shift*C,s,zero],[-shift*F[None,:],zero.T,np.array([[g0]])]])
   derivative=np.zeros_like(h);derivative[:n,:n]=M;derivative[n:n+p,:n]=C;derivative[-1,:n]=-F
   rhs=np.zeros(n+p+1);rhs[-1]=1
   tau=4e-7*math.pi*1e6*.01**2
   coefficients=constrained_coefficients(h,derivative,rhs,9,tau)
   stored=np.array(run['full_Z_coefficients']);scale=np.maximum(abs(coefficients),1e-14*np.max(abs(coefficients)))
   assert np.max(abs(stored-coefficients)/scale)<1e-7
   matched=np.array(run['galerkin_Z_coefficients']);assert np.max(abs(matched[:8]-coefficients[:8])/scale[:8])<1e-7
   assert abs(run['Rhat'][0]/coefficients[0]-1)<1e-8
   for row in run['frequency']:
    omega=2j*math.pi*row['hz'];system=np.block([[K+omega*M,C.T],[omega*C,s]])
    solution=np.linalg.solve(system,np.r_[F,np.zeros(p)])
    z=1/(g0-omega*F@solution[:n]);error=abs(z/complex(*row['Z'])-1);assert error<1e-8;worst=max(worst,float(error))
    assert row['power_defect']<1e-8 and row['ladder_galerkin_error']<1e-8
 # Zero DtN on an UNRESTRICTED trace introduces a physical zero-cost channel.
 q=np.array(export['cases'][-1]['basis']);bad=q.T@kd@q;eigenvectors=np.linalg.eigh(bad)
 threshold=1e-10*eigenvectors[0][-1];nullbad=eigenvectors[1][:,eigenvectors[0]<threshold]
 coupling=float(np.linalg.norm(nullbad.T@(q.T@f))/np.linalg.norm(q.T@f))
 assert nullbad.shape[1]>0 and coupling>1e-3
 return dict(worst_full_response_reconstruction=worst,zero_DtN_nullity=nullbad.shape[1],zero_DtN_source_coupling=coupling)


def main():
 data=json.loads((ROOT/'docs/data/surface_air.json').read_text());assert data['complete']
 for name,digest in data['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 assert {c['tag'] for c in data['cases']}=={'notched_p1_base','coax_p1_base','notched_p2_base'}
 for cohort in data['cases']:
  control=cohort['exact_gauge_schur_control'];assert max(control[k] for k in ['original_residual','magnetic_field_relative','compliance_relative'])<1e-8
  assert len(control['mixed_controls'])==5
  for row in control['mixed_controls']:
   assert max(row[k] for k in ['port_relative','magnetic_field_relative','original_residual','continuity_residual'])<1e-8
  full=np.array(cohort['full_fixed_source_compliance']);previous=np.zeros_like(full)
  for case in cohort['cases']:
   val=np.array(case['fixed_source_compliance']);assert np.min(val-previous)>-1e-9*max(full) and np.max(val-full)<1e-9*max(full);previous=val
   for run in case['runs']:
    assert np.linalg.eigvalsh(run['R_phys']).min()>0 and np.linalg.eigvalsh(run['L_phys']).min()>0
    assert run['electric_orth']<1e-8 and run['magnetic_orth']<1e-8 and max(run['contact_per_coefficient'])<1e-8
    for row in run['frequency']:assert row['power_defect']<1e-8 and row['ladder_galerkin_error']<1e-8
  final=cohort['cases'][-1];assert final['K']==final['positive_rank']
  assert max(row['surface_response_error'] for run in final['runs'] for row in run['frequency'])<1e-8
  assert max(v for run in final['runs'] for key in ['Rhat','L'] for v in run['element_relative_to_full'][key])<1e-8
  assert np.max(abs(np.array(final['fixed_source_compliance'])/full-1))<1e-8
  assert max(v for run in final['runs'] for row in run['frequency'] for v in row['energy_relative_to_full'])<1e-8
  if cohort['tag']=='coax_p1_base':assert max(row['surface_response_error'] for run in cohort['cases'][0]['runs'] for row in run['frequency'])<.01
 export=json.loads((ROOT/'docs/data/surface_air_matrices.json').read_text());assert export['complete'] and export['source_sha256']==data['source_sha256']
 print('PASS physical surface CLN:',check_original(export))

if __name__=='__main__':main()
