"""Reproduce physical surface-trace CLN against matched meshed-air models."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
import ngsolve as ng
from general_shape_air import Model,csr,source_identity,complex_pair
from surface_air import AirTrace
from surface_cln import TraceModel,run

ROOT=Path(__file__).resolve().parents[2]
COHORTS=[('notched_p1_base','notched',1),('coax_p1_base','coax',1),('notched_p2_base','notched',2)]

def identities():
 result=source_identity()
 for name in ['surface_air.py','surface_cln.py','surface_air_study.py']:
  p=Path(__file__).with_name(name);result['validation/cln3d/'+name]=hashlib.sha256(p.read_bytes()).hexdigest()
 return result


def production_schur(base,surface):
 # Exact elimination of the ORIGINAL gauge block, with reconstructed air.
 keep=np.r_[surface.d,surface.gamma];air=surface.air;k=base.Kg
 aa=k[air][:,air].toarray();ag=k[air][:,keep].toarray();lu=la.cho_factor(aa)
 extension=-la.cho_solve(lu,ag);schur=k[keep][:,keep].toarray()+ag.T@extension
 initial=np.zeros(1+base.na+base.np);initial[0]=1;f=base.load(initial)
 rhs=f[keep]-ag.T@la.cho_solve(lu,f[air]);x=la.solve(schur,rhs,assume_a='pos');a=np.zeros(base.na)
 a[keep]=x;a[air]=extension@x+la.cho_solve(lu,f[air]);full=base.magnetic(initial)
 result=dict(original_residual=float(la.norm(k@a-f)/la.norm(f)),
     magnetic_field_relative=float(np.sqrt(abs((a-full)@(base.K@(a-full))/(full@(base.K@full))))),
     compliance_relative=float(abs(f@a/(f@full)-1)),air_load_norm=float(la.norm(f[air])),mixed_controls=[])
 assert np.linalg.norm(base.M[air].data)==0 and np.linalg.norm(base.C[:,air].data)==0
 # Continuity is condensed exactly too, using the original conductor S.
 cp=base.C[:,keep].toarray();lift=base.slu.solve(cp)
 conductivity=base.M[keep][:,keep].toarray()-cp.T@lift
 for shift in [2*math.pi*1e4]+[2j*math.pi*hz for hz in [1e3,1e4,1e5,1e6]]:
  matrix=schur+shift*conductivity;scale=1/np.sqrt(abs(matrix.diagonal()))
  factors=la.lu_factor(scale[:,None]*matrix*scale[None,:]);x=scale*la.lu_solve(factors,scale*rhs)
  for _ in range(3):x+=scale*la.lu_solve(factors,scale*(rhs-matrix@x))
  reconstructed=np.zeros(base.na,dtype=x.dtype);reconstructed[keep]=x;reconstructed[air]=extension@x
  phi=-shift*(lift@x);current=base.G0-shift*(base.f@reconstructed);Z=1/current
  ef,af=base.response(initial,shift);reference=1/base.port(ef)
  original=(base.K+shift*base.M)@reconstructed+base.C.T@phi-base.f
  scalar=shift*(base.C@reconstructed)+base.S@phi
  magnetic=float(np.sqrt(abs((reconstructed-af).conj()@(base.K@(reconstructed-af))/(af.conj()@(base.K@af)))))
  entry=dict(s=complex_pair(shift),Z=complex_pair(Z),port_relative=float(abs(Z/reference-1)),magnetic_field_relative=magnetic,
    original_residual=float(la.norm(original)/la.norm(base.f)),continuity_residual=float(la.norm(scalar)/max(la.norm(shift*(base.C@reconstructed)),1e-300)))
  assert max(entry[key] for key in ['port_relative','magnetic_field_relative','original_residual','continuity_residual'])<1e-8,entry
  result['mixed_controls'].append(entry)
 return result


def compute(tag,geometry,order):
 start=time.time();b=Model(.006,order,airh=.008,geometry=geometry,enclosure=.015 if geometry=='coax' else .020)
 a,w=b.va.TnT();mass=ng.BilinearForm(b.va);mass+=a.Trace()*w.Trace()*ng.ds('surface');mass.Assemble()
 surface=AirTrace(b,csr(mass)[b.fa][:,b.fa].toarray())
 reference=json.loads((ROOT/'docs/data/general_3d_air.json').read_text())
 matched=next(c for c in reference['cases'] if c['tag']==tag)
 assert matched['mesh']['mesh_sha256']==b.metadata['mesh_sha256'],'Stage-B mesh identity differs'
 control=production_schur(b,surface)
 assert max(control['original_residual'],control['magnetic_field_relative'],control['compliance_relative'])<1e-8
 initial=np.zeros(1+b.na+b.np);initial[0]=1;f=b.load(initial);full=b.magnetic(initial);compliance=float(f@full)
 # Fixed source directions beyond the protected DC source test the variational inequality.
 sources=[f]
 for hz in [1e4,1e5]:
  e,_=b.response(initial,2j*math.pi*hz)
  sources.extend([b.load(e.real),b.load(e.imag)])
 full_compliance=np.array([v@b.klu.solve(v) for v in sources])
 counts=sorted(set([0,1,2,4,8,16,32,64,128,len(surface.eigenvalues)]))
 cases=[];previous=np.zeros(len(sources))
 for count in counts:
  if count>len(surface.eigenvalues):continue
  q=surface.basis(count);model=TraceModel(b,q)
  values=np.array([v@(q@la.cho_solve(model.chol,q.T@v)) for v in sources])
  assert np.min(values-previous)>-1e-9*max(full_compliance)
  assert np.max(values-full_compliance)<1e-9*max(full_compliance);previous=values
  runs=[run(model,s) for s in [0.,2*math.pi*1e4]]
  for result,ref in zip(runs,matched['runs']):
   result['element_relative_to_full']={key:(abs(np.array(result[key])/ref[key]-1)).tolist() for key in ['Rhat','L']}
   for row,old in zip(result['frequency'],ref['frequency']):
    z=complex(*row['Z']);row['same_mesh_full_Z']=old['full_Z'];row['surface_response_error']=float(abs(z/complex(*old['full_Z'])-1))
    row['energy_relative_to_full']=(abs(np.array(row['magnetic_regions'])/old['magnetic_regions']-1)).tolist()
   if tag!='notched_p1_base':
    result.pop('electric_modes');result.pop('magnetic_potentials')
  c=dict(K=count,positive_rank=len(surface.eigenvalues),protected_port_rank=surface.port_trace.shape[1],trace_kernel_rank=surface.trace_null.shape[1],
     magnetic_dimension=q.shape[1],fixed_source_compliance=values.tolist(),runs=runs)
  if tag=='notched_p1_base':c['basis']=q.tolist()
  cases.append(c);print('PASS',tag,'K',count,'max Z surface error',max(row['surface_response_error'] for r in runs for row in r['frequency']),flush=True)
 full_case=cases[-1]
 assert max(row['surface_response_error'] for r in full_case['runs'] for row in r['frequency'])<1e-8
 assert max(v for r in full_case['runs'] for key in ['Rhat','L'] for v in r['element_relative_to_full'][key])<1e-8
 no_protection=[]
 for count in [0,min(4,len(surface.eigenvalues)),len(surface.eigenvalues)]:
  q=surface.basis(count,protect=False);k=q.T@b.K@q;value=float(f@(q@la.solve(k,q.T@f,assume_a='pos')))
  no_protection.append(dict(K=count,DC_compliance=value,relative_gap=float(1-value/compliance)))
 result=dict(tag=tag,mesh=b.metadata,seconds=time.time()-start,exact_gauge_schur_control=control,
    full_fixed_source_compliance=full_compliance.tolist(),full_DC_compliance=compliance,
    unprotected_DC_control=no_protection,cases=cases)
 export=dict(complete=True,original=b.export(),surface=surface.export(),cases=cases) if tag=='notched_p1_base' else None
 return result,export


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--export',required=True);p.add_argument('--tags',nargs='*');args=p.parse_args()
 ng.SetNumThreads(4);before=identities();data=dict(complete=False,source_sha256=before,cases=[],scope='Nested Steklov trace space with physical quotient harmonic extension; fixed-source compliance underestimates full inductance; no fitted surface modes')
 for tag,geometry,order in COHORTS:
  if args.tags and tag not in args.tags:continue
  print('START',tag,flush=True)
  with ng.TaskManager():case,export=compute(tag,geometry,order)
  data['cases'].append(case)
  if export:
   export['source_sha256']=before;Path(args.export).write_text(json.dumps(export,indent=2)+'\n',encoding='utf-8',newline='\n')
  Path(args.output).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
 assert before==identities();data['complete']=True
 Path(args.output).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n');print('PASS complete',flush=True)

if __name__=='__main__':main()
