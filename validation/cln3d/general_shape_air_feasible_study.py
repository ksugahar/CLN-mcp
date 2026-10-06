"""Reproduce the air/enclosure cohorts; native NGSolve, no commercial data."""
import argparse,json,math,platform,time,hashlib
from pathlib import Path
import ngsolve as ng,netgen
import numpy as np
from general_shape_air import compute_case,source_identity
from general_shape_air_controls import stage_a_control

COHORTS=[
 ('coax_p1_base','coax',.006,.008,1,.015,None,0),
 ('coax_p1_D1','coax',.006,.008,1,.015,'conductor',1),
 ('coax_p2_base','coax',.006,.008,2,.015,None,0),
 ('coax_p2_D1','coax',.006,.008,2,.015,'conductor',1),
 ('notched_p1_base','notched',.006,.008,1,.020,None,0),
 ('notched_p1_D1','notched',.006,.008,1,.020,'conductor',1),
 ('notched_p1_D2','notched',.006,.008,1,.020,'conductor',2),
 ('notched_p1_A1','notched',.006,.008,1,.020,'air',1),
 ('notched_p1_A2','notched',.006,.008,1,.020,'air',2),
 ('notched_p2_base','notched',.006,.008,2,.020,None,0),
 ('notched_b15','notched',.006,.008,1,.015,None,0),
 ('notched_b30','notched',.006,.008,1,.030,None,0)]

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--export');p.add_argument('--resume');args=p.parse_args()
 ng.SetNumThreads(4);producer_sources=source_identity()
 own_source='validation/cln3d/general_shape_air_feasible_study.py'
 identities=dict(producer_sources);identities[own_source]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 cached={}
 if args.resume:
  previous=json.loads(Path(args.resume).read_text())
  assert previous['source_sha256'] in (producer_sources,identities),'Checkpoint producer source mismatch'
  cached={c['tag']:c for c in previous['cases']}
 result=dict(complete=False,source_sha256=identities,cases=[],
 runtime=dict(python=platform.python_version(),ngsolve=ng.__version__,netgen=netgen.__version__,float64_epsilon=float(np.finfo(np.float64).eps),longdouble_epsilon=float(np.finfo(np.longdouble).eps)),
 solver='Explicit equilibrated sparse LU with three residual corrections of the same factors; NumPy longdouble is platform-dependent (float64 on Windows); no reorthogonalization',
 cached_native_cohorts=[],
 scope='Conductor in a perfectly conducting enclosure with ideal source-break port; sigma_air=0 exactly; no capacitance or return-wall loss',
 limitations='Same-mesh reduction error is separate from FE/oracle error; skin-depth screens use actual conductor maximum chord, not air size. No open-space or general high-frequency accuracy claim. A-T shares the magnetic operator. The refined quadratic coax cohort is not included in this feasible grid; coax h/p controls are the other three cohorts. T-Omega excluded.')
 for tag,geometry,h,airh,order,b,region,steps in COHORTS:
  if tag=='coax_p2_D1':continue
  print('START',tag,flush=True)
  if tag in cached and (tag!='notched_p1_base' or not args.export or Path(args.export).exists()):
   case=cached[tag];m=case['mesh']
   assert (m['geometry'],m['maxh'],m['airh'],m['order'],m['enclosure'],m['refine_region'],m['refine_steps'])==(geometry,h,airh,order,b,region,steps)
   assert previous['runtime']==result['runtime'],'Checkpoint runtime mismatch'
   result['cached_native_cohorts'].append(tag)
  else:
   case=compute_case(h,airh,order,b,geometry,export_path=args.export if tag=='notched_p1_base' else None,refine_region=region,refine_steps=steps)
  case['tag']=tag;result['cases'].append(case)
  Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
  print('PASS',tag,case['mesh']['ne'],case['seconds'],flush=True)
 with ng.TaskManager():result['stage_A_control']=stage_a_control()
 # Same coarse coax, change ONLY the shared explicit quadrature rule.
 q6=next(c for c in result['cases'] if c['tag']=='coax_p1_base');q8=compute_case(.006,.008,1,.015,'coax',with_at=False,bonus=8)
 result['quadrature_check']=dict(scope='coarse coax A-phi, explicit TET rule orders16/18',
  max_element_relative=float(max(np.max(abs(np.array(a[k])/b[k]-1)) for a,b in zip(q6['runs'],q8['runs']) for k in ['Rhat','L'])))
 assert result['quadrature_check']['max_element_relative']<1e-8
 assert source_identity()==producer_sources,'Producer source changed during study'
 assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==identities[own_source]
 if args.export:
  exported=json.loads(Path(args.export).read_text());assert exported['source_sha256'] in (producer_sources,identities)
  exported['source_sha256']=identities
  Path(args.export).write_text(json.dumps(exported,indent=2)+'\n',encoding='utf-8',newline='\n')
 result['complete']=True;Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
 print('PASS complete study',len(result['cases']),flush=True)

if __name__=='__main__':main()
