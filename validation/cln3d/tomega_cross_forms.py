import sys,json,math,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import ngsolve as ng
from general_shape_air import Model,source_identity,complex_pair
from general_shape_air_t import CurrentModel,run_current


def main(directory,output):
 p=Path(directory);sources=source_identity();sources['validation/cln3d/tomega_cross_forms.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();cases=[]
 for name,h in [('c1-notched-p0-final',.012),('c1-notched-h8-p0',.008),('c1-notched-h6-p0',.006),('c1-notched-h8-p1',.008),('c1-notched-h6-p1',.006),('c1-notched-h4-p1',.004)]:
  path=p/(name+'.json');data=json.loads(path.read_text());base=Model(h,1,airh=h,geometry='notched')
  assert base.metadata['mesh_sha256']==data['mesh']['mesh_sha256']
  t=CurrentModel(base,'A-T');runs=[run_current(t,s0,4) for s0 in [0.,2*math.pi*1e4]]
  gaps=[]
  for target,control in zip(data['runs'],runs):
   import numpy as np
   gaps.append({key:abs(np.array(target[key])/control[key]-1).tolist() for key in ['Rhat','L']})
   for row,other in zip(target['frequency'],control['frequency']):
    row['A_T_Z']=other['full_Z'];row['T_Omega_A_T_gap']=float(abs(complex(*row['Z'])/complex(*other['full_Z'])-1))
  cases.append(dict(tag=name,mesh=base.metadata,A_T_runs=runs,T_Omega_runs=data['runs'],element_gaps=gaps,
   source_case_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  print('PASS cross-form',name,flush=True)
 assert sources['validation/cln3d/tomega_cross_forms.py']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 assert all(source_identity()[key]==value for key,value in sources.items() if key!='validation/cln3d/tomega_cross_forms.py')
 Path(output).write_text(json.dumps(dict(complete=True,cases=cases,source_sha256=sources),indent=2)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--directory',required=True);a.add_argument('--output',required=True);args=a.parse_args()
 ng.SetNumThreads(4)
 with ng.TaskManager():main(args.directory,args.output)
