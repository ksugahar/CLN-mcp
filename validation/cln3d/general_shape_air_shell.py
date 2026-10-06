"""Sample the return surface from the interior; independent native field diagnostic."""
import argparse,json,math,hashlib
from pathlib import Path
import numpy as np,ngsolve as ng
from general_shape_air import Model,source_identity,MU,HEIGHT

def sample_return(model,a,current):
    field=ng.curl(model.gf_a(a))/MU;b=model.metadata['enclosure'];eps=1e-3*b
    zs=(np.arange(16)+.5)*HEIGHT/16;values=[]
    for z in zs:
        if model.metadata['geometry']=='coax':
            r=b-eps;ts=(np.arange(512)+.5)*2*math.pi/512
            circulation=sum(np.dot(np.array(field(model.mesh(r*math.cos(t),r*math.sin(t),z))),(-r*math.sin(t),r*math.cos(t),0))*2*math.pi/512 for t in ts)
        else:
            q=b-eps;corners=[(-q,-q),(q,-q),(q,q),(-q,q),(-q,-q)];circulation=0.
            for p,v in zip(corners,corners[1:]):
                for t in (np.arange(256)+.5)/256:
                    x=p[0]+t*(v[0]-p[0]);y=p[1]+t*(v[1]-p[1])
                    circulation+=np.dot(np.array(field(model.mesh(x,y,z))),(v[0]-p[0],v[1]-p[1],0))/256
        values.append(float(-circulation))
    return dict(scope='Near-wall interior sample at a 0.1% radius/half-width offset (inside the quadratic boundary approximation) of K_shell=H cross n; longitudinal return current, averaged over 16 heights. This is a trace approximation, not a meshed finite-conductivity return.',offset_m=eps,
        entering_conductor_current=float(current),return_currents=values,
        relative_mean_balance=float(abs(np.mean(values)/current+1)),relative_max_balance=float(max(abs(np.array(values)/current+1))))

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--study',required=True);args=p.parse_args()
 ng.SetNumThreads(4)
 study=json.loads(Path(args.study).read_text());assert study['complete']
 identities=dict(study['source_sha256']);identities['validation/cln3d/general_shape_air_shell.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 root=Path(__file__).resolve().parents[2]
 for name,digest in identities.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
 tags={c['tag']:c for c in study['cases']};rows=[]
 specifications=[('coax_p2_base',tags['coax_p2_base']['mesh']),('coax_p2_A1_DC',dict(tags['coax_p2_base']['mesh'],refine_region='air',refine_steps=1)),('notched_p1_base',tags['notched_p1_base']['mesh'])]
 for tag,m in specifications:
  with ng.TaskManager():
   model=Model(m['maxh'],m['order'],airh=m['airh'],enclosure=m['enclosure'],geometry=m['geometry'],refine_region=m['refine_region'],refine_steps=m['refine_steps'])
   if tag in tags:assert model.metadata['mesh_sha256']==m['mesh_sha256']
   base=np.zeros(1+model.na+model.np);base[0]=1;e,a=model.response(base,0.)
   row=sample_return(model,a,model.port(e));row['tag']=tag;row['mesh_sha256']=model.metadata['mesh_sha256'];row['mesh']=model.metadata;row['ampere_midplane']=model.circulation(a,model.port(e));row['scope']+=' The air-refined extra cohort is a DC-only field diagnostic, not a dynamic CLN cohort.' if tag=='coax_p2_A1_DC' else '';rows.append(row)
 out=dict(complete=True,source_sha256=identities,cases=rows)
 for name,digest in identities.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
 Path(args.output).write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8',newline='\n')
 print('PASS return-current samples',[(r['tag'],r['relative_mean_balance']) for r in rows])

if __name__=='__main__':main()
