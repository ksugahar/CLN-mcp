"""C1 prototype: Radia air generators embedded in the coupled T--Omega pencil."""
import argparse,hashlib,json,math,time,platform
from importlib.metadata import version
import radia.cohomology as rc
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import ngsolve as ng
from general_shape_air import Model,source_identity,packed,complex_pair,SIGMA,MU
from tomega_air import TOmega
from current_cln import physical_recursion


def identities():
 result=source_identity()
 result["installed/radia/cohomology.py"]=hashlib.sha256(Path(rc.__file__).read_bytes()).hexdigest()
 for name in ['air_cohomology.py','air_subcomplex.py','tomega_air.py','current_cln.py','tomega_air_study.py']:
  p=Path(__file__).with_name(name);result['validation/cln3d/'+name]=hashlib.sha256(p.read_bytes()).hexdigest()
 return result


def sampled_loops(model,raw,R,b):
 base=model.base
 j=la.solve(R,b,assume_a='pos');j/=b@j
 field=ng.GridFunction(model.space);field.vec.FV().NumPy()[:]=raw@j
 geometry=base.metadata['geometry']
 settings=[(0.,0.,radius,.0075,True) for radius in [.008,.011,.013]] if geometry=='coax' else [(0.,0.,.016,z,True) for z in [.004,.008,.012]]
 settings.append((.011 if geometry=='coax' else .017,0.,.001,.0075,False))
 rows=[]
 for cx,cy,radius,z,linked in settings:
  theta=(np.arange(1024)+.5)*2*math.pi/1024
  xyz=np.column_stack([cx+radius*np.cos(theta),cy+radius*np.sin(theta),np.full(len(theta),z)])
  dl=np.column_stack([-radius*np.sin(theta),radius*np.cos(theta),np.zeros(len(theta))])*2*math.pi/len(theta)
  values=np.array([list(field(base.mesh(*point))) for point in xyz])
  circulation=float(np.sum(values*dl));target=1. if linked else 0.
  assert abs(circulation-target)<.005,(geometry,circulation,linked)
  rows.append(dict(linked=linked,unit_current=1.,points_m=xyz.tolist(),H_A_per_m=values.tolist(),dl_m=dl.tolist(),circulation_A=circulation,target_A=target,
    quadrature='1024 midpoint samples; geometric field-sampling check, not exact chain pairing'))
 return rows


def compute(geometry,maxh,order):
 start=time.time();base=Model(maxh,1,airh=maxh,enclosure=.015 if geometry=='coax' else .020,geometry=geometry)
 model=TOmega(base,order=order);topology,changes=model.topology()
 raw=model.Q@changes[0];R=changes[0].T@model.R@changes[0];L=changes[0].T@model.L@changes[0];b=changes[0].T@model.b
 assert abs(b[0]-1)<1e-8 and la.norm(b[1:])<1e-8
 runs=[physical_recursion(R,L,b,s0) for s0 in [0.,2*math.pi*1e4]]
 change=changes[1];other=[physical_recursion(change.T@model.R@change,change.T@model.L@change,change.T@model.b,s0) for s0 in [0.,2*math.pi*1e4]]
 cut_gaps=[]
 initial=np.zeros(1+base.na+base.np);initial[0]=1
 for run,alternative in zip(runs,other):
  gaps={key:float(np.max(abs(np.array(run[key])/alternative[key]-1))) for key in ['Rhat','L']}
  assert max(gaps.values())<1e-8,gaps;cut_gaps.append(gaps)
  for row in run['frequency']:
   s=2j*math.pi*row['hz'];ef,_=base.response(initial,s);reference=1/base.port(ef)
   row['Aphi_Z']=complex_pair(reference);row['cross_form_gap']=float(abs(complex(*row['Z'])/reference-1))
  for j in run['current_modes']:
   field=ng.GridFunction(model.space);field.vec.FV().NumPy()[:]=raw@j
   air=float(ng.Integrate(ng.curl(field)**2*base.dxAir,base.mesh));conductor=float(ng.Integrate(ng.curl(field)**2*base.dxD,base.mesh))
   assert air/max(conductor,1e-300)<1e-12
  for row in run['frequency']:
   s=2j*math.pi*row['hz'];j=la.solve(R+s*L,b,assume_a='sym')
   pieces=[]
   for dx in [base.dxD,base.dxAir]:
    total=0.
    for part in [(raw@j).real,(raw@j).imag]:
     field=ng.GridFunction(model.space);field.vec.FV().NumPy()[:]=part
     total+=float(ng.Integrate(MU*field**2*dx,base.mesh))
    pieces.append(total)
   assert abs(sum(pieces)/row['magnetic_energy']-1)<1e-9
   row['magnetic_regions']=pieces
 # Removing the linked air period clamps the entire net terminal-current channel.
 no_period=la.null_space(model.b[None,:]);omitted=float(la.norm(no_period.T@model.b)/la.norm(model.b))
 assert omitted<1e-12
 result=dict(complete=True,seconds=time.time()-start,geometry=geometry,maxh=maxh,H_order=order,mesh=base.metadata,
  raw_H_dofs=model.space.ndof,current_dimension=len(b),sampled_ampere=sampled_loops(model,raw,R,b),air_closure=model.closure,scalar_stationarity=model.scalar_stationarity,
  topology=topology,cut_element_gaps=cut_gaps,omitted_period_port_norm=omitted,runs=runs,
  export=dict(raw_R=packed(sp.csc_matrix(model.raw_R)),raw_L=packed(sp.csc_matrix(model.raw_L)),raw_aircurl=packed(sp.csc_matrix(model.raw_aircurl)),
    raw_gradient=packed(model.raw_gradient.tocsc()),raw_port=model.raw_b.tolist(),Q=raw.tolist(),R=R.tolist(),L=L.tolist(),port=b.tolist(),original_Aphi=base.export()))
 return result


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--geometry',choices=['coax','notched'],default='coax');parser.add_argument('--maxh',type=float,default=.012);parser.add_argument('--order',type=int,default=0);parser.add_argument('--output',required=True);args=parser.parse_args()
 ng.SetNumThreads(4);sources=identities()
 with ng.TaskManager():result=compute(args.geometry,args.maxh,args.order)
 assert sources==identities();result['source_sha256']=sources
 result['runtime']={name:version(name) for name in ['numpy','scipy','ngsolve','netgen-mesher','radia']}
 result['runtime']['python']=platform.python_version()
 Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print('PASS C1 native case',result['seconds'],result['current_dimension'],result['cut_element_gaps'],flush=True)

if __name__=='__main__':main()
