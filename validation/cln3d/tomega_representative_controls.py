"""Non-trivial cocycle changes before separately assembled native Omega solves.

Small straight-sided notched mesh only. Existing vertex-permutation cohorts
are reproducibility checks, not independent cut changes.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import ngsolve as ng
from general_shape_air import Model,MU,SIGMA,packed
from tomega_air import TOmega
from current_cln import physical_recursion
from tomega_air_study import identities


def compute():
 base=Model(.012,1,airh=.012,geometry='notched');model=TOmega(base,order=0)
 topology,changes=model.topology();h=np.array(topology[0]['generator']);p=np.array(topology[0]['period'])
 gm,scalar=model.space.CreateGradient();G=sp.csr_matrix(gm.CSR(),shape=(gm.height,gm.width)).copy()
 eta=np.sin(np.arange(scalar.ndof)*.37);eta*=la.norm(h)/la.norm(G@eta)
 generators=[h,h+G@eta];difference=la.norm(generators[1]-generators[0])/la.norm(h);assert difference>.5
 free=np.array(list(scalar.FreeDofs()),bool);free[0]=False
 # Reassemble the physical field operators without using the old projector.
 u,v=model.space.TnT();rf=ng.BilinearForm(model.space);rf+=ng.curl(u)*ng.curl(v)/SIGMA*base.dxD;rf.Assemble()
 lf=ng.BilinearForm(model.space);lf+=MU*u*v*base.dx;lf.Assemble()
 bf=ng.LinearForm(model.space);bf+=base.e0*ng.curl(model.space.TestFunction())*base.dxD;bf.Assemble()
 R=sp.csr_matrix(rf.mat.CSR(),shape=(model.space.ndof,model.space.ndof)).copy().toarray()
 L=sp.csr_matrix(lf.mat.CSR(),shape=(model.space.ndof,model.space.ndof)).copy().toarray();b=bf.vec.FV().NumPy().copy()
 zero=model.Q@la.null_space(model.b[None,:]);fields=[];records=[]
 for generator in generators:
  seed=ng.GridFunction(model.space);seed.vec.FV().NumPy()[:]=generator
  a,c=scalar.TnT();stiff=ng.BilinearForm(scalar);stiff+=MU*ng.grad(a)*ng.grad(c)*base.dx;stiff.Assemble()
  rhs=ng.LinearForm(scalar);rhs+=-MU*seed*ng.grad(scalar.TestFunction())*base.dx;rhs.Assemble()
  S=sp.csr_matrix(stiff.mat.CSR(),shape=(scalar.ndof,scalar.ndof)).copy()
  phi=np.zeros(scalar.ndof);phi[free]=sla.spsolve(S[free][:,free].tocsc(),rhs.vec.FV().NumPy().copy()[free])
  natural=generator+G@phi;fields.append(natural)
  basis=np.column_stack([natural,zero]);r=basis.T@R@basis;l=basis.T@L@basis;port=basis.T@b
  assert abs(port[0]-1)<1e-10 and la.norm(port[1:])<1e-10
  runs=[physical_recursion(r,l,port,s0) for s0 in [0.,2*math.pi*1e4]]
  records.append(dict(generator=generator.tolist(),Omega=phi.tolist(),basis=basis.tolist(),runs=runs,
   S=packed(S),rhs=rhs.vec.FV().NumPy().copy().tolist()))
 field_gap=la.norm(fields[1]-fields[0])/la.norm(fields[0]);assert field_gap<1e-8
 gaps=[]
 for left,right in zip(records[0]['runs'],records[1]['runs']):
  row={key:float(max(abs(np.array(left[key])/right[key]-1))) for key in ['Rhat','L']};assert max(row.values())<1e-8;gaps.append(row)
 scalar_difference=records[1]['Omega']-np.array(records[0]['Omega'])+eta-eta[0]
 assert la.norm(scalar_difference)/la.norm(eta)<1e-8
 return dict(complete=True,scope='one straight-sided notched 118-tetrahedron cohort; independently assembled native scalar equations',
  mesh=base.metadata,generator_relative_difference=float(difference),natural_field_relative=float(field_gap),element_gaps=gaps,
  original=dict(R=R.tolist(),L=L.tolist(),b=b.tolist(),G=packed(G),period=p.tolist(),eta=eta.tolist(),free_scalar=free.tolist()),records=records)


if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
 sources=identities();sources['validation/cln3d/tomega_representative_controls.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();ng.SetNumThreads(4)
 with ng.TaskManager():result=compute()
 assert sources['validation/cln3d/tomega_representative_controls.py']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 assert all(identities()[key]==value for key,value in sources.items() if key!='validation/cln3d/tomega_representative_controls.py')
 result['source_sha256']=sources
 Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n');print('PASS genuinely different cocycle',result['generator_relative_difference'],result['natural_field_relative'],result['element_gaps'])
