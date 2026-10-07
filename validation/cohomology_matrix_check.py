"""Independent current/air-cohomology checks from original matrices (NumPy only)."""
import json,math,itertools
from pathlib import Path
import numpy as np
from general_3d_air_matrix_check import dense,series,check_coefficients

TAU=4e-7*math.pi*1e6*.01**2


def relative(a,b):
 return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))


def check(case):
 e=case['export'];r,l,a,g=map(lambda k:dense(e[k]),['raw_R','raw_L','raw_aircurl','raw_gradient'])
 q=np.array(e['Q']);b=np.array(e['raw_port']);R,L,B=map(lambda k:np.array(e[k]),['R','L','port'])
 assert relative(q.T@r@q,R)<1e-9
 assert relative(q.T@l@q,L)<1e-9
 assert relative(q.T@b,B)<1e-9
 assert np.linalg.norm(a@q)/(np.linalg.norm(a)*np.linalg.norm(q))<1e-10
 assert np.linalg.norm(g.T@l@q)/(np.linalg.norm(g)*np.linalg.norm(l)*np.linalg.norm(q))<1e-10
 assert np.linalg.norm(r@g)/(np.linalg.norm(r)*np.linalg.norm(g))<1e-10
 assert np.linalg.eigvalsh(R).min()>0 and np.linalg.eigvalsh(L).min()>0
 # Rebuild the admissible space without importing the producer.
 ev,u=np.linalg.eigh(a);closed=u[:,ev<1e-10*max(ev[-1],1)]
 ug,sg,_=np.linalg.svd(g,full_matrices=False);ug=ug[:,sg>sg[0]*1e-11]
 closed-=ug@np.linalg.solve(ug.T@l@ug,ug.T@l@closed)
 uc,sc,_=np.linalg.svd(closed,full_matrices=False);ind=uc[:,sc>sc[0]*1e-11]
 assert ind.shape[1]==len(B)
 assert relative(q,ind@(ind.T@q))<1e-8
 Ri,Li,bi=ind.T@r@ind,ind.T@l@ind,ind.T@b
 original=e['original_Aphi']
 ka,ma,sa,ca,ga=map(lambda key:dense(original[key]),['K','M','S','C','G'])
 fa=np.array(original['f']);kg=ka+original['gauge_scale']*(ga@ga.T)
 for loop in case['sampled_ampere']:
  H,dl=np.array(loop['H_A_per_m']),np.array(loop['dl_m'])
  circulation=float(np.sum(H*dl))
  assert abs(circulation-loop['circulation_A'])<1e-12
  assert abs(circulation-loop['target_A'])<.005
 for t in case['topology']:
  h,p=np.array(t['generator']),np.array(t['period'])
  vertices=t['air_vertices'];mapping={v:i for i,v in enumerate(vertices)}
  edges=[tuple(edge) for edge in t['air_edges']];edge_index={edge:i for i,edge in enumerate(edges)}
  geometry=original['geometry'];air_index=geometry['materials'].index('air')+1
  faces=set()
  for tetra,region in zip(geometry['tetrahedra'],geometry['region_indices']):
   if region==air_index:
    local=[mapping[v-1] for v in tetra]
    faces.update(tuple(sorted(face)) for face in itertools.combinations(local,3))
  d0=np.zeros((len(edges),len(vertices)))
  for i,(lo,hi) in enumerate(edges):d0[i,lo]=-1;d0[i,hi]=1
  d1=np.zeros((len(faces),len(edges)))
  for row,(aa,bb,cc) in enumerate(sorted(faces)):
   for edge,sign in [((bb,cc),1),((aa,cc),-1),((aa,bb),1)]:d1[row,edge_index[edge]]=sign
  assert np.linalg.norm(d1@d0)==0
  assert len(edges)-np.linalg.matrix_rank(d1)-np.linalg.matrix_rank(d0)==1
  cochain,cycle=np.array(t['cochain']),np.array(t['cycle'])
  assert np.linalg.norm(d1@cochain)<1e-10 and np.linalg.norm(cycle@d0)<1e-10
  assert abs(cycle@cochain-1)<1e-10
  assert t['b1']==1 and abs(p@h-1)<1e-10
  assert relative(p@q,B)<1e-8
  assert t['gradient_representative_relative']<1e-8
  natural=h-ug@np.linalg.solve(ug.T@l@ug,ug.T@l@h)
  assert relative(natural,ind@(ind.T@natural))<1e-8
  # A unit terminal-current lift, not a prescribed scalar-Omega boundary.
  assert abs(b@natural-1)<1e-8
 for run in case['runs']:
  s0=run['s0'];v=np.array(run['current_modes']).T;x=np.array(run['cumulative_modes']).T
  rh=np.array(run['Rhat']);ells=np.array(run['L']);mat=R+s0*L
  j=np.linalg.solve(mat,B);cumulative=np.zeros(len(B))
  mi=Ri+s0*Li;ji=np.linalg.solve(mi,bi);ci=np.zeros(len(bi))
  for k in range(4):
   assert relative(j,v[:,k])<1e-7
   assert relative(ind@ji,q@v[:,k])<1e-7
   value=1/(j@mat@j);assert abs(value/rh[k]-1)<1e-7
   cumulative+=value*j;assert relative(cumulative,x[:,k])<1e-7
   assert abs((cumulative@L@cumulative)/ells[k]-1)<1e-7
   j-=np.linalg.solve(mat,L@cumulative)/ells[k]
   ci+=rh[k]*ji;ji-=np.linalg.solve(mi,Li@ci)/ells[k]
  full=series(Ri,Li,bi,s0,9,TAU)
  projected=series(v.T@R@v,v.T@L@v,v.T@B,s0,9,TAU)
  check_coefficients(run['full_Z_coefficients'],full)
  check_coefficients(run['galerkin_Z_coefficients'],projected)
  check_coefficients(projected[:8],full[:8])
  assert abs(rh[0]/full[0]-1)<1e-8
  for row in run['frequency']:
   s=2j*math.pi*row['hz'];zi=1/(bi@np.linalg.solve(Ri+s*Li,bi));stored=complex(*row['Z'])
   assert abs(zi/stored-1)<1e-8
   block=np.block([[kg+s*ma,ca.T],[ca,sa/s]])
   sol=np.linalg.solve(block,np.r_[fa,np.zeros(len(sa))])
   za=1/(original['G0']-s*(fa@sol[:len(fa)]))
   assert abs(za/complex(*row['Aphi_Z'])-1)<1e-8
   zgal=1/((v.T@B)@np.linalg.solve(v.T@(R+s*L)@v,v.T@B))
   assert abs(zgal/complex(*row['ladder'])-1)<1e-8
   assert abs(abs(zgal/zi-1)-row['reduction_error'])<1e-8
   assert abs(sum(row['magnetic_regions'])/row['magnetic_energy']-1)<1e-9
 assert case['omitted_period_port_norm']<1e-12
 return dict(current_dimension=len(B),independent_space_gap=relative(q,ind@(ind.T@q)))


if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('file');args=parser.parse_args()
 print('PASS',check(json.loads(Path(args.file).read_text())))
