"""Validate air cohomology evidence; independent small matrices, stored fine cohorts."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from cohomology_matrix_check import check
from general_3d_air_matrix_check import series,check_coefficients,dense
ROOT=Path(__file__).resolve().parents[1]


def main():
 d=json.loads((ROOT/'docs/data/tomega_air.json').read_text());assert d['complete']
 matrix=ROOT/'docs/data/tomega_air_matrices.json'
 assert hashlib.sha256(matrix.read_bytes()).hexdigest()==d['matrix_sha256']
 for name,digest in d['source_sha256'].items():
  if name=='installed/radia/cohomology.py':
   assert digest=='a673df2e7632923611e1b8c4071ae1170e7f4ff82ef1cd5caf5fb67ab9131511'
  else:assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 check(json.loads(matrix.read_text()))
 assert len(d['cases'])==9 and len(d['cross_form'])==6
 for c in d['cases']:
  assert c['mesh']['sigma_air']==0 and c['mesh']['minimum_sampled_jacobian_determinant']>0
  assert c['air_closure']<1e-10 and c['scalar_stationarity']<1e-10
  assert c['omitted_period_port_norm']<1e-12 and len(c['topology'])==2
  for loop in c['sampled_ampere']:
   circulation=float(np.sum(np.array(loop['H_A_per_m'])*loop['dl_m']))
   assert abs(circulation-loop['circulation_A'])<1e-12
   assert abs(circulation-loop['target_A'])<.005
  for t in c['topology']:
   assert t['b1']==1 and t['air_chain_closure']<1e-10 and t['period_port_relative']<1e-8
   assert t['gradient_representative_relative']<1e-8
  for gaps in c['cut_element_gaps']:assert max(gaps.values())<1e-8
  assert [r['s0'] for r in c['runs']]==[0.,2*math.pi*1e4]
  for r in c['runs']:
   assert r['modes']==4 and min(r['Rhat']+r['L'])>0
   assert max(r['electric_orth'],r['magnetic_orth'])<1e-8
   rp,lp,bp=map(np.array,[r['R_phys'],r['L_phys'],r['port']])
   coefficients=series(rp,lp,bp,r['s0'],9,4e-7*math.pi*1e6*.01**2)
   check_coefficients(coefficients,r['galerkin_Z_coefficients'])
   check_coefficients(coefficients[:8],r['full_Z_coefficients'][:8])
   for row in r['frequency']:
    assert row['power_defect']<1e-8 and row['ladder_galerkin_error']<1e-8
    assert min(row['magnetic_regions'])>=0
    assert abs(sum(row['magnetic_regions'])/row['magnetic_energy']-1)<1e-9
    assert abs(abs(complex(*row['Z'])/complex(*row['Aphi_Z'])-1)-row['cross_form_gap'])<1e-9
 # The order-1 notched comparison is genuine increasing DOF evidence.
 tags={c['tag']:c for c in d['cross_form']}
 seq=[tags[f'c1-notched-h{h}-p1'] for h in [8,6,4]]
 assert seq[0]['mesh']['ne']<seq[1]['mesh']['ne']<seq[2]['mesh']['ne']
 for region in ['conductor','air']:
  counts=[c['mesh']['region_element_counts'][region] for c in seq]
  assert counts[0]<counts[1]<counts[2]
 for shift in range(2):
  for frequency in range(4):
   for metric in ['cross_form_gap','T_Omega_A_T_gap']:
    values=[c['T_Omega_runs'][shift]['frequency'][frequency][metric] for c in seq]
    assert values[0]>values[1]>values[2],(metric,values)
  for key in ['Rhat','L']:
   values=[c['element_gaps'][shift][key][1] for c in seq]
   assert values[0]>values[1]>values[2],values
 control=json.loads((ROOT/'docs/data/tomega_representative_controls.json').read_text())
 assert control['complete'] and control['mesh']['ne']==118
 for name,digest in control['source_sha256'].items():
  if not name.startswith('installed/'):assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 original=control['original'];r,l,b,g=map(lambda key:np.array(original[key]) if key!='G' else dense(original[key]),['R','L','b','G'])
 eta=np.array(original['eta']);free=np.array(original['free_scalar'],bool);period=np.array(original['period']);fields=[]
 records=control['records'];h0,h1=[np.array(x['generator']) for x in records]
 assert np.linalg.norm(h1-h0)/np.linalg.norm(h0)>.5
 assert np.linalg.norm(h1-h0-g@eta)<1e-10*np.linalg.norm(h0)
 for item in records:
  h=np.array(item['generator']);S=dense(item['S']);rhs=np.array(item['rhs'])
  assert abs(period@h-1)<1e-10
  assert np.linalg.norm(S-g.T@l@g)<1e-10*np.linalg.norm(S)
  assert np.linalg.norm(rhs+g.T@l@h)<1e-10*np.linalg.norm(rhs)
  phi=np.zeros(g.shape[1]);phi[free]=np.linalg.solve(S[free][:,free],rhs[free])
  assert np.linalg.norm(phi-item['Omega'])<1e-9*np.linalg.norm(phi)
  natural=h+g@phi;fields.append(natural);basis=np.array(item['basis'])
  assert np.linalg.norm(basis[:,0]-natural)<1e-9*np.linalg.norm(natural)
  R,L,B=basis.T@r@basis,basis.T@l@basis,basis.T@b
  for run in item['runs']:
   full=series(R,L,B,run['s0'],9,4e-7*math.pi*1e6*.01**2)
   check_coefficients(full,run['full_Z_coefficients'])
 assert np.linalg.norm(fields[1]-fields[0])/np.linalg.norm(fields[0])<1e-8
 for left,right in zip(records[0]['runs'],records[1]['runs']):
  for key in ['Rhat','L']:assert np.max(abs(np.array(left[key])/right[key]-1))<1e-8
 print('PASS cohomology current periods, original small matrices, eight-element contact and disclosed refinement')


if __name__=='__main__':main()
