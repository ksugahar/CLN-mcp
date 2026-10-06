"""Validate stored research results, source identities and terminal reconstructions."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'docs/data'
def read(name):return json.loads((DATA/(name+'.json')).read_text(encoding='utf-8-sig'))
h=read('higher_stages');assert len(h['cases'])==24
w=read('round_wire_three_stages');assert all(w['checks'].values())
assert w['R_normalized']==h['oracle_R_normalized'] and w['L_normalized']==h['oracle_L_normalized']
for name,digest in h['source_sha256_lf'].items():assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()==digest,name
failures=[(v['formulation'],v['order'],v['bonus_intorder'],v['maxh']) for v in h['cases'] if v['status']!='pass']
assert failures==[('CLN_T_Omega',1,4,.003)],failures
quad8=[v for v in h['cases'] if (v['formulation'],v['order'],v['bonus_intorder'],v['maxh'])==('CLN_T_Omega',1,8,.003)][0]
assert quad8['status']=='pass'  # means solve completed, not coefficient accuracy
for formulation in ['CLN_APhi','CLN_T_Omega','CLN_AT']:
    p_errors=[]
    for order in [1,2,3,4]:
        v=[v for v in h['cases'] if v['formulation']==formulation and v['order']==order and v['maxh']==.003 and v['bonus_intorder']==8][0]
        assert v['status']=='pass'
        p_errors.append(max(v['R_relative_error'][-1],v['L_relative_error'][-1]))
    assert all(b<a for a,b in zip(p_errors[:-1],p_errors[1:])),(formulation,p_errors)
    fine=[v for v in h['cases'] if v['formulation']==formulation and v['order']==3 and v['maxh']==.0015][0]
    assert fine['status']=='pass' and max(fine['R_relative_error']+fine['L_relative_error'])<.002
    for order in [2,3]:
        a=[v for v in h['cases'] if v['formulation']==formulation and v['order']==order and v['maxh']==.003 and v['bonus_intorder']==8][0]
        b=[v for v in h['cases'] if v['formulation']==formulation and v['order']==order and v['maxh']==.0015][0]
        assert b['R_relative_error'][-1]<a['R_relative_error'][-1] and b['L_relative_error'][-1]<a['L_relative_error'][-1]
for size in ['coarse','fine']:
    d=read('information_'+size)
    for name,digest in d['source_sha256_lf'].items():assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()==digest,name
    freq=np.asarray(d['frequency_hz']);u=2j*np.pi*freq*d['tau'];ul=2j*np.pi*np.asarray(d['low_frequency_hz'])*d['tau']
    ref=np.asarray(d['reference_real'])+1j*np.asarray(d['reference_imag']);lowref=np.asarray(d['low_reference_real'])+1j*np.asarray(d['low_reference_imag'])
    train=np.asarray(d['training_frequency_hz']);hold=np.all(abs(freq[:,None]/train[None,:]-1)>1e-9,axis=1)&(freq>=1000)
    assert hashlib.sha256((ROOT/d['baseline_file']).read_bytes()).hexdigest()==d['baseline_sha256']
    for name,v in d['runs'].items():
        assert v.get('status')!='optimizer_failure',name
        assert v['L_tail']==0 and v['DC_offset']>=0,name
        p=np.asarray(v['poles']);r=np.asarray(v['residues']);assert len(p)==len(r)==v['rank'] and np.all(p>0) and np.all(r>=0)
        fun=lambda x:v['DC_offset']+v['L_tail']*x+x*np.sum(r[None,:]/(x[:,None]+p[None,:]),axis=1)
        e=abs(fun(u)/ref-1);el=abs(fun(ul)/lowref-1)
        assert np.allclose(e,v['relative_error'],rtol=1e-5,atol=1e-12),name
        assert np.allclose(el,v['low_relative_error'],rtol=1e-5,atol=1e-12),name
        assert np.isclose(e[hold].max(),v['holdout_high_band_max'],rtol=1e-6,atol=1e-12)
        if name.startswith('Foster DC constrained'):assert v['DC_offset']==0 and abs(np.sum(r/p)+v['L_tail']-1)<1e-12
    d=read('nonlinear_matched_'+size)
    for name,digest in d['source_sha256_lf'].items():assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()==digest,name
    for name,v in d['runs'].items():
        assert v.get('status')!='failure',name
        assert v['rank']==int(name.split(' amplitude ')[0].split()[-1]),name
        assert all(np.isfinite(v[k]) and v[k]>=0 for k in ['rod_linkage_relative_error','joule_relative_error','field_energy_rms'])
    assert all(max(v.values())<1e-5 for v in d['same_basis_modal_checks'].values())
mesh=read('mesh_reference');assert len(mesh['meshes'])==3
for filename in ['mesh_reference','mesh_reference_order2']:
    v=read(filename);assert v['complete'] and len(v['meshes'])==3
    assert v['runtime']['ngsolve'] and v['runtime']['netgen']
    assert v['frequency_hz']==[1000.,10000.,100000.,1000000.]
    for name,digest in v['source_sha256_lf'].items():assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()==digest,name
    assert all(row['original_residual_max']<1e-10 and row['penalty_invariance_max']<1e-10 for row in v['meshes'])
p2=read('mesh_reference_order2');assert p2['FE_order']==2
assert [v['maxh'] for v in p2['meshes']]==[.004,.003,.0025]
def susceptibility(v):return v['static_H']*(np.asarray(v['reference_real'])+1j*np.asarray(v['reference_imag']))
pair=[max(abs(susceptibility(a)/susceptibility(b)-1)) for a,b in zip(p2['meshes'][:-1],p2['meshes'][1:])]
assert pair[1]<pair[0] and pair[1]<.0015  # sampled intermesh changes only
assert 1.25<p2['meshes'][-1]['ne']/p2['meshes'][-2]['ne']<1.27
assert 2.98<mesh['meshes'][-1]['ne']/mesh['meshes'][-2]['ne']<3.00
cross=max(abs(susceptibility(mesh['meshes'][-1])/susceptibility(p2['meshes'][-1])-1))
assert .017<cross<.019  # report the cross-order gap, not a continuum certificate
fine=read('nonlinear_matched_fine')
bulk=fine['runs']['protected bulk2 baseline 2 amplitude 1.0'];enriched=fine['runs']['protected bulk2 + auxiliary surface 9 amplitude 1.0']
assert bulk['rank']==2 and enriched['rank']==9
assert enriched['rod_linkage_relative_error']>bulk['rod_linkage_relative_error']
assert enriched['joule_relative_error']<bulk['joule_relative_error']

assert mesh['frequency_hz']==[1000.,10000.,100000.,1000000.]
baseline=read('same_excitation_3d_fine');coarse=mesh['meshes'][0]
a=np.asarray(coarse['reference_real'])+1j*np.asarray(coarse['reference_imag'])
b=(np.asarray(baseline['reference_real'])+1j*np.asarray(baseline['reference_imag']))[[75,150,225,300]]
assert max(abs(a/b-1))<1e-8 and abs(coarse['static_H']/baseline['mesh']['static_H']-1)<1e-8
assert all(v['original_residual_max']<1e-6 and v['penalty_invariance_max']<1e-6 for v in mesh['meshes'])
m=read('research_sources')
for name,digest in m['source_sha256_lf'].items():assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()==digest,name
print('PASS higher-stage convergence, matched-state fits, nonlinear retained ranks and same-basis diagonalization')
