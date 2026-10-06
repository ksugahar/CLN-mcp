"""Check public enclosure evidence and original small matrices; numpy/stdlib only."""
import cmath,hashlib,json,math
from pathlib import Path
import numpy as np
from general_3d_air_matrix_check import check_small,check_run,dense
ROOT=Path(__file__).resolve().parents[1]

def oracle(s,b=.015):
    a=.005;h=.015;mu=4e-7*math.pi;sigma=1e6;z=mu*sigma*a*a*s/4
    t0=t1=1+0j;f0=f1=1+0j
    for k in range(1,180):
        t0*=z/(k*k);t1*=z/(k*(k+1));f0+=t0;f1+=t1
        if max(abs(t0),abs(t1))<1e-16*max(abs(f0),abs(f1)):break
    return h/(sigma*math.pi*a*a)*f0/f1+s*mu*h*math.log(b/a)/(2*math.pi)

def main():
    data=json.loads((ROOT/'docs/data/general_3d_air.json').read_text());assert data['complete']
    for name,digest in data['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    environment=json.loads((ROOT/'docs/data/general_3d_air_environment.json').read_text());assert environment['complete']
    for name,digest in environment['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    for key,value in data['runtime'].items():assert environment['runtime'][key]==value
    tags={c['tag']:c for c in data['cases']};assert len(tags)==11
    for c in tags.values():
        m=c['mesh'];assert m['sigma_air']==0 and m['geometry_order']==2
        assert m['curvaturesafety']==(3 if m['geometry']=='coax' else 2)
        assert len(c['runs'])==2 and all(r['modes']==4 for r in c['runs'])
        assert [r['s0'] for r in c['runs']]==[0.,2*math.pi*1e4]
        assert len(c['A_T_runs'])==2
        assert all(r['modes']==4 for r in c['A_T_runs'])
        assert m['minimum_sampled_jacobian_determinant']>0
        assert sum(m['region_element_counts'].values())==m['ne']
        if m['geometry']=='notched':
            assert m['phi_departure_l2_relative']>1e-3 and m['dc_transverse_current_fraction']>1e-3
        assert max(c['identities'].values())<1e-10
        assert c['kernel_defect']<1e-10 and c['kernel_fraction_max']<=1e-8 and c['original_residual_max']<=1e-8
        for r in c['runs']:
            check_run(r)
            assert r['gauge_port_invariance_max']<1e-8
            assert max(max(g['J_relative'],g['B_relative']) for g in r['gauge_field_invariance'])<1e-8
            for e in r['energy']:
                assert min(e['magnetic_region_integrals']+e['cumulative_region_integrals'])>=0
                assert abs(sum(e['magnetic_region_integrals'])/e['magnetic']-1)<1e-9
            for row in r['frequency']:
                assert min(row['magnetic_regions'])>=0
                assert abs(sum(row['magnetic_regions'])/row['full_magnetic_energy_form']-1)<1e-9
                assert max(abs(np.array(row['magnetic_region_integrals'])/row['magnetic_regions']-1))<1e-9
                expected=row['skin_depth_m']>=2*m['region_max_chord_m']['conductor']/m['order']
                assert row['mesh_resolution_screen']==expected
                if m['geometry']=='coax':
                    reference=oracle(2j*math.pi*row['hz'],m['enclosure'])
                    assert abs(reference/complex(*row['oracle_Z'])-1)<1e-10
                    assert abs(abs(complex(*row['full_Z'])/reference-1)-row['full_oracle_error'])<1e-10
        for r in c['A_T_runs']:check_run(r);assert r['original_residual_max']<1e-8
    control=data['stage_A_control'];assert max(control['operator_defects'].values())<1e-10 and control['air_energy_max']==0
    assert data['quadrature_check']['max_element_relative']<1e-8
    fine=tags['coax_p2_base'];assert max(fine['oracle_dc_region_relative_errors'])<.01
    resolved=[r for r in fine['runs'][0]['frequency'] if r['mesh_resolution_screen']]
    assert resolved and max(r['full_oracle_error'] for r in resolved)<.01
    assert max(r['relative_current_defect'] for r in fine['DC_ampere']['loops'])<.01
    assert abs((fine['mesh']['conductor_volume']+fine['mesh']['air_volume'])/(math.pi*fine['mesh']['enclosure']**2*.015)-1)<.01
    assert max(fine['mesh'][k] for k in ['conductor_volume_error','contact_area_error','conductor_perimeter_error'])<.01
    for region,label in [('conductor','D'),('air','A')]:
        counts=[tags[t]['mesh']['region_element_counts'][region] for t in ['notched_p1_base',f'notched_p1_{label}1',f'notched_p1_{label}2']]
        assert counts[0]<counts[1]<counts[2],counts
    bs=[tags[t] for t in ['notched_b15','notched_p1_base','notched_b30']]
    assert all(a['runs'][0]['L'][0]<b['runs'][0]['L'][0] for a,b in zip(bs,bs[1:]))
    exported=json.loads((ROOT/'docs/data/general_3d_air_matrices.json').read_text());assert exported['complete']
    assert exported['source_sha256']==data['source_sha256']
    k,kd,ka,m,s,c,g,p=map(lambda key:dense(exported[key]),['K','K_D','K_air','M','S','C','G','P'])
    for left,right in [(k,kd+ka),(m@g,c.T@p),(c@g,s@p)]:
        assert np.linalg.norm(left-right)/max(np.linalg.norm(left),np.linalg.norm(right),1e-300)<1e-10
    air_columns=np.flatnonzero(np.max(abs(p),axis=0)==0);assert len(air_columns)>0
    assert np.linalg.norm(m@g[:,air_columns])<1e-8
    assert len(exported['geometry']['region_indices'])==exported['metadata']['ne']
    check_small(exported)
    wls=json.loads((ROOT/'docs/data/general_3d_air_oracle.json').read_text())
    assert all(wls['checks'].values()) and all(r['passed'] for r in wls['cases'])
    for native, exact in zip(fine['runs'], wls['cases']):
        assert abs(native['s0']-exact['s0'])<1e-9
        for key in ['Rhat','L']:
            assert np.max(abs(np.array(native['oracle_'+key])/exact[key]-1))<1e-12
    assert wls['sourceSHA256']==hashlib.sha256((ROOT/'mathematica/derive_coax_enclosure.wls').read_bytes()).hexdigest()
    fields=json.loads((ROOT/'docs/data/general_3d_air_fields.json').read_text());assert fields['complete']
    for name,digest in fields['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    for row in fields['cases']:
        z=oracle(2j*math.pi*row['hz'])
        power=row['unit_current_joule']+2j*math.pi*row['hz']*(row['unit_current_magnetic_conductor']+row['unit_current_magnetic_air'])
        assert abs(power/z-1)<1e-10 and row['power_identity_defect']<1e-35
        fe=next(r for r in fine['runs'][0]['frequency'] if r['hz']==row['hz'])
        current2=abs(1/complex(*fe['full_Z']))**2
        targets=np.array([row['unit_current_magnetic_conductor'],row['unit_current_magnetic_air']])
        if fe['mesh_resolution_screen']:
            assert np.max(abs(np.array(fe['magnetic_regions'])/current2/targets-1))<.01
            assert abs(fe['full_loss']/current2/row['unit_current_joule']-1)<.01
    shell=json.loads((ROOT/'docs/data/general_3d_air_shell.json').read_text());assert shell['complete']
    for name,digest in shell['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    refined=next(r for r in shell['cases'] if r['tag']=='coax_p2_A1_DC')
    assert refined['relative_max_balance']<.01
    assert max(r['relative_current_defect'] for r in refined['ampere_midplane']['loops'])<.01
    assert refined['mesh']['region_element_counts']['air']>fine['mesh']['region_element_counts']['air']
    print('PASS enclosure CLN: original-matrix recurrence/contact, conductor-air energies, resolved coax oracle, geometry budget, regional refinement, stage-A operator control and source hashes')

if __name__=='__main__':main()
