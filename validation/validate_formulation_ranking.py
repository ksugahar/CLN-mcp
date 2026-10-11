"""Portable evidence checks for matched 3D formulation rankings (no FE runtime)."""
import copy
import hashlib
import json
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sources(data):
    for name,digest in data['source_sha256_lf'].items():
        assert '\\' not in name and '..' not in Path(name).parts
        assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()==digest,name


def ladder(el,s):
    out=np.full(np.shape(s),el[-1],dtype=complex)
    for k in range(len(el)-2,-1,-1):
        out=el[k]+out if k%2==0 else s*el[k]*out/(s*el[k]+out)
    return out


def rational_check(c,tau):
    """Check CF evaluation against independent Pade from full-system moments."""
    def cf(values,z,z0,kind):
        q=z-z0
        if kind=='Type2':return complex(ladder(values,np.asarray(q)))
        g=1/values[-1]
        for k in range(len(values)-2,-1,-1):
            g=1/values[k]+g if k%2==0 else q*g/(values[k]*g+q)
        return z/g
    with mp.workdps(60):
        z0=mp.mpf(c['z0']);y=list(map(mp.mpf,c['normalized_admittance_moments']))
        def exact_y(z):
            if z==0:return mp.mpf(1)
            x=mp.sqrt(z)
            return 2*mp.besseli(1,x)/(x*mp.besseli(0,x))
        oracle=mp.taylor(exact_y,z0,4)
        for moments,el in [(y,c['normalized_elements']),(oracle,c['exact_normalized_elements'])]:
            if c['type']=='Type1':
                moments=[z0*moments[0]]+[z0*moments[k]+moments[k-1] for k in range(1,5)]
            p,q=mp.pade(moments[:5],2,2)
            for offset in [1j,10j,3+5j]:
                z=complex(z0)+offset
                ratio=mp.polyval(p[::-1],offset)/mp.polyval(q[::-1],offset)
                expected=(z if c['type']=='Type1' else 1)/ratio
                actual=cf(el,z,float(z0),c['type'])
                assert abs(actual/complex(expected)-1)<1e-10


def verify(data,shifted):
    sources(data);sources(shifted)
    assert data['complete'] and data['passed'] and shifted['complete'] and shifted['passed']
    assert data['parameters']['s0']==0 and data['parameters']['curve_order']==3
    cases=data['cases'];assert len(cases)==40
    meshes={m['maxh']:m for m in data['meshes']}
    assert set(meshes)=={.003,.002,.0015}
    assert all(meshes[b]['ne']>meshes[a]['ne'] for a,b in [(.003,.002),(.002,.0015)])
    oracle=np.asarray(data['oracle_coefficients'])
    r0=1/(np.pi*.01**2*1e6*.01);tau=4e-7*np.pi*1e6*.01**2
    expected=np.array([r0,5e-6,3*r0,2.5e-6,5*r0,5e-6/3,7*r0])
    np.testing.assert_allclose(oracle,expected,rtol=1e-14)
    frequency=np.asarray(data['frequency_hz']);s=2j*np.pi*frequency
    assert len(frequency)==81 and frequency[0]==100 and frequency[-1]==1e6
    exact=np.asarray(data['exact_impedance']);exact=exact[:,0]+1j*exact[:,1]
    with mp.workdps(60):
        x=[mp.sqrt(2j*mp.pi*float(f)*tau) for f in frequency]
        bess=np.array([complex(r0*v*mp.besseli(0,v)/(2*mp.besseli(1,v))) for v in x])
    np.testing.assert_allclose(exact,bess,rtol=1e-14)
    for n in (3,5,7):
        np.testing.assert_allclose(abs(ladder(oracle[:n],s)/exact-1),
                                   data['analytic_truncation_relative_error'][str(n)],rtol=1e-10,atol=1e-14)
    keys=[]
    for c in cases:
        assert c['mesh_sha256']==meshes[c['maxh']]['mesh_sha256']
        assert c['order'] in (1,2,3)
        if c['status']!='pass':
            assert (c['formulation'],c['control'])==('CLN_T_Omega','quadrature4')
            assert 'Incompatible load' in c['error']
            continue
        values=np.asarray(c['coefficients']);signed=values/oracle[:len(values)]-1
        assert np.all(values>0) and np.all(np.isfinite(values))
        np.testing.assert_allclose(signed,c['signed_relative_errors'],rtol=1e-7,atol=1e-13)
        assert c['max_original_residual']<=1e-6
        for info in c['solves']:
            assert info['converged'] and max(info['explicit_residual'],info['original_rhs_residual'])<=1e-6
        for count,recorded in c['max_absolute_relative_error'].items():
            assert abs(max(abs(signed[:int(count)]))-recorded)<1e-13
            np.testing.assert_allclose(abs(ladder(values[:int(count)],s)/exact-1),
                                       c['terminal_total_relative_error'][count],rtol=1e-10,atol=1e-14)
        if c['control']=='profile':
            profile=c['field_profiles'];rho=np.asarray(profile['rho']);zeros=np.zeros_like(rho)
            targets=dict(E0=np.column_stack([zeros,zeros,np.ones_like(rho)]),
                         B1=np.column_stack([zeros,4e-7*np.pi*r0*1e6*.01*rho/2,zeros]),
                         E2=np.column_stack([zeros,zeros,2*rho**2-1]))
            for name,target in targets.items():
                error=np.linalg.norm(np.asarray(profile[name])-target)/np.linalg.norm(target)
                assert abs(error-c['field_relative_rms'][name])<1e-13
        if c['control']=='baseline':
            assert c['bonus_intorder']==8 and c['stages']==3
            keys.append((c['maxh'],c['order'],c['formulation']))
    assert len(keys)==len(set(keys))==27
    def baseline(h,p,name):
        return next(c for c in cases if (c['maxh'],c['order'],c['formulation'],c['control'])==(h,p,name,'baseline'))
    for h in meshes:
        a,t,at=[baseline(h,1,name) for name in ['CLN_APhi','CLN_T_Omega','CLN_AT']]
        assert a['max_absolute_relative_error']['3']<at['max_absolute_relative_error']['3']<t['max_absolute_relative_error']['3']
        for p in (1,2,3):
            a,t,at=[baseline(h,p,name) for name in ['CLN_APhi','CLN_T_Omega','CLN_AT']]
            assert at['max_absolute_relative_error']['7']<a['max_absolute_relative_error']['7']<t['max_absolute_relative_error']['7']
    for p in (1,2):
        a,t,at=[baseline(.003,p,name) for name in ['CLN_APhi','CLN_T_Omega','CLN_AT']]
        ea,et=[next(c for c in cases if (c['control'],c['order'])==(control,p)) for control in ['enrich_A','enrich_T']]
        assert ea['vector_orders']=={'A':p+1,'T':p} and et['vector_orders']=={'A':p,'T':p+1}
        assert abs(ea['signed_relative_errors'][2]-t['signed_relative_errors'][2])<.05*abs(at['signed_relative_errors'][2]-t['signed_relative_errors'][2])
        assert abs(et['signed_relative_errors'][2]-a['signed_relative_errors'][2])<.05*abs(at['signed_relative_errors'][2]-a['signed_relative_errors'][2])
    assert len(shifted['cases'])==36
    shifted_keys=[]
    for c in shifted['cases']:
        rational_check(c,tau)
        assert c['positive_elements'] and len(c['normalized_elements'])==5
        err=np.asarray(c['normalized_elements'])/c['exact_normalized_elements']-1
        np.testing.assert_allclose(err,c['signed_relative_errors'],rtol=1e-7,atol=1e-13)
        assert abs(max(abs(err))-c['max_first5_error'])<1e-13
        assert abs(max(abs(err[:3]))-c['max_first3_error'])<1e-13
        assert len(c['solve_diagnostics'])==(6 if c['formulation']=='A-Phi' and c['z0']==0 else 7)
        assert all(v['cg_flag']==0 and v['true_residual']<1e-8 for v in c['solve_diagnostics'])
        shifted_keys.append((c['maxh'],c['order'],c['formulation'],c['type'],c['z0']))
        # Check that the independent zero-shift pencil describes the field recursion.
        if c['z0']==0:
            b=baseline(c['maxh'],c['order'],{'A-Phi':'CLN_APhi','T-Omega':'CLN_T_Omega','A-T':'CLN_AT'}[c['formulation']])
            expected=np.asarray(b['coefficients'][:5])/(r0*np.array([1,tau,1,tau,1]))
            np.testing.assert_allclose(c['normalized_elements'],expected,rtol=2e-6,atol=1e-10)
    assert len(shifted_keys)==len(set(shifted_keys))
    for mesh in shifted['meshes']:
        assert max(mesh['kernel_defects'])<1e-10 and mesh['mass_perturbation']==0
        assert mesh['magnetic_subsolves']['count']>0 and mesh['magnetic_subsolves']['max_original_residual']<1e-8
        assert mesh['mesh_sha256']==meshes[mesh['maxh']]['mesh_sha256']
        for kind in ('Type1','Type2'):
            group=[c for c in shifted['cases'] if c['maxh']==mesh['maxh'] and c['order']==mesh['order'] and c['z0']>0 and c['type']==kind]
            assert len(group)==3
            winner=min(group,key=lambda c:c['max_first5_error'])
            assert winner['formulation']=='A-T'


if __name__=='__main__':
    data=json.loads((ROOT/'docs/data/formulation_ranking.json').read_text(encoding='utf8'))
    shifted=json.loads((ROOT/'docs/data/shifted_formulation_ranking.json').read_text(encoding='utf8'))
    verify(data,shifted)
    for mode in ('coefficient','metric','mesh','coverage','residual'):
        bad=copy.deepcopy(data)
        if mode=='coefficient':bad['cases'][0]['coefficients'][2]*=1.01
        elif mode=='metric':bad['cases'][0]['max_absolute_relative_error']['3']=0
        elif mode=='mesh':bad['cases'][0]['mesh_sha256']='different'
        elif mode=='coverage':bad['cases'].pop()
        else:bad['cases'][0]['max_original_residual']=1e-3
        try:verify(bad,shifted)
        except AssertionError:pass
        else:raise AssertionError(f'Corruption accepted: {mode}')
    print('PASS matched meshes, oracle coefficients, error windows, fields/pencil agreement, shifts, ranking controls; five corruptions rejected')
