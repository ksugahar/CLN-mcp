"""Verify common physical excitation, stored transfer functions and source versions."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'docs/data'
for size in ['coarse','fine']:
    d=json.loads((DATA/f'same_excitation_3d_{size}.json').read_text(encoding='utf-8-sig'))
    old=json.loads((DATA/f'foster_3d_{size}.json').read_text(encoding='utf-8-sig'))
    assert d['source_contract']['no_virtual_port_excitation']
    assert max(d['common_source_checks'].values())<1e-8
    assert d['direct_field_reference_difference']<1e-8
    freq=np.asarray(d['frequency_hz']);u=2j*np.pi*freq*d['tau']
    target=np.asarray(d['reference_real'])+1j*np.asarray(d['reference_imag'])
    train=np.asarray(d['field_pod']['training_frequency_hz'])
    holdout=np.all(abs(freq[:,None]/train-1)>1e-9,axis=1)
    assert int(holdout.sum())==d['holdout_points']
    for label,v in d['runs'].items():
        p=np.asarray(v['poles']);r=np.asarray(v['residues'])
        assert len(p)==len(r)==v['rank'] and np.all(p>0) and np.all(r>=0)
        z=u*np.sum(r[None,:]/(p[None,:]+u[:,None]),axis=1)
        error=abs(z/target-1)
        assert np.allclose(error,v['relative_error'],atol=1e-12,rtol=1e-6)
        assert np.isclose(error[(freq>=1e3)&holdout].max(),v['holdout_high_band_max'],atol=1e-12,rtol=1e-6)
        for metric, series in [('holdout_field_high_band_max','field_energy_relative_error'),
                               ('holdout_joule_high_band_max','joule_relative_error')]:
            assert np.isclose(np.asarray(v[series])[(freq>=1e3)&holdout].max(),
                              v[metric],atol=1e-12,rtol=1e-6),(label,metric)
        if 'POD' in label:
            assert v['construction_seconds'] is None,label
            assert v['basis_processing_seconds_partial'] >= 0,label
            assert 'full eigensystem cost excluded' in v['timing_scope'],label
        assert v['coupled_vs_foster']<1e-8
        for key in ['field_energy_relative_error','joule_relative_error','magnetic_energy_relative_error']:
            a=np.asarray(v[key]);assert a.shape==u.shape and np.all(np.isfinite(a)) and np.all(a>=0)
    for n in [4,8,12]:
        # CLN was already physical-source-driven. Keep its numerical result.
        for new_label,old_label in [(f'ordinary CLN {n}',f'CLN single point rank {n}'),(f'multipoint CLN {n}',f'CLN multipoint band rank {n}')]:
            v=d['runs'][new_label];assert v['rank']==n
            assert np.allclose(v['relative_error'],old['runs'][old_label]['relative_error'],atol=1e-10,rtol=1e-5)
        assert d['runs'][f'bulk2 + boundary-response POD {n}']['rank']==n
        assert d['runs'][f'field POD {n}']['rank']==n
    print(size+': common source, CLN regression, holdouts, positive Foster equivalence passed')
m=json.loads((DATA/'same_excitation_3d_sources.json').read_text(encoding='utf-8'))
for relative,expected in m['source_sha256_lf'].items():
    content=(ROOT/relative).read_text(encoding='utf-8-sig').replace('\r\n','\n')
    assert hashlib.sha256(content.encode()).hexdigest()==expected,relative
print('Common-excitation source integrity passed')
