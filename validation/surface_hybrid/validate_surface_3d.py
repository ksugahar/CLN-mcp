"""Verify stored 3D surface/POD evidence without starting a FEM solver."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'docs/data'
for name in ['surface_3d_coarse','surface_3d_fine']:
    d=json.loads((DATA/(name+'.json')).read_text(encoding='utf-8-sig'))
    u=2j*np.pi*np.asarray(d['frequency_hz'])*d['tau']
    exact=np.asarray(d['reference_real'])+1j*np.asarray(d['reference_imag'])
    assert d['direct_field_reference_difference']<1e-8
    for label,v in d['runs'].items():
        p=np.asarray(v['poles']);r=np.asarray(v['residues'])
        assert len(p)==len(r)==v['rank']
        assert np.all(p>0) and np.all(r>=0)
        assert v['coupled_vs_foster']<1e-8
        z=u*np.sum(r[None,:]/(p[None,:]+u[:,None]),axis=1)
        assert np.min(z.real)>-1e-12
        error=abs(z/exact-1)
        assert np.allclose(error,v['relative_error'],atol=1e-12,rtol=1e-6)
        assert np.isclose(abs(np.sum(r/p)-1),v['DC_slope_error'],atol=1e-12)
        for metric in ['field_energy_relative_error','joule_relative_error','magnetic_energy_relative_error']:
            values=np.asarray(v[metric]);assert values.shape==u.shape
            assert np.all(np.isfinite(values)) and np.all(values>=0)
        if label.startswith('bulk2 +'):
            assert v['bulk_surface_coupling_ratio']>0
    for n in [4,8,12]:
        assert d['runs'][f'ordinary CLN {n}']['rank']==n
        assert d['runs'][f'bulk2 + surface POD {n}']['rank']==n
        assert d['runs'][f'physical-port POD {n}']['rank']==n
    # Assertions describe this recorded case, not a universal ranking of methods.
    assert d['runs']['physical-port POD 4']['high_band_max']<d['runs']['ordinary CLN 4']['high_band_max']
    assert d['runs']['bulk2 + surface POD 4']['high_band_max']>d['runs']['ordinary CLN 4']['high_band_max']
    print('HISTORICAL virtual-port evidence: ', name+': stored Foster reconstruction, ranks, coupling and errors passed')
manifest=json.loads((DATA/'surface_3d_sources.json').read_text())
for path,expected in manifest['source_sha256_lf'].items():
    content=(ROOT/path).read_text(encoding='utf-8-sig').replace('\r\n','\n')
    assert hashlib.sha256(content.encode()).hexdigest()==expected,path
print('HISTORICAL virtual-port evidence: ', 'Surface/POD source integrity passed')
