"""Portable NumPy gate for analytic AC evidence and both CF terminations."""
import copy,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def verify(data):
    for name,digest in data['source_sha256'].items():
        assert '\\' not in name
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    f=np.asarray(data['frequency_hz']);z=2j*np.pi*f*data['tau_s']
    assert len(f)==601 and np.all(np.diff(f)>0)
    exact=data['exact_admittance'];exact=np.asarray(exact['real'])+1j*np.asarray(exact['imag'])
    assert np.all(exact.real>0) and np.all(exact.imag<0)
    for row in data['models']:
        a=row['admittance'];y=np.asarray(a['real'])+1j*np.asarray(a['imag'])
        error=abs(y/exact-1)
        assert np.all(np.isfinite(y))
        np.testing.assert_allclose(error,row['relative_y_error'],rtol=1e-8,atol=1e-15)
        mask=error<=.01;spans=[];start=None
        for i,good in enumerate(mask):
            if good and start is None:start=i
            if start is not None and (not good or i==len(f)-1):
                spans.append([float(f[start]),float(f[i if good else i-1])]);start=None
        assert spans==row['one_percent_intervals_hz']
        if row['family']=='Type1 analytic H continued fraction':
            c=np.asarray(row['coefficients']);q=z-row['z0']
            assert len(c)==6 and np.all(c>0)
            value=c[-1]*np.ones_like(q)
            for i in range(len(c)-2,-1,-1):
                value=(1/c[i] if i%2==0 else c[i])+q/value
            np.testing.assert_allclose(value/(data['rdc_ohm']*z),y,rtol=1e-11)
            assert row['pade_relative_defect']<1e-50
        if row['family']=='Type2':
            c=np.asarray(row['coefficients']);q=z-row['z0'];count=row['elements']
            assert count in (6,7) and len(c)==count and np.all(c>0)
            value=c[-1]*np.ones_like(q) if count%2 else q*c[-1]
            for i in range(count-2,-1,-1):
                value=c[i]+value if i%2==0 else q*c[i]*value/(q*c[i]+value)
            np.testing.assert_allclose(1/(data['rdc_ohm']*value),y,rtol=1e-11)
            ez=abs(exact/y-1)
            np.testing.assert_allclose(ez,row['relative_z_error'],rtol=1e-8,atol=1e-15)
            np.testing.assert_allclose(error,ez/abs(exact/y),rtol=1e-8,atol=1e-15)
            assert row['pade_relative_defect']<1e-50
            for component in ('real','imag'):
                np.testing.assert_allclose(abs(getattr(y,component)/getattr(exact,component)-1),row['relative_'+component+'_y_error'],rtol=1e-8,atol=1e-15)

if __name__=='__main__':
    data=json.loads((ROOT/'docs/data/ac_axis_study.json').read_text());verify(data)
    bad=copy.deepcopy(data);bad['models'][0]['coefficients'][-1]*=1.01
    try:verify(bad)
    except AssertionError:pass
    else:raise AssertionError('corrupted terminal element was accepted')
    print('PASS AC source identity, six/seven-element evaluation, reciprocal/component errors, sampled bands; corrupted terminal rejected')