"""Portable gate for the native high-order element curl experiment."""
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(data):
    source = (ROOT/'validation/element_curl_recovery.py').read_text(encoding='utf-8-sig').encode()
    assert data['source_sha256'] == hashlib.sha256(source).hexdigest()
    assert data['schema'] == 1 and data['host'] and data['ngsolve']
    cases = data['cases']
    assert len(cases) == 6
    assert {(c['order'], c['seed']) for c in cases} == {(p, s) for p in (1, 2, 3) for s in (17, 29)}
    for c in cases:
        assert c['elements'] == 12 and c['ndof'] == {1:52, 2:168, 3:392}[c['order']]
        assert c['local_curl_ranks'] == [{1:3, 2:11, 3:26}[c['order']]]*12
        assert set(c['greedy_curl_errors']) == {'forward','reverse','shuffled'}
        small = list(c['greedy_curl_errors'].values()) + [c[k] for k in (
            'iterative_curl_error', 'assembled_reference_curl_error',
            'iterative_reference_curl_difference', 'integrated_curl_error',
            'original_residual', 'element_assembly_defect')]
        assert all(math.isfinite(x) and 0 <= x < 1e-8 for x in small)
        assert .1 < c['incompatible_curl_error'] < 1
        assert c['lsmr_stop'] in (1, 2) and 0 < c['iterations'] < 10000


if __name__ == '__main__':
    data = json.loads((ROOT/'docs/data/element_curl_recovery.json').read_text(encoding='utf8'))
    verify(data)
    for field in ('original_residual', 'incompatible_curl_error'):
        bad = copy.deepcopy(data)
        bad['cases'][0][field] = .01 if field == 'original_residual' else 0
        try:
            verify(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid evidence accepted: '+field)
    print('PASS six p=1/2/3 native curl cases; residual and incompatible-field corruption rejected')
