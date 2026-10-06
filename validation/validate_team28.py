"""Lightweight checks of recorded physical CLN evidence; no native FE dependency."""
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sy
from surface_hybrid.ladder_eval import z_type1

ROOT = Path(__file__).resolve().parents[1]


def symbolic_crosscheck():
    """Independent CAS reconstruction of the two-state Wolfram identity."""
    a,b,c,l,s = sy.symbols('a b c l s', nonzero=True)
    k = sy.eye(2); m = sy.Matrix([[a,b],[b,c]])
    f = sy.Matrix([sy.sqrt(l),0]); magnetic = f; electric = sy.zeros(2,1)
    elements = []
    for _ in range(2):
        inductance = sy.simplify((magnetic.T*k*magnetic)[0])
        electric = sy.simplify(electric+magnetic/inductance)
        resistance = sy.simplify(1/(electric.T*m*electric)[0])
        elements.extend([inductance,resistance])
        magnetic = sy.simplify(magnetic-resistance*m*electric)
    field = s*(f.T*(k+s*m).inv()*f)[0]
    assert sy.simplify(z_type1(elements,0,s)-field) == 0
    assert magnetic == sy.zeros(2,1)


def equilibrium(rows, key, weight):
    roots = []
    for a, b in zip(rows[:-1], rows[1:]):
        fa, fb = a[key]-weight, b[key]-weight
        if fa > 0 > fb:
            roots.append(a['height_mm']+(b['height_mm']-a['height_mm'])*fa/(fa-fb))
    assert len(roots) == 1
    return roots[0]


def check_row(row, count, frequency, current):
    elements = np.asarray(row['elements_LR'])
    assert elements.shape == (2*count,) and np.all(np.isfinite(elements)) and np.all(elements > 0)
    assert np.array_equal(elements[::2], row['L_henry'])
    assert np.array_equal(elements[1::2], row['R_ohm'])
    assert row['ne'] > 0 and row['free_dofs'] > 0
    for key, value in row.items():
        if isinstance(value, (float, int)):
            assert np.isfinite(value), key
    error = abs(row['full_upward_force_N']-row['cln_upward_force_N'])
    assert abs(error-row['force_absolute_error_N']) < 1e-14 and error < .001
    for key, limit in [('full_relative_residual',1e-9),('circuit_field_relative_defect',1e-8),
                       ('magnetic_energy_orthogonality',1e-8),('electric_loss_orthogonality',1e-8),
                       ('element_volume_integral_defect',1e-9),('full_power_relative_defect',1e-8)]:
        assert 0 <= row[key] < limit, key
    z = complex(row['cln_impedance_real'], row['cln_impedance_imag'])
    full_z = complex(row['full_impedance_real'], row['full_impedance_imag'])
    assert abs(z_type1(elements,0,2j*np.pi*frequency)/z-1) < 1e-8
    assert abs(abs(z/full_z-1)-row['impedance_relative_error']) < 1e-12
    assert row['impedance_relative_error'] < 1e-6
    assert row['full_joule_W'] > 0
    assert abs(.5*current**2*full_z.real/row['full_joule_W']-1) < 1e-8
    assert abs(.5*current**2*full_z.real/row['full_input_loss_W']-1) < 1e-12


def main():
    symbolic_crosscheck()
    data = json.loads((ROOT/'docs/data/team28_cln.json').read_text())
    assert data['complete'] is True and data['stages'] == 6 and data['element_count'] == 12
    assert data['expansion_point'] == 0 and data['frequency_hz'] == 50 and data['peak_current_A'] == 20
    assert data['disk_mass_kg'] == .107 and data['measured_stationary_height_mm'] == 11.3
    assert data['measured_reference_url'] == 'https://www.compumag.org/jsite/images/stories/TEAM/problem28.pdf'
    assert len(data['heights']) == 25
    assert np.allclose([r['height_mm'] for r in data['heights']],3.8+np.arange(25),rtol=0,atol=1e-12)
    weight = data['disk_mass_kg']*data['gravity_m_s2']
    for row in data['heights']:
        check_row(row,6,50,20)
    for kind in ['full','cln']:
        height = equilibrium(data['heights'],kind+'_upward_force_N',weight)
        assert abs(height-data[kind+'_equilibrium_height_mm']) < 1e-12
        error = abs(height-11.3)
        assert abs(error-data[kind+'_equilibrium_reference_error_mm']) < 1e-12 and error < .6
    assert {p['label'] for p in data['settings_probes']} == {'finer_mesh','larger_air'}
    for probe in data['settings_probes']:
        for row in probe['heights']:
            check_row(row,6,50,20)
        assert abs(equilibrium(probe['heights'],'full_upward_force_N',weight)-probe['full_equilibrium_height_mm']) < 1e-12
    assert set(data['source_sha256_lf']) == {'validation/team28_cln.py','validation/surface_hybrid/ladder_eval.py'}
    for name,digest in data['source_sha256_lf'].items():
        assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest() == digest,name
    proof = json.loads((ROOT/'docs/data/energy_recursion.json').read_text())
    assert set(proof['checks']) == {'ladder_equals_field','magnetic_energy_orthogonality',
                                  'electric_loss_orthogonality','positive_elements','expected_elements','termination',
                                  'peak_phasor_force_average'}
    assert all(value is True for value in proof['checks'].values())
    assert hashlib.sha256((ROOT/'mathematica/derive_energy_recursion.wls').read_text().encode()).hexdigest() == proof['source_sha256']
    assert len(proof['truncated_instances']) == 2
    for case in proof['truncated_instances']:
        k=sy.diag(*case['K_diagonal']); m=sy.diag(*case['M_diagonal'])
        f=sy.Matrix(case['source']); s=sy.Symbol('s'); count=case['states']
        assert count < len(f) and 0 in case['M_diagonal'] and all(v>=0 for v in case['M_diagonal'])
        assert all(v is True for v in case['checks'].values())
        a=k.inv()*f; e=sy.zeros(len(f),1); modes=[]; elements=[]
        for _ in range(count):
            l=(a.T*k*a)[0]; e=e+a/l; r=1/(e.T*m*e)[0]
            modes.append(a); elements.extend([l,r]); a=a-r*k.inv()*m*e
        assert all(v > 0 for v in elements)
        assert elements == [sy.sympify(v.replace('^','**')) for v in case['elements_LR']]
        q=sy.Matrix.hstack(*modes); fr=q.T*f
        gal=s*(fr.T*(q.T*(k+s*m)*q).inv()*fr)[0]
        full=s*(f.T*(k+s*m).inv()*f)[0]
        assert sy.cancel(z_type1(elements,0,s)-gal) == 0
        error=sy.series(sy.cancel(full-gal),s,0,2*count+2).removeO()
        assert all(error.coeff(s,j)==0 for j in range(2*count+1))
        assert error.coeff(s,2*count+1)!=0 and case['first_error_power']==2*count+1
    root=json.loads((ROOT/'docs/data/team28_equilibrium.json').read_text(encoding='utf-8'))
    assert root['complete'] and 0 < root['bracket_width_mm'] < .002
    assert abs(root['weight_N']-weight)<1e-14
    assert abs(root['bracket_width_mm']-(root['bracket'][1]['height_mm']-root['bracket'][0]['height_mm'])) < 1e-14
    for row in root['fresh_heights']+root['bracket']:
        check_row(row,6,50,20)
    for kind in ['full','cln']:
        height=equilibrium(root['bracket'],kind+'_upward_force_N',weight)
        assert abs(height-root[kind+'_equilibrium_height_mm']) < 1e-12
        assert abs(height-11.3)<.6 and abs(abs(height-11.3)-root[kind+'_reference_gap_mm'])<1e-12
        assert abs(height-data[kind+'_equilibrium_height_mm']-root[kind+'_grid_interpolation_shift_mm'])<1e-12
        assert abs(root['fresh_heights'][-1][kind+'_upward_force_N']-weight)<root['peak_force_balance_tolerance_N']
    assert set(root['source_sha256_lf'])=={'validation/team28_refine_equilibrium.py','validation/team28_cln.py',
                                        'validation/surface_hybrid/ladder_eval.py','docs/data/team28_cln.json'}
    for name,digest in root['source_sha256_lf'].items():
        assert hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()==digest,name
    print('PASS TEAM 28: 25 forces, 1 mN / 0.6 mm gates, physical elements, power, settings probes and source hashes')


if __name__ == '__main__':
    main()
