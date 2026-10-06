"""Fresh FE force evaluations refine the TEAM 28 grid interpolant locally."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import netgen
import ngsolve
from ngsolve import SetNumThreads
from team28_cln import ROOT, MASS, GRAVITY, solve_height, equilibrium


def run(output):
    paths = ['validation/team28_refine_equilibrium.py', 'validation/team28_cln.py',
             'validation/surface_hybrid/ladder_eval.py', 'docs/data/team28_cln.json']
    def identities():
        return {name: hashlib.sha256((ROOT/name).read_text(encoding='utf-8-sig').encode()).hexdigest()
                for name in paths}
    hashes = identities()
    base = json.loads((ROOT/'docs/data/team28_cln.json').read_text(encoding='utf-8'))
    assert base['complete'] and base['stages'] == 6
    SetNumThreads(4)
    weight = MASS*GRAVITY
    lo, hi = [dict(r) for r in base['heights'] if r['height_mm'] in [10.8,11.8]]
    fresh = []
    for _ in range(8):
        flo = lo['full_upward_force_N']-weight
        fhi = hi['full_upward_force_N']-weight
        assert flo > 0 > fhi
        height = lo['height_mm']+(hi['height_mm']-lo['height_mm'])*flo/(flo-fhi)
        row = solve_height(height)
        fresh.append(row)
        defect = row['full_upward_force_N']-weight
        print(height, defect, row['cln_upward_force_N']-weight, flush=True)
        if defect > 0:
            lo = row
        else:
            hi = row
        if hi['height_mm']-lo['height_mm'] < .002 and max(
                abs(row['full_upward_force_N']-weight),
                abs(row['cln_upward_force_N']-weight)) < .0002:
            break
    assert hi['height_mm']-lo['height_mm'] < .002, 'Local force bracket not resolved'
    bracket = [lo, hi]
    full = equilibrium(bracket, 'full_upward_force_N')
    cln = equilibrium(bracket, 'cln_upward_force_N')
    assert abs(full-11.3) < .6 and abs(cln-11.3) < .6
    assert identities() == hashes, 'Sources or baseline changed during refinement'
    out = dict(scope='Local equilibrium refinement of the same finite-domain Model A FE, not a continuum/device error certificate',
               method='Safeguarded sign-bracket secant force solves; same heights evaluate full FE and physical CLN',
               complete=True, source_sha256_lf=hashes,
               runtime=dict(python=platform.python_version(), ngsolve=ngsolve.__version__, netgen=netgen.__version__),
               weight_N=weight, fresh_heights=fresh,
               bracket=bracket, bracket_width_mm=hi['height_mm']-lo['height_mm'],
               full_equilibrium_height_mm=full, cln_equilibrium_height_mm=cln,
               full_reference_gap_mm=abs(full-11.3), cln_reference_gap_mm=abs(cln-11.3),
               full_grid_interpolation_shift_mm=full-base['full_equilibrium_height_mm'],
               cln_grid_interpolation_shift_mm=cln-base['cln_equilibrium_height_mm'],
               peak_force_balance_tolerance_N=.0002)
    output.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('PASS local equilibrium bracket and both 0.6 mm gates', full, cln)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    run(Path(parser.parse_args().output))
