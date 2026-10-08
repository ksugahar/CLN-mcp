"""Recompute C2 washer evidence from owned FE fields and original matrices.

Run with a fixed Radia/NGSolve environment; native calculations are not CI
requirements. CI reconstructs the exported small pencils with NumPy only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import ngsolve as ng
import radia
import radia.cohomology as rc
from c2_washer import run
from c2_current import evaluate
from c2_axisymmetric import solve as axisymmetric
from c2_topology_temporary import insulated_loop_currents

ROOT=Path(__file__).resolve().parents[2]
SOURCES=['validation/cln3d/'+name for name in
         ['c2_washer.py','c2_current.py','c2_study.py','c2_axisymmetric.py',
          'c2_topology_temporary.py','derive_c2_solid_cylinder.wls']]
SOURCES+=['validation/surface_hybrid/ladder_eval.py']


def sparse(matrix):
    matrix=matrix.tocsr().copy()
    return dict(shape=list(matrix.shape),data=matrix.data.tolist(),
                indices=matrix.indices.tolist(),indptr=matrix.indptr.tolist())


def stripped(result):
    """Large field arrays live only in the matrix export, not summary rows."""
    result=json.loads(json.dumps(result))
    for row in result['runs']:
        for key in ['magnetic_modes','electric_modes','K_reduced','M_reduced','f_reduced']:
            row.pop(key,None)
    return result


def main(output, fine=True):
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    ng.SetNumThreads(4)
    with ng.TaskManager():
        primary,state=run(return_state=True)
        dual,dual_original=evaluate(state,primary)
        filled,filled_state=run(return_state=True,filled=True)
        filled_loops=insulated_loop_currents(filled_state['mesh'],conductor_regions='conductor',
                                            sigma=1e6,insulating_boundaries='surface')
        assert filled_loops.topology.b1==0 and len(filled_loops.currents)==0
        assert filled_loops.divergence_free_dimension==filled_loops.curl_dimension
        assert filled_loops.curl_dimension>0
    matrices=dict(primary={key:sparse(state[key]) for key in ['K','Kg','M','m','C','S','G']},
                  f=state['f'].tolist(),primary_runs=primary['runs'],dual=dual_original)
    summaries=[stripped(primary)]
    if fine:
        for order,coil_h in [(0,.002),(2,.002)]:
            print('START coil / magnetic refinement',order,coil_h,flush=True)
            with ng.TaskManager():case=run(coil_h=coil_h,a_order=order)
            summaries.append(stripped(case));print('PASS refined primary',case['ne'],case['dofs'],flush=True)
    references=[]
    for oracle,h in [(True,.0005),(False,.0005),(False,.00025)]:
        with ng.TaskManager():case=axisymmetric(h,oracle)
        references.append(case)
    reference=references[-1]
    for case in summaries:
        case['axisymmetric_comparison']=[]
        for row,ref in zip(case['runs'][0]['frequency'],reference['frequency']):
            z=complex(*row['Z']);zr=complex(*ref['Z'])
            case['axisymmetric_comparison'].append(dict(hz=row['hz'],full_Z_gap=float(abs(z/zr-1)),
                                                        loss_signed_gap=float(z.real/zr.real-1)))
        # Compare DC coil self inductance to the axisymmetric DC value, not
        # Im Z at 1 kHz. The reference producer explicitly computes it.
        case['self_inductance_signed_gap']=float(case['self_inductance']/reference['self_inductance']-1)
    output.mkdir(parents=True,exist_ok=True)
    mp=output/'c2_washer_matrices.json';mp.write_text(json.dumps(matrices,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')
    result=dict(complete=True,source_sha256=hashes,matrix_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),
                runtime=dict(radia=radia.__version__,ngsolve=ng.__version__,
                             cohomology_sha256=hashlib.sha256(Path(rc.__file__).read_bytes()).hexdigest()),
                primary=summaries,dual=stripped(dual),axisymmetric=references,
                filled_hole=dict(b1=0,loop_channels=0,curl_dimension=filled_loops.curl_dimension,
                                 current_dimension=filled_loops.divergence_free_dimension,primary=stripped(filled)),
                claim='Same-mesh energy-normalized Type1 CLN; no continuum accuracy or independent T-Omega claim.',
                limitations=['PEC return enclosure, quasi-static coil excitation, sigma=0 outside the conductor.',
                             'A-T shares the A magnetic inverse; a genuine independent T-Omega cross-check remains open.',
                             'Skin depths at 100 kHz and 1 MHz are under-resolved in the 3D cohorts.',
                             'Temporary topology adapters await published Radia API and a tested version pin.'])
    for name,digest in hashes.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    (output/'c2_washer.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('PASS C2 producer, loops, contact, changed representative and disclosed FE comparison',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'docs/data')
    parser.add_argument('--coarse-only',action='store_true');args=parser.parse_args()
    main(args.output,not args.coarse_only)
