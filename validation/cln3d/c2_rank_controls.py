"""Rerun magnetic quotient sensitivity with the C2 production producer."""
import hashlib,json
from pathlib import Path
import numpy as np
import ngsolve as ng
from c2_washer import run
from c2_current import evaluate

ROOT=Path(__file__).resolve().parents[2]


def main():
    primary,state=run(return_state=True);records=[]
    for tolerance in [1e-10,1e-11,1e-12]:
        result,original=evaluate(state,primary,quotient_tolerance=tolerance)
        records.append(dict(tolerance=tolerance,nullity=result['magnetic_quotient_nullity'],
                            spectrum=original['magnetic_spectrum'],
                            elements=[record['elements'] for record in result['runs']],
                            full_Z=[row['Z'] for row in result['frequency']]))
    reference=records[1]
    for record in records:
        record['max_element_relative_gap']=float(np.max(abs(np.array(record['elements'])/reference['elements']-1)))
        assert record['nullity']==reference['nullity'] and record['max_element_relative_gap']<1e-8
        assert np.max(abs(np.array(record['full_Z'])-reference['full_Z']))<1e-10
    sources=[Path(__file__),Path(__file__).with_name('c2_washer.py'),Path(__file__).with_name('c2_current.py'),
             Path(__file__).with_name('c2_topology_temporary.py')]
    result=dict(complete=True,ne=state['mesh'].ne,records=records,
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (ROOT/'docs/data/c2_rank_controls.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('PASS C2 production quotient-threshold controls')


if __name__=='__main__':
    ng.SetNumThreads(4)
    with ng.TaskManager():main()
