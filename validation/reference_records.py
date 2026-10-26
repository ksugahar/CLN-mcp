"""Read-only source-bound records; no symbolic engine, FE or implicit fallback."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def lf_hash(path):return hashlib.sha256(path.read_text(encoding='utf-8-sig').encode('utf-8')).hexdigest()
def historical(name):
 ledger=json.loads((ROOT/'docs/data/reference_migration.json').read_text(encoding='utf-8'))
 path=ROOT/'docs/data'/name
 if lf_hash(path)!=ledger['record_sha256_lf'][name]:raise ValueError('historical record changed: '+name)
 return json.loads(path.read_text(encoding='utf-8-sig'))
def symbolic_record():
 path=ROOT/'docs/data/symbolic_cln.json'
 if not path.is_file():raise RuntimeError('Mathematica migration output has not been qualified; run the reviewed CAS producer first')
 d=json.loads(path.read_text(encoding='utf-8'))
 if d['schema']!='cln.symbolic-migration.v1' or d['passed'] is not True:raise ValueError('failed symbolic record')
 if not d['checks'] or any(type(x) is not bool or not x for x in d['checks'].values()):raise ValueError('failed/nonboolean symbolic checks')
 for name,digest in d['source_sha256_lf'].items():
  if lf_hash(ROOT/name)!=digest:raise ValueError('symbolic producer/source changed: '+name)
 if set(d['source_sha256_lf'])!={'src/cln_mcp_server/resources/SymbolicCLN.wl','validation/symbolic_cln.wls'}:raise ValueError('incomplete source binding')
 required={'port: matched conjugate duality','port: B/H two-index shift','approximate coefficients rejected','caveat: incompatible arbitrary magnetic boundary','gauge: A-phi E','gauge: A-phi B','gauge: T-Omega H','gauge: T-Omega J','termination inversion','parallel identity','inductive-tail exact round trip'}
 if not required.issubset(d['checks']):raise ValueError('missing migration check')
 return d

def wire_record():
 d=json.loads((ROOT/'docs/data/round_wire_maxwell.json').read_text(encoding='utf-8'))
 if d['schema']!='cln.round-wire-maxwell.v1' or d['passed'] is not True or len(d['checks'])!=67:raise ValueError('wire record schema/check set')
 if any(c['passed'] is not True for c in d['checks']):raise ValueError('failed round-wire check')
 if set(d['source_sha256_lf'])!={'src/cln_mcp_server/resources/RoundWireMaxwell.wl','validation/round_wire_maxwell.wls'}:raise ValueError('wire source binding incomplete')
 for name,digest in d['source_sha256_lf'].items():
  if lf_hash(ROOT/name)!=digest:raise ValueError('wire source changed: '+name)
 return d
