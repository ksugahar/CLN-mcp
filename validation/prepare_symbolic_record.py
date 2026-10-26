"""Normalize a completed CAS result for portable source-bound record checks.

No CAS launch. Both producer and package raw hashes must match the evaluated
source bytes (LF or checkout CRLF); scientific values/checks are not rewritten.
"""
import argparse,hashlib,json
from pathlib import Path
from reference_records import ROOT,lf_hash
NAMES=['src/cln_mcp_server/resources/SymbolicCLN.wl','validation/symbolic_cln.wls']
def prepare(raw,output):
 data=json.loads(raw.read_text(encoding='utf-8'))
 if data['passed'] is not True or not all(x is True for x in data['checks'].values()):raise ValueError('CAS result failed')
 for name,field in zip(NAMES,['package_sha256','producer_sha256']):
  source=(ROOT/name).read_text(encoding='utf-8-sig')
  candidates={hashlib.sha256(source.encode()).hexdigest(),hashlib.sha256(source.replace('\n','\r\n').encode()).hexdigest()}
  if data[field] not in candidates:raise ValueError('CAS source hash mismatch '+name)
 data['source_sha256_lf']={name:lf_hash(ROOT/name) for name in NAMES}
 data['source_hash_convention']='UTF-8 without BOM, universal newlines to LF; raw CAS hashes retained separately'
 output.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.raw,a.output)
