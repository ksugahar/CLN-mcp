"""Collect separately executed native T--Omega cohorts; no numerical relabelling."""
import argparse,hashlib,json,math
from pathlib import Path

CASES=['c1-notched-p0-final','c1-notched-h8-p0','c1-notched-h6-p0',
       'c1-notched-h8-p1','c1-notched-h6-p1','c1-notched-h4-p1',
       'c1-coax-h12-p0','c1-coax-h8-p0','c1-coax-h12-p1']


def collect(directory,root):
 directory,root=Path(directory),Path(root);cases=[];sources=None;small=None
 for tag in CASES:
  item=json.loads((directory/(tag+'.json')).read_text());assert item['complete']
  if sources is None:sources=item['source_sha256']
  assert sources==item['source_sha256']
  item['tag']=tag
  if tag==CASES[0]:small=item.copy()
  del item['export'];cases.append(item)
 cross=json.loads((directory/'c1-cross-forms.json').read_text());assert cross['complete']
 identities=dict(sources)
 for name in ['validation/cln3d/tomega_cross_forms.py','validation/cln3d/collect_tomega_air.py']:
  identities[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
 for name,digest in cross['source_sha256'].items():assert identities[name]==digest
 out=root/'docs/data';out.mkdir(exist_ok=True)
 (out/'tomega_air_matrices.json').write_text(json.dumps(small,indent=2)+'\n',encoding='utf-8',newline='\n')
 record=dict(complete=True,cases=cases,cross_form=cross['cases'],source_sha256=identities,
   matrix_sha256=hashlib.sha256((out/'tomega_air_matrices.json').read_bytes()).hexdigest(),
   limitations=['Same-mesh reduction error is distinct from FE and cross-form error.',
    'HCurl orders 0 and 1 have equal curl-current dimension here; order 1 enriches the magnetic gradient field.',
    'Coax maxh 12 to 8 mm did not increase element counts and is not refinement.',
    'Deep elements are not cross-validated; no continuum accuracy at 100 kHz or 1 MHz is established.',
    'Air closure and magnetic scalar stationarity are discrete weak identities; there is no open-space exterior.'])
 (out/'tomega_air.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
 print('PASS collected',len(cases),'native cases and',len(cross['cases']),'three-form checks')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--directory',required=True);args=p.parse_args()
 collect(args.directory,Path(__file__).resolve().parents[2])
