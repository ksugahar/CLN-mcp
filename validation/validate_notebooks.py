"""Validate executed notebooks, local links, syntax and recorded source identities."""
from pathlib import Path
import ast,hashlib,json,re
import nbformat
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/'validation').rglob('*.py'):
    ast.parse(p.read_text(encoding='utf-8-sig'))
record=json.loads((ROOT/'docs/data/three_dimensional.json').read_text(encoding='utf-8-sig'))
for name,digest in record['source_sha256'].items():
    assert hashlib.sha256((ROOT/'validation'/name).read_text(encoding='utf-8-sig').encode()).hexdigest()==digest,name
gauge=json.loads((ROOT/'docs/data/three_dimensional_gauge.json').read_text(encoding='utf-8-sig'))
for name,digest in gauge['provenance']['source_sha256_lf'].items():
    assert hashlib.sha256((ROOT/'validation/cln3d'/name).read_text(encoding='utf-8-sig').encode()).hexdigest()==digest,name
for p in (ROOT/'docs').glob('*.ipynb'):
    notebook=nbformat.read(p,as_version=4);nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code':
            assert cell.execution_count is not None,p.name
            assert not any(o.output_type=='error' for o in cell.outputs),p.name
        else:
            for link in re.findall(r'\]\(([^)]+)\)',cell.source):
                if '://' not in link and not link.startswith('#'):
                    assert (p.parent/link.split('#')[0]).exists(),(p.name,link)
print('PASS executed notebooks, links, Python syntax and 3D source identities')
