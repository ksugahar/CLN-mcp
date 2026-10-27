"""Portable symbolic-boundary contracts and independent fixed-case numerical oracles."""
from pathlib import Path
from fractions import Fraction as F
import ast,copy,json,sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'validation'))
from reference_oracles import bessel_series,type2_series,truncated_energy
from reference_records import historical,wire_record

def test_no_live_sympy_imports():
    """Guards: a second CAS is not reintroduced. Assumes: explicit Python imports."""
    for directory in ['src','validation','tests','tools']:
        for p in (ROOT/directory).rglob('*.py'):
            tree=ast.parse(p.read_text(encoding='utf-8-sig'))
            for n in ast.walk(tree):
                names=[x.name for x in n.names] if isinstance(n,ast.Import) else [n.module or ''] if isinstance(n,ast.ImportFrom) else []
                assert not any(x.split('.')[0]=='sympy' for x in names),str(p)

def test_independent_bessel_contact_rejects_element_mutation():
    """Guards: numeric energy ladder matches defining Bessel moments. Assumes: fixed wire normalization."""
    rs=[1,3,5,7,9];ls=[F(1,8),F(1,16),F(1,24),F(1,32)]
    assert type2_series(rs,ls)==bessel_series()
    bad=ls.copy();bad[-1]*=2
    assert type2_series(rs,bad)!=bessel_series()

def test_truncated_energy_exact_moments_and_negative_control():
    """Guards: fixed rational Galerkin/contact oracle survives CAS removal. Assumes: immutable accepted energy data."""
    for case in historical('energy_recursion.json')['truncated_instances']:
        truncated_energy(case)
        changed=copy.deepcopy(case);changed['elements_LR'][0]='999'
        with pytest.raises(AssertionError,match='elements differ'):truncated_energy(changed)

def test_wire_record_has_exact_source_binding():
    """Guards: saved E-H/A/T proof is not detached from its sources. Assumes: portable record."""
    assert wire_record()['series_contact_degree']==8

def test_missing_symbolic_record_is_not_a_fallback(monkeypatch,tmp_path):
    """Guards: missing Mathematica output cannot pass or start another CAS. Assumes: empty data root."""
    import reference_records as records
    monkeypatch.setattr(records,'ROOT',tmp_path)
    with pytest.raises(RuntimeError,match='not been qualified'):records.symbolic_record()

def test_changed_symbolic_source_rejects_record(monkeypatch,tmp_path):
    """Guards: boolean success cannot mask source drift. Assumes: synthetic schema only."""
    import reference_records as records
    package=tmp_path/'src/cln_mcp_server/resources/SymbolicCLN.wl';package.parent.mkdir(parents=True);package.write_text('changed')
    directory=tmp_path/'docs/data';directory.mkdir(parents=True)
    (directory/'symbolic_cln.json').write_text(json.dumps({'schema':'cln.symbolic-migration.v1','passed':True,'checks':{'toy':True},'source_sha256_lf':{'src/cln_mcp_server/resources/SymbolicCLN.wl':'0'*64}}))
    monkeypatch.setattr(records,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='source changed'):records.symbolic_record()
