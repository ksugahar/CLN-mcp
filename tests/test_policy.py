"""The repository publishes implementations only as .py, .wls/.wl, or executed docs/*.ipynb
(docs/REPOSITORY_POLICY.md). tools/policy_lint.py enforces that and the publication boundary.
The negative tests build throw-away trees in tmp_path; the forbidden tokens live only in this
allow-listed test file, never in production content."""
import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("policy_lint", Path(__file__).resolve().parents[1]/"tools"/"policy_lint.py")
policy_lint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy_lint)


def test_repository_policy():
    problems = policy_lint.check()
    assert not problems, "\n".join(" ".join(p) for p in problems)


def _rules(tmp_path, files):
    for name, content in files.items():
        f = tmp_path/name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(content) if isinstance(content, bytes) else f.write_text(content, encoding="utf-8")
    return {p[0] for p in policy_lint.check(tmp_path)}


@pytest.mark.parametrize("files, rule", [
    ({"src/a.m": "x = 1;"}, "P2"),
    ({"src/model.mph": b"\x00"}, "P2"),
    ({"notes.md": "validated against COMSOL"}, "P3"),
    ({"a.py": 'p = r"C:\\Users\\someone\\x"'}, "P4"),
    ({"a.py": 'p = "Z:/fictional_project/fixture.json"'}, "P4"),          # forward-slash drive path
    ({"data/r.dat": "1 2 3"}, "P6"),
    ({"data/r.npz": b"PK"}, "P6"),
    ({"a.wls": 'FileNameJoin[{"W:", "x"}]'}, "P7"),
    ({"README.md": "see <archive>/x"}, "P7"),
    ({"a.py": "from empy import solver"}, "P8"),
    ({"a.py": "cpp_solver = 'EMPY'"}, "P8"),
    ({"bin/solver.pyd": b"MZ"}, "P8"),
    ({"bin/x.dll": b"MZ"}, "P8"),
    ({"bin/x.exe": b"MZ"}, "P8"),
    ({"tutorial.ipynb": json.dumps({"cells": []})}, "P1"),            # notebooks only under docs/
    ({"docs/t.ipynb": json.dumps({"cells": [{"cell_type": "code", "source": ["1+1"], "execution_count": None}]})}, "P5"),
])
def test_each_rule_fires(tmp_path, files, rule):
    assert rule in _rules(tmp_path, files)


def test_clean_tree_and_allowed_install_path(tmp_path):
    assert _rules(tmp_path, {
        "src/ok.py": 'LTSPICE = r"C:\\Program Files\\ADI\\LTspice\\LTspice.exe"\n',
        "mathematica/ok.wls": "Print[1]\n",
        "data/ok.json": "{}",
        "docs/ok.ipynb": json.dumps({"cells": [{"cell_type": "code", "source": ["1"], "execution_count": 1}]}),
    }) == set()
