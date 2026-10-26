"""Version metadata contracts; no solver, build or installed-package mutation."""
import ast
import json
from pathlib import Path
import subprocess
import sys
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_versions_follow_the_declared_build_source():
    """Guards: namespace/build version drift. Assumes: Hatch uses its configured path."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "version" not in config["project"], config["project"]
    assert "version" in config["project"]["dynamic"], config["project"]
    source = ROOT / config["tool"]["hatch"]["version"]["path"]
    tree = ast.parse(source.read_text(encoding="utf-8"))
    values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)]
    assert len(values) == 1 and isinstance(values[0], str), values
    code = """import json, sys
sys.path.insert(0, sys.argv[1])
import cln, cln_mcp_server
print(json.dumps({'cln': cln.__version__, 'mcp': cln_mcp_server.__version__,
                  'numeric_imports': [m for m in ['numpy','ngsolve','radia'] if m in sys.modules]}))
"""
    result = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT / "src")],
                            capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    assert actual == {"cln": values[0], "mcp": values[0], "numeric_imports": []}, actual
