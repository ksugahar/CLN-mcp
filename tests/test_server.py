import importlib.util
import pathlib
import pytest
from cln_mcp_server import server


def test_core_tools_return_text():
    for fn in (server.cln_overview, server.cln_boundary_conditions, server.cln_pitfalls,
               server.cln_wire_benchmark, server.cln_references_and_docs):
        assert len(fn()) > 100
    assert "a(T0, N)" in server.cln_algorithm("T-Omega")


@pytest.mark.parametrize("area", list(server._MODULES))
def test_every_knowledge_area_loads(area):
    assert len(server.cln_knowledge(area)) > 200


def test_diffusion_formulation():
    txt = server.cln_diffusion_formulation("heat")
    assert "R_2n" in txt and "thermal resistance" in txt
    with pytest.raises(ValueError, match="physics"):
        server.cln_diffusion_formulation("unknown")


@pytest.mark.skipif(__import__("shutil").which("wolframscript") is None, reason="no Mathematica")
def test_diffusion_derive_cylinder():
    import json
    out = json.loads(server.cln_diffusion_derive("cylinder", 3))
    assert out["R_2n"].replace(" ", "") == "{1/(8*a*Pi),1/(256*a*Pi),1/(1944*a*Pi)}"


def test_mathematica_package_is_package_data():
    """C3: the WL resource lives inside cln_mcp_server, so a wheel install can find it."""
    import importlib.resources
    from cln_mcp_server import diffusion_tools
    res = importlib.resources.files("cln_mcp_server") / "resources" / "DiffusionCLN.wl"
    assert res.is_file()
    assert diffusion_tools._PKG.is_file()
    assert diffusion_tools._PKG.resolve() == pathlib.Path(str(res)).resolve()


@pytest.mark.skipif(importlib.util.find_spec("hatchling") is None, reason="hatchling not installed: wheel not built")
def test_wheel_contains_mathematica_package(tmp_path):
    import ast
    from email.parser import BytesParser
    import subprocess
    import sys
    import tarfile
    import zipfile
    root = pathlib.Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "src/cln/_version.py").read_text(encoding="utf-8"))
    expected = next(ast.literal_eval(node.value) for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "__version__"
                            for t in node.targets))
    subprocess.run([sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-t", "sdist", "-d", str(tmp_path)],
                   cwd=root, check=True, capture_output=True)
    with zipfile.ZipFile(next(tmp_path.glob("*.whl"))) as wheel:
        names = wheel.namelist()
        assert "cln_mcp_server/resources/DiffusionCLN.wl" in names
        assert "cln/_version.py" in names
        assert wheel.read("cln/_version.py") == (root / "src/cln/_version.py").read_bytes()
        metadata = next(n for n in names if n.endswith(".dist-info/METADATA"))
        actual = BytesParser().parsebytes(wheel.read(metadata))["Version"]
        assert actual == expected, (actual, expected)
    with tarfile.open(next(tmp_path.glob("*.tar.gz"))) as sdist:
        names = sdist.getnames()
        source = next(n for n in names if n.endswith("/src/cln/_version.py"))
        assert sdist.extractfile(source).read() == (root / "src/cln/_version.py").read_bytes()
        metadata = next(n for n in names if n.count("/") == 1 and n.endswith("/PKG-INFO"))
        actual = BytesParser().parsebytes(sdist.extractfile(metadata).read())["Version"]
        assert actual == expected, (actual, expected)



def test_knowledge_index_and_notebooks():
    assert "basics" in server.cln_knowledge("index")
    assert "T_Omega" in server.cln_notebook()


@pytest.mark.parametrize("area,topic", [("3d", "overview"), ("3d", "AT"),
    ("bem", "overview"), ("sibc_orthogonal", "math"), ("sphere_dd", "overview")])
def test_specific_knowledge_sections_are_reachable(area, topic):
    """Guards: every formerly no-argument area accepts topics. Assumes: text only."""
    assert len(server.cln_knowledge(area, topic)) > 100


@pytest.mark.parametrize("area", list(server._MODULES))
def test_unknown_knowledge_topic_is_an_error(area):
    """Guards: unknown topics do not succeed as prose/index. Assumes: invalid key."""
    with pytest.raises(ValueError, match="Unknown"):
        server.cln_knowledge(area, "no_such_topic")


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, 9, 1000000])
def test_symbolic_stage_bound_precedes_any_launch(monkeypatch, bad):
    """Guards: invalid stages cannot launch CAS. Assumes: operational cap, not accuracy."""
    from cln_mcp_server import diffusion_tools as dt
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid input reached runtime discovery/launch")
    monkeypatch.setattr(dt.shutil, "which", forbidden)
    monkeypatch.setattr(dt.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="nstages"):
        dt.derive("cylinder", bad)


def test_symbolic_missing_runtime_and_failed_derivation_raise(monkeypatch):
    """Guards: runtime/derivation failures are errors. Assumes: mocked CAS only."""
    from types import SimpleNamespace
    from cln_mcp_server import diffusion_tools as dt
    monkeypatch.setattr(dt.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="wolframscript"):
        dt.derive()
    monkeypatch.setattr(dt.shutil, "which", lambda _: "fake-wolframscript")
    monkeypatch.setattr(dt.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="failed"))
    with pytest.raises(RuntimeError, match="derivation failed"):
        dt.derive()


@pytest.mark.parametrize("tool,arguments", [
    ("cln_algorithm", {"formulation": "unknown"}),
    ("cln_knowledge", {"area": "unknown"}),
    ("cln_notebook", {"name": "unknown"}),
    ("cln_diffusion_formulation", {"physics": "unknown"}),
    ("cln_diffusion_derive", {"nstages": 9}),
    ("cln_diffusion_derive", {"nstages": True}),
])
def test_invalid_requests_propagate_through_mcp_error_path(tool, arguments):
    """Guards: FastMCP emits errors, not normal text. Assumes: in-process/no launch."""
    import asyncio
    from mcp.server.fastmcp.exceptions import ToolError
    with pytest.raises(ToolError):
        asyncio.run(server.mcp.call_tool(tool, arguments))
