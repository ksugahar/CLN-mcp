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


def test_wheel_contains_mathematica_package(tmp_path):
    """Guards: Version/resource drift in both artifacts. Assumes: explicit build lane."""
    import json
    import os
    import subprocess
    import sys
    if importlib.util.find_spec("hatchling") is None:
        if os.environ.get("CLN_REQUIRE_BUILD") == "1":
            pytest.fail("required artifact build needs hatchling")
        pytest.skip("optional local artifact build needs hatchling")
    root = pathlib.Path(__file__).resolve().parents[1]
    retained = os.environ.get("CLN_BUILD_ARTIFACT_DIR")
    destination = pathlib.Path(retained) if retained else tmp_path / "artifacts"
    destination.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-t", "sdist", "-d", str(destination)],
                   cwd=root, check=True, capture_output=True)
    subprocess.run([sys.executable, str(root / "tools/verify_package_artifacts.py"),
                    "--artifact-dir", str(destination)], cwd=root, check=True)
    manifest = json.loads((destination / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert {a["kind"] for a in manifest["artifacts"]} == {"wheel", "sdist"}, manifest



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
