"""Content contracts for the CLN knowledge base (migrated from radia-mcp).

The former test only checked that every area returned > 200 characters. These
tests pin the topic structure, a few load-bearing facts, the public boundary
and the SIBC terminology contract that came over from radia-mcp.
"""
import importlib
import re
import pytest
from cln_mcp_server import server

DISPATCHED = {
    "input_ports": ("input_ports", "get_input_port_documentation"),
    "advanced": ("cln_advanced_knowledge", "get_cln_advanced_documentation"),
    "specialty": ("cln_specialty_knowledge", "get_cln_specialty_documentation"),
    "multiport": ("cln_multiport_knowledge", "get_cln_multiport_documentation"),
}


def _module(area):
    return importlib.import_module("cln_mcp_server.knowledge." + DISPATCHED[area][0])


@pytest.mark.parametrize("area", list(DISPATCHED))
def test_documented_topics_exist_and_vice_versa(area):
    mod = _module(area)
    doc = getattr(mod, DISPATCHED[area][1]).__doc__
    documented = set(re.findall(r'^\s+"?([a-z0-9_]+)"?\s+-\s', doc, re.M))
    if re.search(r'^\s+"?all"?(\s|$)', doc, re.M):
        documented.add("all")
    assert documented == set(mod._TOPICS), (area, documented ^ set(mod._TOPICS))
    for topic in mod._TOPICS:
        assert len(server.cln_knowledge(area, topic)) > 100, (area, topic)
    with pytest.raises(ValueError, match="Available:"):
        server.cln_knowledge(area, "no_such_topic")


REMOVED_TOPICS = ("high_frequency_code", "nonlinear_code", "fp_cln_code",
                  "termination_code", "hiruma_code", "bem_fem_code")


@pytest.mark.parametrize("area", list(server._MODULES))
def test_no_reference_to_removed_topics(area):
    text = server.cln_knowledge(area).lower()
    for name in REMOVED_TOPICS:
        assert name not in text, (area, name)


@pytest.mark.parametrize("area", list(DISPATCHED))
def test_no_archive_file_index(area):
    """User decision 2026-10-03: the knowledge base does not index the lab archive."""
    assert "key_files" not in _module(area)._TOPICS
    with pytest.raises(ValueError, match="Available:"):
        server.cln_knowledge(area, "key_files")


@pytest.mark.parametrize("area, topic, needles", [
    ("advanced", "fp_cln", ["Two production drivers", "INDEPENDENT of", "Harmonic Balance"]),
    ("advanced", "citations", ["CEFC"]),
    ("specialty", "bem_fem_coupled", ["5-10x", "TSVD"]),
    ("specialty", "hiruma_method", ["Lanczos"]),
    ("multiport", "euclid_tiles", ["16/22", "Foster"]),
    ("multiport", "multi_expansion_theory", ["s_0"]),
])
def test_load_bearing_facts(area, topic, needles):
    text = server.cln_knowledge(area, topic)
    for needle in needles:
        assert needle in text, (area, topic, needle)


@pytest.mark.parametrize("area", list(server._MODULES))
def test_public_boundary_no_commercial_tools(area):
    text = server.cln_knowledge(area)
    for name in ("COMSOL", "JMAG", "FEMM"):
        assert name.lower() not in text.lower(), (area, name)


# --- migrated from radia-mcp tests/test_cln_sibc_orthogonal_terminology.py and the
# --- terminology part of test_cln_sibc_evrs_esim_knowledge.py (the EVRS/ESIM production
# --- contract itself stayed in Radia's ESIM topic, so it must NOT appear here).
from cln_mcp_server.knowledge.cln_sibc_orthogonal import get_cln_sibc_orthogonal_section


def test_current_basis_terminology_precedes_historical_sections():
    for name in ("overview", "all"):
        text = get_cln_sibc_orthogonal_section(name)
        assert text.startswith("## Current terminology")
        assert text.count("## Current terminology") == 1
        assert "**Modal-bulk SIBC**" in text
        assert "**CLN + SIBC**" in text
        assert "surface_enrichment" in text


def test_evrs_esim_contract_not_in_cln_sibc():
    assert "evrs_esim" not in get_cln_sibc_orthogonal_section("list")
    assert "Production HCurl EVRS" not in get_cln_sibc_orthogonal_section("all")


def test_surface_topic_identifies_its_basis_without_retired_aliases():
    """Guards: neutral method naming. Assumes: CLN bulk differs from modal bulk."""
    text = server.cln_knowledge("basics", "surface_enrichment")
    assert text.startswith("\n# Surface enrichment:")
    assert "bulk CLN Krylov modes" in text
    with pytest.raises(ValueError, match="Unknown topic"):
        server.cln_knowledge("basics", "mixed_galerkin")


