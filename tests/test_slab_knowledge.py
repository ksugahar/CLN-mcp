"""Structured knowledge contracts, not numerical validation."""
import json
from cln_mcp_server.knowledge.input_ports import get_input_port_documentation


def test_slab_and_radial_topics_are_structured_and_discoverable():
    """Guards: structured topics and source/limit metadata. Assumes: static knowledge."""
    slab = json.loads(get_input_port_documentation('slab'))
    radial = json.loads(get_input_port_documentation('radial_modes'))
    assert slab['schema'] == 'cln.knowledge.symmetric-slab.v1', slab
    assert 'ohm' in slab['port']['transfer'] and 'A/m' in slab['port']['current'], slab['port']
    assert slab['verification_status'] and slab['references'], slab
    assert radial['electric'] and radial['magnetic'] and radial['reference'], radial
    assert radial['execution_status'], radial
    all_topics = get_input_port_documentation('all')
    assert get_input_port_documentation('slab') in all_topics
    assert get_input_port_documentation('radial_modes') in all_topics
