"""Exercise the real stdio entry point in a fresh numerical process.

Direct Python calls cannot catch Windows native-runtime/stdio startup deadlocks.
"""
import os
from pathlib import Path
import subprocess
import sys


CLIENT = r'''
import asyncio, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def check():
    params = StdioServerParameters(command=sys.executable, args=['-m', 'cln_mcp_server.server'], env=dict(os.environ))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert 'cln_diffusion_physical_model' in [tool.name for tool in tools.tools]
            result = await session.call_tool('cln_diffusion_physical_model', {
                'stiffness': [[2, -1], [-1, 2]], 'capacity': [[2, 0], [0, 1]],
                'load': [1, 0], 'nstages': 2, 'nodes': [1, 0], 'drive': .5,
                'initial_field': [.7, -.2]})
            assert not result.isError, result
            model = json.loads(result.content[0].text)
            assert model['certificate']['passed']
            assert abs(model['state']['balance_residual']) < 1e-12
            assert model['initial_projection']['residual_quadratic_storage'] < 1e-25
            assert 'data.x0 = [' in model['matlab_function']
            info = await session.call_tool('cln_diffusion_energy_formulation', {})
            assert not info.isError and 'PRIMA can preserve' in info.content[0].text
            bad = await session.call_tool('cln_diffusion_physical_model', {
                'stiffness': [[0]], 'capacity': [[1]], 'load': [1]})
            assert bad.isError
            exported = await session.call_tool('cln_export_lti', {
                'resistance': [2, 3], 'storage': [0.0001, 0.0002], 'kind': 'rc', 'name': 'thermal'})
            assert not exported.isError, exported
            bundle = json.loads(exported.content[0].text)
            assert bundle['states'] == 2
            assert 'Laplace=' in bundle['files']['thermal.lib']
            assert 'ss(data.A, data.B, data.C, data.D)' in bundle['files']['thermal.m']
    print('MCP energy protocol passed')

async def main():
    await asyncio.wait_for(check(), timeout=30)
asyncio.run(main())
'''


def test_energy_tools_over_stdio(tmp_path):
    client = tmp_path / 'client.py'
    client.write_text(CLIENT, encoding='utf-8')
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    result = subprocess.run([sys.executable, str(client)], env=env, capture_output=True,
                            text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'MCP energy protocol passed' in result.stdout
