"""Backend validation contracts without importing or executing a FE runtime."""
import ast
import math
from numbers import Integral
from pathlib import Path
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'src/cln/t_omega.py'


def constructor():
    """Execute the actual constructor AST; stop at its first FE helper call."""
    module = ast.parse(SOURCE.read_text(encoding='utf-8'))
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == 'TOmegaCLN')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    def forbidden(*args):
        raise AssertionError('FE helper reached')
    scope = {'math': math, 'Integral': Integral, 'electromagnetic_measures': forbidden}
    exec(compile(ast.Module(body=[init], type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope['__init__']


@pytest.mark.parametrize('options', [{}, {'solver': 'sparsesolv'}])
def test_positive_shift_rejects_implicit_or_iccg_before_mesh(options):
    """Guards: positive-shift rejection. Assumes: actual AST, no FE imports."""
    state = type('State', (), {})()
    with pytest.raises(ValueError, match='requires explicit'):
        constructor()(state, None, 1., 1., 1., 1., s0=1., **options)
    assert state.__dict__ == {}, state.__dict__


@pytest.mark.parametrize('s0,solver', [(0., 'sparsesolv'), (0., 'direct'), (1., 'direct')])
def test_admissible_methods_reach_fe_boundary(s0, solver):
    """Guards: admissible methods remain reachable. Assumes: FE boundary sentinel."""
    with pytest.raises(AssertionError, match='FE helper reached'):
        constructor()(type('State', (), {})(), None, 1., 1., 1., 1., s0=s0, solver=solver)
