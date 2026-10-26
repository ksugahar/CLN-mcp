"""Initial-field maps checked against original FE coordinates and evolution."""
import json
import numpy as np
import pytest
from scipy.linalg import expm
from cln.energy import physical_cln
from cln_mcp_server.energy_tools import physical_model


def example(n=3):
    K = np.array([[3., -1, 0], [-1, 4, -.5], [0, -.5, 2]])
    M = np.array([[2., .2, 0], [.2, 1, .1], [0, .1, 3]])
    return K, M, physical_cln(K, M, [1., .3, -.1], n)


def test_full_order_initial_field_and_free_response():
    K, M, model = example()
    u0 = np.array([.7, -.3, .5])
    projection = model.project_initial(u0)
    assert np.allclose(model.reconstruct(projection.x0), u0, rtol=1e-11, atol=1e-12)
    lti = model.to_lti(initial_field=u0)
    for t in (0., .03, .2, 1., 4.):
        field = expm(-np.linalg.solve(M, K)*t) @ u0
        reconstructed = model.reconstruct(expm(lti.A*t) @ lti.initial_state)
        assert np.allclose(reconstructed, field, rtol=1e-10, atol=1e-12)
    assert projection.diagnostics['residual_quadratic_storage'] < 1e-25


def test_truncated_projection_is_metric_best_fit_not_euclidean():
    _, M, model = example(2)
    u0 = np.array([.7, -.3, .5])
    p = model.project_initial(u0)
    residual = u0-model.reconstruct(p.x0)
    assert np.linalg.norm(model.basis.T @ M @ residual) < 1e-12
    assert np.linalg.norm(p.x0-np.linalg.lstsq(model.basis, u0, rcond=None)[0]) > .01
    d = p.diagnostics
    assert d['full_quadratic_storage'] == pytest.approx(
        d['retained_quadratic_storage']+d['residual_quadratic_storage'], rel=1e-12)
    assert d['residual_quadratic_storage'] > 0
    for delta in ([.1, 0], [0, -.1], [.1, .2]):
        r = u0-model.reconstruct(p.x0+delta)
        assert r @ M @ r > residual @ M @ residual


def test_zero_field_and_invalid_fields():
    _, _, model = example()
    assert model.project_initial(np.zeros(3)).diagnostics['relative_field_error'] == 0
    for value in ([1, 2], [1, 2, np.nan], [1j, 0, 0], [[1, 2, 3]]):
        with pytest.raises(ValueError):
            model.project_initial(value)


def test_initial_state_export_does_not_drop_nonzero_state(tmp_path):
    _, _, model = example()
    lti = model.to_lti(initial_field=[.7, -.3, .5])
    files = lti.write(tmp_path, 'initialized')
    assert set(files) == {'initialized.m', 'initialized.json'}
    data = json.loads((tmp_path/'initialized.json').read_text())
    assert data['initial_state'] == lti.initial_state.tolist()
    assert 'data.x0 = [' in (tmp_path/'initialized.m').read_text()
    with pytest.raises(ValueError, match='nonzero initial state'):
        lti.ltspice_subcircuit()
    assert np.all(model.to_lti().initial_state == 0)
    model.to_lti().write(tmp_path, 'zero')
    with pytest.raises(FileExistsError, match='Laplace files'):
        lti.write(tmp_path, 'zero', overwrite=True)
    with pytest.raises(ValueError):
        lti.with_initial_state([1, 2])


def test_mcp_returns_field_projection_and_initialized_factory():
    K, M, _ = example()
    data = json.loads(physical_model(K.tolist(), M.tolist(), [1., .3, -.1], 3,
                                     initial_field=[.7, -.3, .5]))
    assert data['initial_projection']['residual_quadratic_storage'] < 1e-25
    assert data['initialized_lti']['initial_state'] == data['initial_projection']['initial_state']
    assert 'data.x0 = [' in data['matlab_function']


def test_linear_heat_content_is_reported_separately():
    K, M, _ = example()
    weights = np.array([1., 2, 4])
    model = physical_cln(K, M, [1., .3, -.1], 1, storage_integral=weights)
    p = model.project_initial([.7, -.3, .5])
    assert p.diagnostics['full_linear_storage_content'] == pytest.approx(2.1)
    assert p.diagnostics['retained_linear_storage_content'] != pytest.approx(2.1)
