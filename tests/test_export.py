import json
import numpy as np
import pytest
from scipy.linalg import expm
from cln import rc_lti, rl_lti, physical_cln, ladder_impedance, rc_ladder_impedance


@pytest.mark.parametrize("termination", ["capacitance", "resistance"])
@pytest.mark.parametrize("n", [1, 2, 4])
def test_rc_states_and_laplace_match_circuit(termination, n):
    r = np.arange(1., n+1)
    c = np.arange(2., n+2)
    model = rc_lti(r, c, termination=termination)
    for s in (0., .1, 1j, 10j, 3+4j):
        ref = rc_ladder_impedance(r, c, s, termination=termination)
        assert model.response(s) == pytest.approx(ref, rel=1e-11)
        assert eval(model.expression, {"s": s}) == pytest.approx(ref, rel=1e-11)
    assert model.to_dict()["asymptotically_stable"]


@pytest.mark.parametrize("quantity", ["impedance", "admittance"])
@pytest.mark.parametrize("s0", [0., 2.])
@pytest.mark.parametrize("nmax", [0, 1, 2, 3, 4, 6])
def test_rl_states_and_laplace_match_circuit(quantity, s0, nmax):
    r = {0: 1., 2: 2., 4: 3., 6: 4.}
    ell = {1: .2, 3: .3, 5: .4}
    if nmax % 2 and quantity == "impedance":
        with pytest.raises(ValueError, match="improper"):
            rl_lti(r, ell, s0=s0, nmax=nmax)
        return
    model = rl_lti(r, ell, s0=s0, nmax=nmax, length=2.5, quantity=quantity)
    for s in (.1j, 1j, 10j, 3+4j):
        z = 2.5*ladder_impedance(r, ell, s, s0, nmax)
        ref = z if quantity == "impedance" else 1/z
        assert model.response(s) == pytest.approx(ref, rel=1e-10)
        assert eval(model.expression, {"s": s}) == pytest.approx(ref, rel=1e-10)


def test_energy_model_lti_preserves_state_coordinates_and_transient():
    energy = physical_cln([[2., -1.], [-1., 2.]], [[2., 0.], [0., 1.]], [1., 0.], 2)
    model = energy.to_lti()
    v0 = np.array([1., .2])
    force = .4
    np.testing.assert_allclose(model.A @ v0 + model.B[:, 0]*force, energy.balance(v0, force)["node_rates"])
    for t in (0., .1, 1., 4.):
        v = expm(model.A*t) @ v0
        assert (model.C @ v).item() == pytest.approx(v[0])
        assert energy.balance(v)["quadratic_storage"] <= energy.balance(v0)["quadratic_storage"]


def test_shift_not_stabilized_or_applied_twice():
    model = rl_lti({0: 1., 2: 2.}, {1: 1.}, s0=4.)
    assert model.A[0, 0] == pytest.approx(2.)
    assert not model.to_dict()["asymptotically_stable"]
    assert model.response(4.) == pytest.approx(1.)


def test_export_bundle_and_overwrite_guard(tmp_path):
    model = rc_lti([2., 3.], [4., 5.])
    paths = model.write(tmp_path, "thermal_cln")
    assert len(paths) == 4
    payload = json.loads((tmp_path/"thermal_cln.json").read_text())
    np.testing.assert_allclose(payload["A"], model.A)
    code = (tmp_path/"thermal_cln.m").read_text()
    assert "function [sys, data] = thermal_cln()" in code
    assert "ss(data.A, data.B, data.C, data.D)" in code
    with pytest.raises(FileExistsError):
        model.write(tmp_path, "thermal_cln")


@pytest.mark.parametrize("name", ["../bad", "a\n.end", "1model", "a-b", "end"])
def test_names_cannot_inject_generated_code(name):
    with pytest.raises(ValueError):
        rc_lti([1.], [1.]).matlab_function(name)


@pytest.mark.parametrize("r,c", [([], []), ([0.], [1.]), ([1.], [-1.]), ([1.], [float('nan')]), ([1.,2.], [])])
def test_bad_rc_elements(r, c):
    with pytest.raises(ValueError):
        rc_lti(r, c)


def test_mcp_ordered_elements_and_counts():
    from cln_mcp_server.server import cln_export_lti
    bundle = json.loads(cln_export_lti([2., 3.], [.1], kind="rl", name="wire"))
    model = rl_lti({0: 2., 2: 3.}, {1: .1})
    np.testing.assert_allclose(bundle["A"], model.A)
    assert "wire.m" in bundle["files"]
    with pytest.raises(ValueError, match="improper"):
        cln_export_lti([2.], [.1], kind="rl")
    with pytest.raises(ValueError, match="termination"):
        cln_export_lti([2., 3., 4.], [.1], kind="rl")
    with pytest.raises(ValueError, match="already scaled"):
        cln_export_lti([2.], [.1], kind="rc", length=2.)
