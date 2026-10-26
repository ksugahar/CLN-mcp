import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg import expm

from cln.energy import EnergyCLN, physical_cln, _certified_prefix
from cln_mcp_server.server import cln_diffusion_physical_model, cln_diffusion_energy_formulation

K = np.array([[4., -1., 0.], [-1., 3., -1.], [0., -1., 2.]])
M = np.diag([2., 1., 3.])
F = np.array([1., .3, .2])


@pytest.fixture
def model():
    return physical_cln(K, M, F, 3, storage_integral=M @ np.ones(3))


def test_field_circuit_identities_and_energy_balance(model):
    E, H, B = model.basis, model.odd_basis, model.B
    np.testing.assert_allclose(E, H @ B, atol=2e-13)
    np.testing.assert_allclose(E.T @ K @ E, model.G, atol=1e-9)
    np.testing.assert_allclose(H.T @ M @ H, np.diag(model.C), atol=1e-9)
    np.testing.assert_allclose(E.T @ M @ E, model.M, atol=1e-9)
    np.testing.assert_allclose(E.T @ F, model.port, atol=1e-11)
    v = np.array([1., .2, -.1])
    u = E @ v
    balance = model.balance(v, .5)
    assert balance["quadratic_storage"] == pytest.approx(.5*u @ M @ u)
    assert balance["dissipation"] == pytest.approx(u @ K @ u)
    assert balance["linear_storage_content"] == pytest.approx(np.ones(3) @ M @ u)
    assert abs(balance["balance_residual"]) < 1e-10
    assert abs(balance["rate_tested_balance"]["balance_residual"]) < 1e-10
    assert not np.isclose(balance["linear_storage_content"], np.dot(model.C, B @ v))
    assert not np.allclose(E @ v, H @ v)  # odd modes need capacitor drops!


@pytest.mark.parametrize("s", [0., 1., 1j, 10j, 100j])
def test_full_dimension_frequency_matches_direct_system(model, s):
    assert model.impedance(s) == pytest.approx(F @ np.linalg.solve(K+s*M, F), rel=1e-10)


def test_transient_reconstruction_and_free_decay(model):
    v0 = np.array([.2, -.1, .02])
    u0 = model.reconstruct(v0)
    last_energy = np.inf
    for t in np.linspace(0., 2., 15):
        v = expm(-np.linalg.solve(model.M, model.G)*t) @ v0
        u = expm(-np.linalg.solve(M, K)*t) @ u0
        np.testing.assert_allclose(model.reconstruct(v), u, atol=2e-12)
        energy = model.balance(v)["quadratic_storage"]
        assert energy <= last_energy + 1e-12
        last_energy = energy


def test_midpoint_energy_balance(model):
    v0 = np.array([.2, -.1, .02])
    dt, drive = .01, .7
    v1 = np.linalg.solve(model.M + dt*model.G/2,
                         (model.M - dt*model.G/2) @ v0 + dt*model.port*drive)
    midpoint = (v1+v0)/2
    rate = (model.balance(v1)["quadratic_storage"] - model.balance(v0)["quadratic_storage"])/dt
    assert rate == pytest.approx(drive*midpoint[0] - midpoint @ model.G @ midpoint, abs=1e-11)


def test_same_space_congruence_and_first_2n_moments():
    k = np.diag([1., 2., 3., 4., 5.]); m = np.eye(5); f = np.ones(5)
    n = 2
    model = physical_cln(k, m, f, n)
    b0 = np.linalg.solve(k, f)
    Q, _ = np.linalg.qr(np.column_stack([b0, np.linalg.solve(k, m @ b0)]))
    gq, mq, bq = Q.T @ k @ Q, Q.T @ m @ Q, Q.T @ f
    for s in (0., .5, 2j, 10j):
        assert model.impedance(s) == pytest.approx(bq @ np.linalg.solve(gq+s*mq, bq), rel=1e-12)
    full = b0
    reduced = np.linalg.solve(model.G, model.port)
    for _ in range(2*n):
        assert f @ full == pytest.approx(model.port @ reduced, rel=1e-12)
        full = np.linalg.solve(k, m @ full)
        reduced = np.linalg.solve(model.G, model.M @ reduced)
    assert not np.isclose(f @ full, model.port @ reduced, rtol=1e-8, atol=0)


def test_physical_network_matches_independent_mna(model):
    # Assemble the actual emitted R/C network without using its circuit matrices.
    net = [line.split() for line in model.netlist().splitlines() if not line.startswith("*")]
    nodes = ["p", "n1", "n2"]
    for s in (0., 1j, 12j):
        Y = np.zeros((3, 3), dtype=complex)
        for name, a, b, value in net:
            admittance = 1/float(value) if name.startswith("R") else s*float(value)
            for i, j, sign in ((a, a, 1), (b, b, 1), (a, b, -1), (b, a, -1)):
                if i != "0" and j != "0":
                    Y[nodes.index(i), nodes.index(j)] += sign*admittance
        assert np.linalg.solve(Y, [1, 0, 0])[0] == pytest.approx(model.impedance(s))


def test_termination_changes_dynamics():
    from cln.diffusion import rc_ladder_impedance
    model = physical_cln([[2.]], [[3.]], [1.], 6)
    assert model.R == pytest.approx([.5])
    assert model.C == pytest.approx([3.])
    assert model.metadata["effective_stages"] == 1
    assert model.impedance(2.) == pytest.approx(1/8)
    assert rc_ladder_impedance(model.R, model.C, 2.) == .5
    assert rc_ladder_impedance(model.R, model.C, 2., termination="capacitance") == pytest.approx(1/8)


def test_source_scaling_and_invariant_subspace():
    small = physical_cln(np.eye(3), np.eye(3), [1, 1, 1], 3)
    assert small.metadata["effective_stages"] == 1
    assert small.metadata["stop_reason"] == "invariant_subspace_breakdown"
    original = physical_cln(K, M, F, 2)
    scaled = physical_cln(K, M, 5*F, 2)
    np.testing.assert_allclose(scaled.R, 25*original.R)
    np.testing.assert_allclose(scaled.C, original.C/25)
    assert scaled.impedance(1j) == pytest.approx(25*original.impedance(1j))
    signed = physical_cln(K, M, [1., -1., 0.], 3)
    assert signed.certificate["passed"]  # a zero-sum source is a valid SISO port


@pytest.mark.parametrize("k,m,f", [
    ([[1, 2], [0, 1]], np.eye(2), [1, 0]),
    ([[1, -1], [-1, 1]], np.eye(2), [1, 0]),
    (np.eye(2), [[1, 0], [0, -1]], [1, 0]),
    (np.eye(2), np.eye(2), [0, 0]),
    ([[np.nan]], [[1]], [1]),
    (np.eye(2), np.eye(3), [1, 0]),
])
def test_unsupported_systems_are_rejected(k, m, f):
    with pytest.raises(ValueError):
        physical_cln(k, m, f)


def test_false_energy_certificate_is_rejected():
    with pytest.raises(ValueError, match="certificate failed"):
        EnergyCLN([1.], [1.], [[1.]], [[2.]], [1.], [[1.]])


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_energy_certificate_survives_extreme_metric_scales(scale):
    with pytest.raises(ValueError, match="certificate failed"):
        EnergyCLN([1/scale], [1.], [[.1*scale]], [[1.]], [1.], [[1.]])
    valid = EnergyCLN([1/scale], [1.], [[scale]], [[1.]], [1.], [[1.]])
    assert valid.certificate["passed"]


def test_nonfinite_circuit_matrices_are_not_certified():
    with pytest.raises(ValueError, match="floating-point range"):
        EnergyCLN([1e-320], [1.], [[1.]], [[1.]], [1.], [[1.]])


def test_large_finite_spd_input_is_not_overflowed_by_symmetrization():
    model = physical_cln([[1e308]], [[1.]], [1.], 1)
    assert model.certificate["passed"]
    assert model.R[0] == pytest.approx(1e-308, rel=1e-12, abs=0)


def test_failed_extra_stages_are_reported_not_certified():
    # Emulate an accumulated last-mode error. Only the first state is certified.
    kg = np.diag([1., 1.]); mg = np.array([[1., -1.], [-1., 2.]])
    model = _certified_prefix([1., 1.], [1., 1.], kg, mg, np.array([1., .1]), np.eye(2))
    assert model.R.size == 1
    assert model.certificate["passed"]
    assert model.certificate["rejected_larger_prefixes"]["2"]["collocated_port"] == .1


def test_mcp_returns_real_energy_model_and_units():
    result = json.loads(cln_diffusion_physical_model(K.tolist(), M.tolist(), F.tolist(), 3,
                        storage_integral=(M @ np.ones(3)).tolist(), nodes=[1., .2, -.1], drive=.5))
    assert result["certificate"]["passed"]
    assert result["units"]["quadratic_storage"] == "J K"
    assert result["units"]["linear_storage_content"] == "J"
    assert result["termination"] == "capacitor_to_reference"
    assert result["summary"]["effective_stages"] == 3
    assert "even_basis" in result and "odd_basis" in result
    assert abs(result["state"]["balance_residual"]) < 1e-10
    text = cln_diffusion_energy_formulation()
    assert "PRIMA can preserve" in text
    short = json.loads(cln_diffusion_physical_model([[2.]], [[3.]], [1.], 4))
    assert short["summary"]["effective_stages"] == 1
    assert "Fewer stages" in short["summary"]["notice"]


def test_matrix_only_api_does_not_import_ngsolve():
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/"src"))
    code = "from cln import physical_cln; import sys; physical_cln([[2]],[[3]],[1],1); assert 'ngsolve' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.parametrize("count", [0, -1, True, 1.5, 65])
def test_mcp_bounds(count):
    with pytest.raises(ValueError):
        cln_diffusion_physical_model([[1]], [[1]], [1], nstages=count)
