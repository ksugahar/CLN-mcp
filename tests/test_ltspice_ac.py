"""Run the exported ladder netlists in LTspice (.ac) and compare with ladder_impedance.

Skipped when LTspice or spicelib is not installed (reported as a skip, not a pass)."""
import math
import pathlib
import subprocess
import numpy as np
import pytest

from cln import ladder_netlist, ladder_impedance, wire_cln_s0_zero

LTSPICE = pathlib.Path(r"C:\Program Files\ADI\LTspice\LTspice.exe")
spicelib = pytest.importorskip("spicelib")
pytestmark = pytest.mark.skipif(not LTSPICE.is_file(), reason="LTspice not installed")

A, SIG, MU = 0.01, 1e6, 4e-7*math.pi
R = {n: wire_cln_s0_zero(n, A, SIG, MU) for n in range(0, 7, 2)}
L = {n: wire_cln_s0_zero(n, A, SIG, MU) for n in range(1, 7, 2)}


def _run(tmp_path, netlist, name):
    cir = tmp_path/f"{name}.cir"
    cir.write_text("* CLN ladder AC check\n" + netlist +
                   "I1 0 p AC 1\n.ac dec 10 1e2 1e6\n.end\n", encoding="ascii")
    subprocess.run([str(LTSPICE), "-b", str(cir)], check=True, timeout=120)
    raw = spicelib.RawRead(str(cir.with_suffix(".raw")))
    f = np.real(raw.get_trace("frequency").get_wave())
    v = raw.get_trace("V(p)").get_wave()
    return f, v


@pytest.mark.parametrize("s0", [0.0, 2*math.pi*1e5])
@pytest.mark.parametrize("style", ["rl", "laplace"])
def test_ltspice_ac_matches_ladder(tmp_path, s0, style):
    f, v = _run(tmp_path, ladder_netlist(R, L, s0=s0, nmax=6, style=style), f"{style}_{int(s0)}")
    z = np.array([ladder_impedance(R, L, 2j*math.pi*fk, s0, 6) for fk in f])
    assert np.max(np.abs(v - z)/np.abs(z)) < 1e-4


def test_ltspice_physical_diffusion_matches_full_system(tmp_path):
    from cln import physical_cln
    K = np.array([[2., -1.], [-1., 2.]])
    M = 1e-4*np.diag([2., 1.])
    load = np.array([1., 0.])
    model = physical_cln(K, M, load, 2)
    frequency, voltage = _run(tmp_path, model.netlist(), "physical_diffusion")
    direct = np.array([load @ np.linalg.solve(K+2j*math.pi*f*M, load) for f in frequency])
    assert np.max(np.abs(voltage-direct)/np.abs(direct)) < 1e-4


@pytest.mark.parametrize("interface", ["one_port", "signal"])
@pytest.mark.parametrize("kind", ["rc", "rc_resistance", "rl", "rl_shift", "rl_inductance"])
def test_exported_laplace_subcircuits(tmp_path, interface, kind):
    from cln import rc_lti, rl_lti
    if kind.startswith("rc"):
        model = rc_lti([2., 3.], [1e-4, 2e-4],
                       termination="resistance" if kind.endswith("resistance") else "capacitance")
    else:
        model = rl_lti({0: 2., 2: 3., 4: 4.}, {1: 1e-4, 3: 2e-4},
                       nmax=3 if kind.endswith("inductance") else 4,
                       quantity="admittance" if kind.endswith("inductance") else "impedance",
                       s0=2000. if kind.endswith("shift") else 0.)
    net = model.ltspice_subcircuit("dut", interface)
    if interface == "one_port":
        net += "Xcln p 0 dut\n"
    else:
        net += "Vin in 0 AC 1\nXcln in 0 p 0 dut\n"
    f, voltage = _run(tmp_path, net, f"export_{kind}_{interface}")
    expected = model.response(2j*math.pi*f)
    if interface == "one_port" and model.metadata["quantity"] == "admittance":
        expected = 1/expected
    assert np.max(np.abs(voltage-expected)/np.abs(expected)) < 1e-4


def test_rc_laplace_transient_against_state_space(tmp_path):
    from scipy.integrate import solve_ivp
    from cln import rc_lti
    model = rc_lti([2., 3.], [1e-4, 2e-4])
    cir = tmp_path/'rc_transient.cir'
    cir.write_text('* RC Laplace transient\n'+model.ltspice_subcircuit('dut')+
                   'Xcln p 0 dut\nIinput 0 p PWL(0 0 .001 0 .00101 1 .015 1 .01501 0 .03 0)\n'
                   '.options plotwinsize=0\n.tran 0 .03 0 1e-5\n.end\n', encoding='ascii')
    subprocess.run([str(LTSPICE), '-b', str(cir)], check=True, timeout=120)
    raw = spicelib.RawRead(str(cir.with_suffix('.raw')))
    time = np.abs(raw.get_trace('time').get_wave())
    voltage = raw.get_trace('V(p)').get_wave()
    knots = [0, .001, .00101, .015, .01501, .03]
    forcing = lambda t: np.interp(t, knots, [0, 0, 1, 1, 0, 0])
    sol = solve_ivp(lambda t, x: model.A@x + model.B[:, 0]*forcing(t),
                    (0., float(max(time))), [0., 0.], rtol=1e-10, atol=1e-12,
                    max_step=1e-5, dense_output=True)
    assert sol.success
    expected = (model.C @ sol.sol(time))[0]
    assert np.max(np.abs(voltage-expected)) < .01*np.max(np.abs(expected))
