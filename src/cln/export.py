"""Continuous-time SISO exports of RC/RL CLNs, without polynomial expansion.

State-space realizations come from the circuit equations, not curve fitting.
LTspice uses the original continued fraction to avoid high-order polynomials.
"""
from dataclasses import dataclass, replace
import json
from numbers import Integral
from pathlib import Path
import re

import numpy as np


def _positive(values, label):
    if np.iscomplexobj(values):
        raise ValueError(f"{label} must be real")
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{label} must be a vector of finite positive elements")
    return values


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", value):
        raise ValueError("name must start with a letter and contain only letters, digits, underscores (max 48)")
    # Used as a MATLAB function name as well as a SPICE identifier.
    if value.lower() in {"end", "if", "else", "elseif", "for", "while", "function", "classdef",
                         "switch", "case", "otherwise", "try", "catch", "global", "persistent",
                         "return", "break", "continue", "parfor", "spmd"}:
        raise ValueError("name must not be a MATLAB keyword")
    return value


def _num(value):
    return format(float(value), ".17g")


def _matrix_text(matrix):
    if not matrix.size:
        return f"zeros({matrix.shape[0]},{matrix.shape[1]})"
    return "[" + "; ".join(" ".join(_num(v) for v in row) for row in matrix) + "]"


@dataclass
class CLNLTI:
    """xdot=A*x+B*u, y=C*x+D*u, time in seconds; zero-state Laplace response."""
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    expression: str
    impedance_expression: str
    metadata: dict
    initial_state: np.ndarray | None = None

    def __post_init__(self):
        self.A, self.B, self.C, self.D = (np.asarray(x, dtype=float) for x in (self.A, self.B, self.C, self.D))
        n = self.A.shape[0]
        if (self.A.shape != (n, n) or self.B.shape != (n, 1) or
                self.C.shape != (1, n) or self.D.shape != (1, 1)):
            raise ValueError("invalid SISO state-space dimensions")
        if any(not np.all(np.isfinite(x)) for x in (self.A, self.B, self.C, self.D)):
            raise ValueError("state-space matrices exceed the supported floating-point range")
        if self.initial_state is None:
            self.initial_state = np.zeros(n)
        if np.iscomplexobj(self.initial_state):
            raise ValueError("initial_state must be real")
        self.initial_state = np.asarray(self.initial_state, dtype=float).copy()
        if self.initial_state.shape != (n,) or not np.all(np.isfinite(self.initial_state)):
            raise ValueError("initial_state must be a finite vector matching the state dimension")

    def with_initial_state(self, initial_state):
        """Return a copy carrying initial state in these exact realization coordinates."""
        return replace(self, initial_state=initial_state, metadata=dict(self.metadata))

    def response(self, s):
        """Evaluate at complex Laplace variables, e.g. s=2j*pi*frequency_hz."""
        values = np.asarray(s, dtype=complex)
        if not np.all(np.isfinite(values)):
            raise ValueError("s must be finite")
        eye = np.eye(self.A.shape[0])
        result = np.array([(self.C @ np.linalg.solve(z*eye-self.A, self.B)+self.D)[0, 0]
                           for z in values.flat]).reshape(values.shape)
        return result[()] if result.ndim == 0 else result

    def to_dict(self):
        poles = np.linalg.eigvals(self.A)
        return {"A": self.A.tolist(), "B": self.B.tolist(), "C": self.C.tolist(),
                "D": self.D.tolist(), "states": self.A.shape[0], "sample_time": 0,
                "time_unit": "seconds", "metadata": self.metadata,
                "poles": [[float(p.real), float(p.imag)] for p in poles],
                "asymptotically_stable": bool(np.all(poles.real < 0)),
                "laplace_expression": self.expression,
                "initial_state": self.initial_state.tolist()}

    def ltspice_subcircuit(self, name="cln_model", interface="one_port"):
        """A two-terminal impedance, or a four-terminal voltage signal transfer.

        Signal pins: in_p in_n out_p out_n. One-port pins: p n, current p->n.
        RC uses a sensed-current/VCVS impedance; RL uses a VCCS admittance.
        The latter also supports inductance-terminated (improper Z) RL ladders.
        """
        if np.any(self.initial_state):
            raise ValueError("LTspice Laplace export cannot encode nonzero initial state; use MATLAB/Simulink")
        name = _name(name)
        header = ["* CLN Laplace export; s in 1/second; zero internal initial state",
                  "* Continuous-time rational model; verify transient accuracy over the intended bandwidth"]
        if interface == "signal":
            lines = [f".subckt {name} in_p in_n out_p out_n",
                     f"Eresponse out_p out_n in_p in_n Laplace={self.expression}"]
        elif interface == "one_port":
            lines = [f".subckt {name} p n"]
            if self.metadata["kind"] == "rc":
                # Do not differentiate the port voltage through the improper
                # RC admittance. Convert sensed port current to a signal voltage.
                lines += ["Vsense p internal 0", "Fmeasure 0 measured Vsense 1", "Rmeasure measured 0 1",
                          f"Eresponse internal n measured 0 Laplace={self.impedance_expression}"]
            else:
                lines += [f"Gresponse p n p n Laplace=(1/({self.impedance_expression}))"]
        else:
            raise ValueError("interface must be 'one_port' or 'signal'")
        return "\n".join(header + lines + [f".ends {name}", ""])

    def matlab_function(self, name="cln_model"):
        """Standalone .m function returning ss plus matrices/initial state.

        No base-workspace mutation, model edits, discretization or minreal.
        """
        name = _name(name)
        lines = [f"function [sys, data] = {name}()",
                 "% CLN continuous-time LTI model (requires Control System Toolbox).",
                 f"% Simulink: [cln_sys, cln_data] = {name}();",
                 "% LTI System block: LTI system variable = cln_sys; Initial states = cln_data.x0.",
                 "% State-Space block alternative: cln_data.A/B/C/D and cln_data.x0.",
                 "% LTspice Laplace exports have zero initial state; this ss preserves the stated coordinates.",
                 f"% State coordinates: {self.metadata['state_coordinates']}",
                 f"% Transfer: {self.metadata['quantity']}; s0 = {_num(self.metadata.get('s0', 0))} 1/s."]
        for key in ("A", "B", "C", "D"):
            lines.append(f"data.{key} = {_matrix_text(getattr(self, key))};")
        lines += [f"data.x0 = {_matrix_text(self.initial_state[:, None])};",
                  "sys = ss(data.A, data.B, data.C, data.D);",
                  "sys.TimeUnit = 'seconds';",
                  f"sys.InputName = {{'{self.metadata['input']}'}};",
                  f"sys.OutputName = {{'{self.metadata['output']}'}};",
                  "data.poles = pole(sys);", "data.stable = isstable(sys);",
                  "% A positive expansion shift can give an unstable truncated model; no poles are moved.",
                  "end", ""]
        return "\n".join(lines)

    def write(self, directory, name="cln_model", *, overwrite=False):
        """Write .m/.json; also .lib variants only for zero initial state.

        Refuse stale zero-state Laplace files beside a nonzero-state export.
        """
        name = _name(name)
        directory = Path(directory)
        if np.any(self.initial_state) and any((directory/f).exists() for f in
                                              (f"{name}.lib", f"{name}_signal.lib")):
            raise FileExistsError("zero-state Laplace files already exist; choose a new name for the initialized model")
        artifacts = {f"{name}.m": self.matlab_function(name),
                     f"{name}.json": json.dumps(self.to_dict(), indent=2, allow_nan=False)}
        # A Laplace transfer describes only zero-state response. Never silently
        # drop initial conditions when exporting a field-initialized model.
        if not np.any(self.initial_state):
            artifacts.update({f"{name}.lib": self.ltspice_subcircuit(name),
                              f"{name}_signal.lib": self.ltspice_subcircuit(name+"_signal", "signal")})
        if not overwrite and any((directory/f).exists() for f in artifacts):
            raise FileExistsError("export files already exist; choose a new name or overwrite=True")
        directory.mkdir(parents=True, exist_ok=True)
        for filename, content in artifacts.items():
            (directory/filename).write_text(content, encoding="utf-8")
        return {key: str((directory/key).resolve()) for key in artifacts}


def rc_lti(R, C, *, termination="capacitance"):
    """RC impedance F->u (heat: injected power -> weighted excess temperature).

    Full termination: states are node potentials, matching EnergyCLN.reconstruct.
    Resistance termination omits the final C; states are remaining capacitor drops.
    """
    r, c = _positive(R, "R"), _positive(C, "C")
    n = len(r)
    if not n or termination not in ("capacitance", "resistance"):
        raise ValueError("nonempty R and termination 'capacitance' or 'resistance' required")
    nc = n if termination == "capacitance" else n-1
    if len(c) not in ({n} if nc == n else {n-1, n}):
        raise ValueError("C count does not match the specified RC termination")
    c = c[:nc]
    incidence = np.eye(n)-np.eye(n, k=1)
    port = np.eye(n)[:, :1]
    if nc == n:
        mass = incidence.T @ np.diag(c) @ incidence
        A = -np.linalg.solve(mass, np.diag(1/r))
        B = np.linalg.solve(mass, port)
        cout, d = port.T, np.zeros((1, 1))
        state_coordinates = "node potentials (u = even_basis * x)"
    else:
        incidence = incidence[:nc]
        h = (incidence*r) @ incidence.T
        b = (incidence*r) @ port
        hi = np.linalg.solve(h, np.eye(nc))
        A = -hi/c[:, None]
        B = (hi @ b)/c[:, None]
        cout, d = b.T @ hi, np.array([[1/np.sum(1/r)]])
        state_coordinates = "capacitor drops of resistance-terminated RC network"
    z = f"({_num(r[-1])}/(1+s*{_num(c[-1])}*{_num(r[-1])}))" if nc == n else _num(r[-1])
    for i in range(n-2, -1, -1):
        sc = f"(s*{_num(c[i])})"
        z = f"(1/(1/{_num(r[i])}+{sc}/(1+{sc}*{z})))"
    return CLNLTI(A, B, cout, d, z, z,
                  {"kind": "rc", "quantity": "impedance", "termination": termination,
                   "input": "injected_flux", "output": "weighted_potential", "s0": 0.,
                   "state_coordinates": state_coordinates, "elements_scaled": True})


def rl_lti(R, L, *, s0=0.0, nmax=None, length=1.0, quantity="impedance"):
    """RL CLN from indexed dictionaries R[0,2,...], L[1,3,...].

    Admittance is proper for both terminations. Impedance requires resistance
    termination: an inductance-terminated Z has an input derivative and cannot
    be represented by an ordinary ss/LTI System block.
    """
    if nmax is None:
        if not R:
            raise ValueError("R must include index 0")
        nmax = max(list(R)+list(L))
    if isinstance(nmax, bool) or not isinstance(nmax, Integral) or nmax < 0:
        raise ValueError("nmax must be a nonnegative integer")
    if not np.isfinite(s0) or not np.isfinite(length) or length <= 0:
        raise ValueError("s0 must be finite and length must be finite and positive")
    if quantity not in ("impedance", "admittance"):
        raise ValueError("quantity must be 'impedance' or 'admittance'")
    if nmax % 2 and quantity == "impedance":
        raise ValueError("inductance-terminated impedance is improper; use quantity='admittance' or resistance termination")
    try:
        r = _positive([R[i]*length for i in range(0, nmax+1, 2)], "R")
        ell = _positive([L[i]*length for i in range(1, nmax+1, 2)], "L")
    except KeyError as exc:
        raise ValueError(f"missing ladder element at index {exc.args[0]}") from exc
    m = len(ell)
    h = np.zeros((m, m))
    if m:
        h[0, 0] = 1/r[0]
        for j in range(1, m):
            g = 1/r[j]
            h[j-1, j-1] += g; h[j, j] += g
            h[j-1, j] -= g; h[j, j-1] -= g
        if nmax % 2 == 0:
            h[-1, -1] += 1/r[-1]
    hi = np.linalg.solve(h, np.eye(m))
    drive = np.zeros((m, 1))
    if m:
        drive[0, 0] = 1/r[0]
    A = s0*np.eye(m)-hi/ell[:, None]
    B = (hi @ drive)/ell[:, None]
    cout = drive.T @ hi
    d = np.array([[0. if nmax % 2 else 1/np.sum(r)]])
    if quantity == "impedance":
        A = A-B @ cout/d[0, 0]
        B = B/d[0, 0]
        cout = -cout/d[0, 0]
        d = 1/d
    ds = f"(s-({_num(s0)}))"
    z = _num(r[-1]) if nmax % 2 == 0 else f"({ds}*{_num(ell[-1])})"
    for i in range(nmax-1, -1, -1):
        if i % 2 == 0:
            z = f"({_num(r[i//2])}+{z})"
        else:
            zl = f"({ds}*{_num(ell[i//2])})"
            z = f"(1/(1/{zl}+1/{z}))"
    expression = z if quantity == "impedance" else f"(1/({z}))"
    return CLNLTI(A, B, cout, d, expression, z,
                  {"kind": "rl", "quantity": quantity, "s0": float(s0), "length": float(length),
                   "termination": "resistance" if nmax % 2 == 0 else "inductance",
                   "input": "port_current" if quantity == "impedance" else "port_voltage",
                   "output": "port_voltage" if quantity == "impedance" else "port_current",
                   "state_coordinates": ("physical shunt inductor currents" if s0 == 0 else
                       "shunt inductor currents of the shifted ladder (v=L*(di/dt-s0*i)); no field-state projection"),
                   "elements_scaled": True})
