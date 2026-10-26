# Importing a reduced CLN into LTspice and Simulink

Install the NumPy-based exporter with `pip install -e ".[energy]"`.
Exporters derive state equations from the ladder and preserve its continued
fraction. They do not fit frequency samples or expand a high-degree polynomial.
Time is in seconds, with Laplace variable `s` in inverse seconds.

## Export an existing result

For the physical diffusion model:

```python
from cln import physical_cln, rc_lti

energy = physical_cln(K, M, load, nstages=6)
lti = energy.to_lti()
paths = lti.write("exports", "thermal_cln")
# Or, from already scaled arrays produced by DiffusionCLN.run():
lti = rc_lti(result["R"], result["C"], termination="capacitance")
```

For the electromagnetic NGSolve result:

```python
from cln import rl_lti

R, L, ratios = solver.run(nmax=6)  # solver is TOmegaCLN
lti = rl_lti(R, L, s0=solver.s0, length=solver.length, quantity="impedance")
lti.write("exports", "wire_cln")
```

`TOmegaCLN` returns elements per unit length. The exporter multiplies **both R
and L** by `length` exactly once. Diffusion R/C arrays already include their
extrusion scaling and must not be scaled again. Specify the same `nmax` if
exporting only a prefix of an RL result.

The bundle contains:

| File | Purpose |
|---|---|
| `thermal_cln.lib` | LTspice two-terminal physical one-port, V/I = Z(s) |
| `thermal_cln_signal.lib` | LTspice four-terminal signal transfer model |
| `thermal_cln.m` | MATLAB function returning `ss` and A/B/C/D/initial state |
| `thermal_cln.json` | Numeric matrices, port direction, state coordinates, poles and metadata |

Existing files are refused unless `overwrite=True` is specified. Run
`python examples/export_cln.py exports` for a complete algebraic example.

## LTspice: connecting the Laplace element

Place the `.lib` beside the circuit file. Add a SPICE directive and an X instance
(or create a two-pin symbol with Prefix `X` and Value `thermal_cln`):

```spice
* One-port example: current enters p, voltage is V(p,0)
.include thermal_cln.lib
Xcln p 0 thermal_cln
Iinput 0 p AC 1
.ac dec 40 0.01 100
.end
```

The two pins are `p n`; positive current flows from `p` to `n`. RL exports use
a VCCS with `Laplace=1/Z(s)`. RC exports use a zero-volt current sensor and a
VCVS with `Laplace=Z(s)`, avoiding differentiation of the port voltage through
the improper RC admittance. Circuit impedance remains V/I regardless of the
chosen signal-model `quantity`.

For a signal-chain model, use:

```spice
.include thermal_cln_signal.lib
Vin in 0 AC 1
Xcln in 0 out 0 thermal_cln_signal
.ac dec 40 0.01 100
.end
```

Signal pins are `in_p in_n out_p out_n`, and the E source enforces
`V(out_p,out_n) = H(s)*V(in_p,in_n)`. With `quantity="impedance"`, the input
voltage numerically represents current (or heat input), and the output voltage
represents the conjugate potential (or excess temperature). These are analog
signal units, not additional physical electrical connections.

All Laplace exports represent zero internal initial state. Verify `.tran`
accuracy and time step over the intended bandwidth: LTspice's Laplace transient
implementation is numerical. Native ladder elements remain useful when
individual branch initial conditions are required. See Analog Devices'
[Laplace modeling guide](https://www.analog.com/en/resources/technical-articles/model-transfer-functions-by-applying-the-laplace-transform-in-ltspice.html).

## Simulink: LTI System block

Add the export directory to the MATLAB path and run:

```matlab
[cln_sys, cln_data] = thermal_cln();
```

The function constructs a continuous-time `ss(A,B,C,D)` object. In Simulink's
**LTI System** block (Control System Toolbox), set:

| Block parameter | Value |
|---|---|
| LTI system variable | `cln_sys` |
| Initial states (state-space only) | `cln_data.x0` |

Connect the input signal to the block and its output to the receiving system or
Scope. Keep the initialization function on the MATLAB path and execute it
before loading/simulating the model, or put the same assignment in that model's
initialization script. Generated functions do not modify existing models or
the base workspace themselves. No saved `.slx` is required to import the LTI.

For the standard **State-Space** block, use `cln_data.A`, `cln_data.B`,
`cln_data.C`, `cln_data.D`, and `cln_data.x0` in its corresponding fields.
The provided factory uses Control System Toolbox; numeric JSON matrices are
also available for a workflow using only the standard State-Space block.

Validation commands:

```matlab
pole(cln_sys)
isstable(cln_sys)
freqresp(cln_sys, 2*pi*[1 10 100])   % angular frequencies in rad/s
lsim(cln_sys, input_signal, time_seconds, cln_data.x0)
```

MathWorks documents the [LTI System block](https://www.mathworks.com/help/control/ref/ltisystem.html)
and [`ss` model](https://www.mathworks.com/help/control/ref/ss.html).

## State coordinates, termination and expansion point

- Full RC termination retains every capacitor. States are **node potentials**,
  so `energy.reconstruct(x)` and the energy balance use the same coordinates.
  The signal is injected flux -> weighted potential (heat: power -> temperature).
- Legacy `termination="resistance"` omits the final capacitor. Its states are
  the remaining capacitor drops, and its finite feedthrough is retained.
  It is not the full energy-certified model; do not reuse its states as node
  potentials. A one-resistor model correctly has zero states.
- At zero shift, RL states are physical shunt-inductor currents. At nonzero
  `s0`, they are currents of the **shifted** ladder, whose branches obey
  `v=L*(di/dt-s0*i)` (an inductor in series with a `-s0*L` resistance).
  They are not automatically a projection of the original FE field's initial
  condition. Use zero initial states unless that mapping is explicitly supplied.
  With resistance termination,
  both impedance and admittance have ordinary state-space realizations.
  With inductance termination, impedance includes an input derivative and is
  **improper**. Request `quantity="admittance"` to obtain a causal ordinary LTI
  realization. The exporter rejects an improper impedance instead of removing
  that derivative or changing the termination.
- `s0` is a **real Laplace expansion point**, not a sinusoidal angular frequency.
  `TOmegaCLN` generates coefficients only for nonnegative `s0`. The algebraic
  exporter also accepts finite negative real shifts for externally supplied
  coefficients; this does not extend the NGSolve solver's supported domain.
  The shifted dynamics use `s-s0`; the state matrix includes `+s0*I` exactly once.
  Do not shift the exported LTI again. A truncated shifted model is not promised
  stable/passive over all frequencies merely because its listed R/L are positive.
  Poles and stability are reported; no stabilization or pole cancellation is
  silently applied. Validate against the original FE model on the intended
  `s=j*2*pi*f` band. Agreement at real `s=s0` alone does not establish that.

Initial states default to zero. For nonzero states, use the documented physical
coordinates; pole/zero or balanced realizations could change their meaning.
No `minreal`, balancing, discretization or state transformation is applied.

For reproducible block simulations, bind the exported variables explicitly in
the model workspace or with `Simulink.SimulationInput.setVariable(...,
'Workspace', modelName)`. Bind the `ss` object and initial state as well as the
numeric A/B/C/D when comparing LTI System and State-Space blocks. This avoids
silently resolving an older model from the base workspace. RL models can be
stiff; verify time-step/tolerance convergence over the intended signal bandwidth.

## MCP usage

`cln_export_lti` accepts ordered arrays rather than JSON integer-key dictionaries:

```json
{"resistance": [2, 3], "storage": [0.0001, 0.0002],
 "kind": "rc", "termination": "capacitance", "name": "thermal_cln"}
```

For RL, `storage` holds L1,L3,...; set `kind="rl"`, the original `s0`, `length`,
and `quantity`. `termination="auto"` infers RL termination from element counts
and chooses full capacitor termination for RC. The response returns the file
**contents**, state matrices, poles, and usage instructions. It does not execute
MATLAB, run a solver, or write to the caller's filesystem.

## Initial finite-element fields

See [initial field projection](initial_fields.md) for capacity/mu-metric projection, storage diagnostics and initialized MATLAB factories. The physical RC and inductance-terminated magnetic APIs retain the required field modes; an element-only export cannot infer this map.
