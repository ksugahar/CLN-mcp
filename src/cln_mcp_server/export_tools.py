"""Return importable LTspice and MATLAB artifacts without executing or writing them."""
import json


def export_model(resistance, storage, kind="rc", termination="auto", s0=0.0,
                 length=1.0, quantity="impedance", name="cln_model"):
    from cln.export import rc_lti, rl_lti
    if not 1 <= len(resistance) <= 64 or len(storage) > 64:
        raise ValueError("export supports 1..64 resistance elements and up to 64 storage elements")
    if kind == "rc":
        if s0 != 0 or length != 1 or quantity != "impedance":
            raise ValueError("RC input elements are already scaled; use s0=0, length=1 and quantity='impedance'")
        model = rc_lti(resistance, storage, termination="capacitance" if termination == "auto" else termination)
    elif kind == "rl":
        if termination == "auto":
            termination = "resistance" if len(resistance) == len(storage)+1 else "inductance"
        if termination == "resistance" and len(resistance) == len(storage)+1:
            nmax = 2*len(storage)
        elif termination == "inductance" and len(resistance) == len(storage):
            nmax = 2*len(storage)-1
        else:
            raise ValueError("RL resistance termination needs len(R)=len(L)+1; inductance termination needs len(R)=len(L)")
        model = rl_lti({2*i: r for i, r in enumerate(resistance)},
                       {2*i+1: l for i, l in enumerate(storage)}, s0=s0, nmax=nmax,
                       length=length, quantity=quantity)
    else:
        raise ValueError("kind must be 'rc' or 'rl'")
    result = model.to_dict()
    result["files"] = {f"{name}.m": model.matlab_function(name),
                       f"{name}.lib": model.ltspice_subcircuit(name),
                       f"{name}_signal.lib": model.ltspice_subcircuit(name+"_signal", "signal")}
    result["usage"] = {"matlab": f"[cln_sys, cln_data] = {name}();",
                       "simulink": "LTI System: cln_sys; Initial states: cln_data.x0. State-Space alternative: cln_data.A/B/C/D.",
                       "ltspice_one_port": f".include {name}.lib\nXcln p 0 {name}",
                       "ltspice_signal": f".include {name}_signal.lib\nXcln input 0 output 0 {name}_signal",
                       "notes": "One-port always presents impedance V/I; signal subcircuit follows quantity. Zero initial state for Laplace. Positive s0 is preserved, not a stability guarantee."}
    return json.dumps(result, allow_nan=False)
