"""Check recorded evidence without launching a finite-element solver.

Run from any directory: python validation/surface_hybrid/validate_evidence.py
This checks stored results and source integrity, not continuum convergence.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/data"

def read(name):
    return json.loads((DATA / (name + ".json")).read_text(encoding="utf-8-sig"))

def close(a, b):
    assert np.isclose(a, b, rtol=1e-8, atol=1e-12), (a, b)

for name in ("foster_circle_controlled", "foster_3d_coarse", "foster_3d_fine"):
    d = read(name)
    frequencies = np.asarray(d["frequency_hz"])
    assert np.all(np.diff(frequencies) > 0)
    for label, v in d["runs"].items():
        error = np.asarray(v["relative_error"])
        assert error.shape == frequencies.shape
        assert np.all(np.isfinite(error)) and np.all(error >= 0)
        if "poles" in v and not isinstance(v["poles"], int):
            assert np.all(np.asarray(v["poles"]) > 0)
            assert np.all(np.asarray(v["residues"]) > 0)
        if "Foster fitted" in label:
            assert v["optimizer_converged"]
            assert v["DC_offset"] >= 0 and v["L_tail"] >= 0
            if "free" not in label and "band" not in label:
                close(v["DC_offset"], 0)
                close(np.sum(np.asarray(v["residues"]) / v["poles"]) + v["L_tail"], 1)
        if "circuit_elements" in v:
            assert v["circuit_elements_count"] == 2 * v["rank"]
            assert len(v["circuit_elements"]) == v["circuit_elements_count"]
            assert np.all(np.asarray(v["circuit_elements"]) > 0)
            assert v["circuit_vs_ritz"] < 1e-6
            assert v["independent_span_response_difference"] < 1e-6
    if "direct_reference_relative_difference" in d:
        assert d["direct_reference_relative_difference"] < 1e-8
        assert d["mesh"]["kernel_load_fraction"] < 1e-8
        for probe in d["depth_probe"]:
            assert probe["constructed_columns"] == probe["independent_rank"] == probe["requested"]
            assert np.all(np.isfinite(probe["relative_singular_values"]))
    print(name + ": positive models, completed fits and response checks passed")

for name in ("fp_surface_coarse", "fp_surface_fine"):
    d = read(name)
    u = np.asarray(d["input"])
    ref = np.asarray(d["reference"]["lambda_trace"])
    assert len(ref) == len(u) == 400
    sec = slice(199, None)  # includes both endpoints of the second period
    c0 = d["static_air_coefficient"]
    static_work = np.sum(.5 * (u[sec][1:] + u[sec][:-1]) * np.diff(c0 * u[sec]))
    assert abs(static_work) < 1e-10
    rod_ref = ref - c0 * u
    runs = list(d["runs"].values())
    assert [v["basis"] for v in runs] == [3, 5, 7, 9, 19, 55]
    for audit in d["basis_audit"]:
        assert 0 < audit["independent_rank"] <= audit["columns"]
        assert np.all(np.isfinite(audit["relative_singular_values"]))
    for v in runs:
        trace = np.asarray(v["lambda_trace"])
        assert np.all(np.isfinite(trace))
        close(v["max_rel_err"], np.max(abs(trace[sec] - ref[sec])) / np.max(abs(ref[sec])))
        close(v["rod_linkage_relative_error"], np.max(abs((trace-c0*u)[sec]-rod_ref[sec])) / np.max(abs(rod_ref[sec])))
        work = np.sum(.5 * (u[sec][1:] + u[sec][:-1]) * np.diff(trace[sec]))
        close(work, v["input_cycle_work"])
        assert v["joule_cycle"] > 0 and v["joule_reference"] > 0
        close(v["joule_reference"], d["reference"]["joule_cycle"])
        close(abs(v["joule_cycle"]-v["joule_reference"])/v["joule_reference"], v["joule_relative_error"])
    assert runs[-1]["max_rel_err"] < runs[3]["max_rel_err"]
    assert runs[-1]["joule_relative_error"] < runs[3]["joule_relative_error"]
    print(name + ": ranks, trace metrics, Joule errors and reversible-air work passed")

manifest = read("surface_experiment_sources")
for relative, expected in manifest["source_sha256_lf"].items():
    source = (ROOT / relative).read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    actual = hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert actual == expected, relative
print("Source integrity passed (UTF-8, LF-normalized)")
