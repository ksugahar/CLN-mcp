"""High-order element-wise curl recovery with shared trace compatibility.

Research prototype, not a replacement for the CLN field solver. A single
greedy local sweep and a compatible iterative element operator are compared
against an independently assembled curl-curl pseudoinverse. No mass gauge.
"""
import argparse
import hashlib
import json
import platform
import socket
from pathlib import Path

import ngsolve as ng
import numpy as np
from netgen.occ import Box, OCCGeometry, Pnt
from scipy.sparse.linalg import LinearOperator, lsmr


def run(mesh, order, seed):
    fes = ng.HCurl(mesh, order=order, nograds=False)
    u, v = fes.TnT()
    bfi = ng.SymbolicBFI(ng.curl(u)*ng.curl(v), bonus_intorder=6)
    assembled = ng.BilinearForm(fes, symmetric=True)
    assembled += bfi
    assembled.Assemble()
    blocks = []
    for el in fes.Elements():
        ids = np.array(fes.GetDofNrs(el), dtype=int)
        assert np.all(ids >= 0)
        k = np.array(bfi.CalcElementMatrix(el.GetFE(), el.GetTrafo()))
        w, q = np.linalg.eigh((k+k.T)/2)
        keep = w > w.max()*1e-11
        c = np.sqrt(w[keep])[:, None]*q[:, keep].T
        blocks.append((ids, c))
    count = sum(len(c) for _, c in blocks)

    def apply(x):
        return np.concatenate([c@x[ids] for ids, c in blocks])

    def transpose(y):
        out = np.zeros(fes.ndof)
        start = 0
        for ids, c in blocks:
            np.add.at(out, ids, c.T@y[start:start+len(c)])
            start += len(c)
        return out

    op = LinearOperator((count, fes.ndof), matvec=apply, rmatvec=transpose)
    target = np.random.default_rng(seed).normal(size=fes.ndof)
    rhs = apply(target)  # exactly compatible curl; includes high-order modes
    offsets = np.cumsum([0]+[len(c) for _, c in blocks])
    sweep_errors = {}
    permutations = dict(forward=np.arange(len(blocks)), reverse=np.arange(len(blocks))[::-1],
                        shuffled=np.random.default_rng(seed+1).permutation(len(blocks)))
    for name, sequence in permutations.items():
        greedy = np.zeros(fes.ndof)
        fixed = np.zeros(fes.ndof, dtype=bool)
        for j in sequence:
            ids, c = blocks[j]
            free = ~fixed[ids]
            residual = rhs[offsets[j]:offsets[j+1]]-c@greedy[ids]
            if free.any():
                greedy[ids[free]] += np.linalg.lstsq(c[:, free], residual, rcond=1e-11)[0]
            fixed[ids] = True
        sweep_errors[name] = float(np.linalg.norm(apply(greedy)-rhs)/np.linalg.norm(rhs))
    assert max(sweep_errors.values()) < 1e-8, sweep_errors
    result = lsmr(op, rhs, atol=1e-13, btol=1e-13, maxiter=10000)
    solution = result[0]
    rows, cols, vals = assembled.mat.COO()
    global_k = np.zeros((fes.ndof, fes.ndof))
    np.add.at(global_k, (rows, cols), vals)
    matrix_defect = np.linalg.norm(global_k@target-transpose(rhs))/np.linalg.norm(global_k@target)
    w, q = np.linalg.eigh(global_k)
    keep = w > w.max()*1e-11
    reference = q[:, keep]@((q[:, keep].T@(global_k@target))/w[keep])
    relative = lambda x: float(np.linalg.norm(x)/np.linalg.norm(rhs))
    original_residual = float(np.linalg.norm(global_k@(solution-target))/np.linalg.norm(global_k@target))
    gf = ng.GridFunction(fes)
    gf.vec.FV().NumPy()[:] = solution-target
    field_error = float(ng.Integrate(ng.curl(gf)*ng.curl(gf), mesh))**0.5/np.linalg.norm(rhs)
    incompatible = np.random.default_rng(seed+2).normal(size=count)
    bad = lsmr(op, incompatible, atol=1e-13, btol=1e-13, maxiter=10000)[0]
    incompatible_error = float(np.linalg.norm(apply(bad)-incompatible)/np.linalg.norm(incompatible))
    record = dict(order=order, seed=seed, ndof=fes.ndof, elements=mesh.ne,
                  local_curl_ranks=[len(c) for _, c in blocks],
                  greedy_curl_errors=sweep_errors, integrated_curl_error=field_error,
                  incompatible_curl_error=incompatible_error,
                  iterative_curl_error=relative(apply(solution)-rhs),
                  assembled_reference_curl_error=relative(apply(reference)-rhs),
                  iterative_reference_curl_difference=relative(apply(solution-reference)),
                  original_residual=original_residual, element_assembly_defect=float(matrix_defect),
                  lsmr_stop=int(result[1]), iterations=int(result[2]))
    assert record['iterative_curl_error'] < 1e-8, record
    assert record['original_residual'] < 1e-8, record
    assert record['iterative_reference_curl_difference'] < 1e-8, record
    assert matrix_defect < 1e-10, record
    assert field_error < 1e-8 and incompatible_error > 1e-3, record
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    ng.SetNumThreads(4)
    mesh = ng.Mesh(OCCGeometry(Box(Pnt(0, 0, 0), Pnt(1, 1, 1))).GenerateMesh(maxh=2))
    records = []
    with ng.TaskManager():
        for order in (1, 2, 3):
            for seed in (17, 29):
                record = run(mesh, order, seed)
                records.append(record)
                print(json.dumps(record), flush=True)
    source = Path(__file__).read_text(encoding='utf-8-sig').encode()
    data = dict(schema=1, scope='compatible discrete curl recovery on an affine simply connected cube; not CLN or speed evidence',
                source_sha256=hashlib.sha256(source).hexdigest(), host=socket.gethostname(),
                python=platform.python_version(), ngsolve=ng.__version__, numpy=np.__version__,
                threads=4, cases=records)
    Path(args.output).write_text(json.dumps(data, indent=2)+'\n', encoding='utf8')
