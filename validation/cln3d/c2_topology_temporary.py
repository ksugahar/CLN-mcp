"""TEMPORARY validation adapters for the proposed Radia topology API.

Same contract as region_cohomology, natural_magnetic_representative and
insulated_loop_currents in the reviewed Radia contribution. Not a maintained
second API. Radia 5.2.3 does not expose these functions. Replace this module
with version-pinned Radia imports once that API is published. No CLN code here.
The caller owns TaskManager; returned vectors/fields own their storage.
"""
import scipy.linalg as la
import scipy.sparse as sp
import radia.cohomology as rc
MU_0 = 4e-7 * 3.141592653589793

import math

import numpy as np

RELATIVE_LIMIT = 1e-6


def frobenius_norm_on(matrix, free) -> float:
    """||A||_F restricted to the free rows and columns (``CSR()`` holds both triangles)."""
    values, columns, pointers = (np.asarray(item) for item in matrix.CSR())
    pointers = pointers.astype(np.int64)
    rows = np.repeat(np.arange(len(pointers) - 1), np.diff(pointers))
    keep = free[rows] & free[columns.astype(np.int64)]
    return float(np.sqrt(np.sum(np.abs(values[keep]) ** 2)))


def residual_scale(rhs_norm: float, reference_norm: float = 0.0) -> float:
    """Use an explicit fixed physical load scale, never the computed solution.

    A caller may supply its original effective-load norm for Newton correction
    equations. The reference must be fixed before solving, finite and nonnegative.
    """
    if not math.isfinite(reference_norm) or reference_norm < 0:
        raise ValueError("reference_norm must be finite and nonnegative")
    return max(float(rhs_norm), float(reference_norm), 1e-300)


def check_true_residual(matrix, residual, solution, rhs, free, what,
                        *, reference_norm: float = 0.0) -> float:
    """Require the shared limit; return ||r||/max(||b||, reference_norm).

    ``residual``, ``solution`` and ``rhs`` are full-length arrays; ``free`` is
    the boolean mask of the rows that were solved.
    """
    x = solution[free]
    r = residual[free]
    b = rhs[free]
    r_norm = float(np.linalg.norm(r))
    relative = r_norm / residual_scale(float(np.linalg.norm(b)), reference_norm)
    backward = math.inf
    finite = bool(np.all(np.isfinite(x))) and math.isfinite(relative)
    if finite and relative > RELATIVE_LIMIT:
        backward = r_norm / max(frobenius_norm_on(matrix, free) * float(np.linalg.norm(x))
                                + float(np.linalg.norm(b)), 1e-300)
    if not finite or relative > RELATIVE_LIMIT:
        raise RuntimeError(
            f"{what} true relative residual {relative:.3e} exceeds {RELATIVE_LIMIT:g}; diagnostic "
            f"normwise backward error is {backward:.3e}")
    return relative

from dataclasses import dataclass
from collections import deque
import numpy as np


def material_names(mesh, regions):
    """Validate explicit material names (no regular-expression selection)."""
    names = (regions,) if isinstance(regions, str) else tuple(regions)
    if any(not isinstance(n, str) or not n for n in names):
        raise ValueError("regions must be nonempty material names")
    if not names or len(set(names)) != len(names):
        raise ValueError("regions must contain distinct material names")
    missing = set(names) - set(mesh.GetMaterials())
    if missing:
        raise ValueError(f"unknown material regions: {sorted(missing)}")
    return names


def material_region(mesh, names):
    import re
    return mesh.Materials("|".join(re.escape(n) for n in names))


def _cycle_rows(vertex_count, edges, selected):
    adjacency = [[] for _ in range(vertex_count)]
    for i, (lo, hi) in enumerate(edges):
        adjacency[lo].append((hi, i)); adjacency[hi].append((lo, i))
    tree = [[] for _ in range(vertex_count)]; visited = set()
    for root in range(vertex_count):
        if root in visited:
            continue
        visited.add(root); queue = deque([root])
        while queue:
            v = queue.popleft()
            for w, e in adjacency[v]:
                if w not in visited:
                    visited.add(w); queue.append(w)
                    tree[v].append((w, e)); tree[w].append((v, e))
    rows = np.zeros((len(selected), len(edges)))
    for k, edge in enumerate(selected):
        lo, hi = edges[edge]; queue = deque([hi]); parent = {hi: None}
        while queue and lo not in parent:
            v = queue.popleft()
            for w, e in tree[v]:
                if w not in parent:
                    parent[w] = (v, e); queue.append(w)
        rows[k, edge] = 1.; vertex = lo
        while vertex != hi:
            previous, e = parent[vertex]
            rows[k, e] += 1. if edges[e] == (previous, vertex) else -1.
            vertex = previous
    return rows


@dataclass
class RegionCohomology:
    mesh: object
    submesh: object
    regions: tuple
    space: object
    generators: tuple
    b1: int
    parent_vertices: np.ndarray
    parent_edge_dofs: np.ndarray
    edge_signs: np.ndarray
    cochains: np.ndarray
    cycles: np.ndarray
    periods: np.ndarray
    closure_residual: float


def region_cohomology(mesh, *, regions, target_space=None):
    """Return unit-period generators on a parent lowest-order HCurl space.

Zero and multiple holes are supported. Returned arrays and GridFunctions own
their storage. Unsupported non-tetrahedral meshes fail explicitly.
"""
    import ngsolve as ng
    from netgen.meshing import Mesh, MeshPoint, Point3d, Element3D
    if target_space is not None:
        raise ValueError("target_space transfer is not supported; use the returned lowest-order space")
    names = material_names(mesh, regions)
    elements = [e for e in mesh.Elements(ng.VOL) if e.mat in names]
    if mesh.dim != 3 or not elements or any(len(e.vertices) != 4 for e in elements):
        raise ValueError("region cohomology requires nonempty 3D tetrahedral regions")
    vertices = sorted({int(v.nr) for e in elements for v in e.vertices})
    mapping = {v: i + 1 for i, v in enumerate(vertices)}
    points = [tuple(p.p) for p in mesh.ngmesh.Points()]
    child = Mesh(dim=3)
    for v in vertices:
        child.Add(MeshPoint(Point3d(*points[v])))
    for e in elements:
        child.Add(Element3D(index=1, vertices=[mapping[int(v.nr)] for v in e.vertices]))
    child.SetMaterial(1, "selected")
    submesh = ng.Mesh(child)
    basis, b1, childspace, ctx, loops = rc.cohomology_basis(submesh)
    d0, d1, _, edges, nv, _, _ = ctx
    if abs(d1 @ d0).sum() > 1e-10:
        raise RuntimeError("tetrahedral incidence is not a chain complex")
    space = ng.HCurl(mesh, order=0)
    parent_dofs = {tuple(sorted(int(v.nr) for v in e.vertices)): space.GetDofNrs(e)[0]
                   for e in mesh.edges}
    child_dofs = {tuple(sorted(int(v.nr) for v in e.vertices)): childspace.GetDofNrs(e)[0]
                  for e in submesh.edges}
    dofs = np.array([parent_dofs[tuple(sorted((vertices[a], vertices[b])))] for a, b in edges])
    signs = np.array([1. if vertices[a] < vertices[b] else -1. for a, b in edges])
    cochains = np.zeros((len(edges), b1))
    generators = []
    for k, source in enumerate(basis):
        cochains[:, k] = [source.vec.FV().NumPy()[child_dofs[e]] for e in edges]
        gf = ng.GridFunction(space)
        gf.vec.FV().NumPy()[dofs] = signs * cochains[:, k]
        generators.append(gf)
    cycles = _cycle_rows(nv, edges, loops or [])
    periods = np.zeros((b1, space.ndof))
    periods[:, dofs] = cycles * signs[None, :]
    closure = float(np.linalg.norm(d1 @ cochains))
    if closure > 1e-8 or np.linalg.norm(cycles @ cochains - np.eye(b1)) > 1e-8:
        raise RuntimeError("cohomology generator closure/period check failed")
    return RegionCohomology(mesh, submesh, names, space, tuple(generators), b1,
                            np.array(vertices), dofs.copy(), signs.copy(), cochains.copy(),
                            cycles.copy(), periods.copy(), closure)


@dataclass
class NaturalMagneticRepresentative:
    """Natural-flux field H=h+grad(Omega); arrays own their storage."""
    field: object
    omega: object
    matrix: object
    rhs: object
    free_dofs: np.ndarray
    gauge_dofs: tuple
    relative_residual: float


def natural_magnetic_representative(mesh, representative, *, regions, mu=MU_0, order=2):
    """Minimize integral mu|h+grad(Omega)|ﾂｲ on explicit material regions.

    Uses the natural zero-normal magnetic-flux condition and one constant gauge
    per connected scalar component. This extends the scalar cut formulation,
    with no extra air conductivity or boundary Dirichlet condition. The caller
    owns TaskManager; the direct FE solve uses sparsecholesky only.
    """
    import ngsolve as ng
    names = material_names(mesh, regions)
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError("scalar order must be a positive integer")
    if isinstance(mu, dict):
        if set(mu) != set(names) or any(not np.isfinite(v) or v <= 0 for v in mu.values()):
            raise ValueError("mu must be positive for every selected material")
        weight = mesh.MaterialCF(mu, default=0.)
    else:
        if not np.isfinite(mu) or mu <= 0:
            raise ValueError("mu must be positive and finite")
        weight = ng.CF(float(mu))
    region = material_region(mesh, names)
    fes = ng.H1(mesh, order=order, definedon=region)
    parent = list(range(fes.ndof)); active = set()
    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]; v = parent[v]
        return v
    for element in mesh.Elements(ng.VOL):
        if element.mat not in names:
            continue
        dofs = [d for d in fes.GetDofNrs(element) if d >= 0]
        active.update(dofs)
        for d in dofs[1:]:
            parent[find(d)] = find(dofs[0])
    if not active:
        raise ValueError("selected scalar domain is empty")
    components = {}
    for d in sorted(active):
        components.setdefault(find(d), []).append(d)
    free = ng.BitArray(fes.ndof); free[:] = False
    for d in active:
        free[d] = True
    gauges = tuple(min(component) for component in components.values())
    for d in gauges:
        free[d] = False
    u, v = fes.TnT(); dx = ng.dx(definedon=region, bonus_intorder=12)
    form = ng.BilinearForm(fes); form += weight * ng.grad(u) * ng.grad(v) * dx
    form.Assemble()
    rhs = ng.LinearForm(fes); rhs += -weight * representative * ng.grad(v) * dx
    rhs.Assemble()
    omega = ng.GridFunction(fes)
    omega.vec.data = form.mat.Inverse(free, inverse="sparsecholesky") * rhs.vec
    mask = np.array(list(free), dtype=bool)
    residual = rhs.vec.CreateVector(); residual.data = rhs.vec - form.mat * omega.vec
    relative = check_true_residual(form.mat, residual.FV().NumPy(), omega.vec.FV().NumPy(),
                                   rhs.vec.FV().NumPy(), mask, "natural cohomology potential")
    saved_rhs = rhs.vec.CreateVector(); saved_rhs.data = rhs.vec
    return NaturalMagneticRepresentative(representative + ng.grad(omega), omega,
                                         form.mat, saved_rhs, mask.copy(), gauges, relative)


def _csr(matrix):
    return sp.csr_matrix(matrix.CSR(), shape=(matrix.height, matrix.width)).copy()


def _inverse_columns(form, free, columns):
    """Native FE inverse, checked on every RHS, with owned return storage."""
    inv = form.mat.Inverse(free, inverse="sparsecholesky")
    mask = np.array(list(free), bool)
    output = np.zeros((form.mat.height, columns.shape[1]))
    for k in range(columns.shape[1]):
        rhs = form.mat.CreateColVector(); rhs[:] = 0.
        rhs.FV().NumPy()[mask] = columns[:, k]
        solution = rhs.CreateVector(); solution.data = inv * rhs
        residual = rhs.CreateVector(); residual.data = rhs - form.mat * solution
        check_true_residual(form.mat, residual.FV().NumPy(), solution.FV().NumPy(),
                            rhs.FV().NumPy(), mask, "insulated current mass inverse")
        output[:, k] = solution.FV().NumPy().copy()
    return output


@dataclass
class LoopCurrentBasis:
    space: object
    currents: tuple
    topology: object
    coefficients: np.ndarray
    cut_flux: np.ndarray
    joule_matrix: np.ndarray
    divergence_residual: float
    stationarity_residual: float
    divergence_free_dimension: int
    curl_dimension: int
    rank_tolerance: float


def insulated_loop_currents(mesh, *, conductor_regions, sigma, insulating_boundaries,
                            current_order=0, max_dense_dofs=4000, rank_tolerance=1e-10):
    """Construct all minimum-Joule unit-cut-flux currents jointly.

    Supported contract: RT0/DG0/lowest-order curl on closed, entirely insulated
    tetrahedral conductors. This bounded dense-constraint implementation raises
    for larger or higher-order cases instead of changing the numerical route.
    ``sigma`` is positive SI conductivity, scalar or per-material mapping.
    """
    import ngsolve as ng
    names = material_names(mesh, conductor_regions)
    if current_order != 0:
        raise ValueError("only the compatible order-0 current complex is supported")
    boundaries = ((insulating_boundaries,) if isinstance(insulating_boundaries, str)
                  else tuple(insulating_boundaries))
    if not boundaries or set(boundaries) - set(mesh.GetBoundaries()):
        raise ValueError("insulating boundary names must exist")
    if not 0 < rank_tolerance < 1e-4:
        raise ValueError("rank_tolerance must lie between 0 and 1e-4")
    if isinstance(sigma, dict):
        if set(sigma) != set(names) or any(not np.isfinite(v) or v <= 0 for v in sigma.values()):
            raise ValueError("sigma must be positive for every conductor material")
        conductivity = mesh.MaterialCF(sigma, default=1.)
    else:
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be positive and finite")
        conductivity = ng.CF(float(sigma))
    region = material_region(mesh, names)
    import re
    boundary_pattern = "|".join(re.escape(n) for n in boundaries)
    current = ng.HDiv(mesh, order=0, definedon=region, dirichlet=boundary_pattern)
    scalar = ng.L2(mesh, order=0, definedon=region)
    mask = np.array(list(current.FreeDofs()), bool); fs = np.array(list(scalar.FreeDofs()), bool)
    if not np.any(mask):
        raise ValueError("mesh has no interior current degrees of freedom")
    if sum(mask) > max_dense_dofs:
        raise ValueError("current constraint system exceeds max_dense_dofs")
    j, w = current.TnT(); dx = ng.dx(definedon=region, bonus_intorder=12)
    mass = ng.BilinearForm(current); mass += j * w * dx; mass.Assemble()
    loss = ng.BilinearForm(current); loss += j * w / conductivity * dx; loss.Assemble()
    pair = current * scalar; (j, p), (w, q) = pair.TnT()
    divergence = ng.BilinearForm(pair); divergence += ng.div(j) * q * dx; divergence.Assemble()
    D = _csr(divergence.mat)[current.ndof:, :current.ndof][fs][:, mask].toarray()
    _, singular, vh = la.svd(D, full_matrices=False)
    rank = int(sum(singular > rank_tolerance * (singular[0] if len(singular) else 1.)))
    div_constraints = vh[:rank]
    topology = region_cohomology(mesh, regions=names)
    gamma = np.zeros((topology.b1, sum(mask)))
    for k, h in enumerate(topology.generators):
        load = ng.LinearForm(current); load += h * current.TestFunction() * dx; load.Assemble()
        gamma[k] = load.vec.FV().NumPy().copy()[mask]
    # Every conductor boundary face must be clamped; terminal devices need a
    # different port space and are intentionally rejected by this closed API.
    face_counts = {}
    for element in mesh.Elements(ng.VOL):
        if element.mat in names:
            for facet in element.faces:
                face_counts[facet] = face_counts.get(facet, 0) + 1
    for element in mesh.Elements(ng.VOL):
        if element.mat not in names:
            continue
        for facet in element.faces:
            if face_counts[facet] == 1:
                face_dofs = current.GetDofNrs(facet)
                if any(d >= 0 and mask[d] for d in face_dofs):
                    raise ValueError("all conductor boundary normal-current DOFs must be insulated")
    coefficients = np.zeros((current.ndof, topology.b1)); stationarity = 0.
    if topology.b1:
        norms = la.norm(gamma, axis=1)
        if np.any(norms == 0):
            raise RuntimeError("cut flux pairing has a zero row")
        constraints = np.vstack([div_constraints, gamma / norms[:, None]])
        if np.linalg.matrix_rank(constraints, tol=rank_tolerance) != len(constraints):
            raise RuntimeError("cut constraints are dependent on divergence constraints")
        inverse = _inverse_columns(loss, current.FreeDofs(), constraints.T)
        schur = constraints @ inverse[mask]
        target = np.vstack([np.zeros((rank, topology.b1)), np.diag(1 / norms)])
        multipliers = la.solve(schur, target, assume_a="pos")
        coefficients = inverse @ multipliers
        R = _csr(loss.mat)[mask][:, mask]
        stationarity = float(la.norm(R @ coefficients[mask] - constraints.T @ multipliers)
                             / max(la.norm(constraints.T @ multipliers), 1e-300))
    curls = ng.HCurl(mesh, order=0, definedon=region, dirichlet=boundary_pattern)
    ft = np.array(list(curls.FreeDofs()), bool)
    pair = current * curls; (j, t), (w, v) = pair.TnT()
    curl_form = ng.BilinearForm(pair); curl_form += ng.curl(t) * w * dx; curl_form.Assemble()
    C = _csr(curl_form.mat)[:current.ndof, current.ndof:][mask][:, ft].toarray()
    sv = la.svdvals(C); curl_rank = int(sum(sv > rank_tolerance * (sv[0] if len(sv) else 1.)))
    curl_coefficients = _inverse_columns(mass, current.FreeDofs(), C)[mask]
    if la.norm(gamma @ curl_coefficients) > 1e-8 * max(la.norm(gamma) * la.norm(curl_coefficients), 1.):
        raise RuntimeError("unit-period pairing does not annihilate the insulated curl range")
    dimension = int(sum(mask)) - rank
    if dimension - curl_rank != topology.b1:
        raise RuntimeError("incompatible insulated divergence/curl topology dimensions")
    flux = gamma @ coefficients[mask]
    div_defect = float(la.norm(D @ coefficients[mask]) / max(la.norm(D) * la.norm(coefficients[mask]), 1.))
    if la.norm(flux - np.eye(topology.b1)) > 1e-8 or div_defect > 1e-8 or stationarity > 1e-6:
        raise RuntimeError("minimum-Joule loop constraints or stationarity failed")
    fields = []
    for k in range(topology.b1):
        gf = ng.GridFunction(current); gf.vec.FV().NumPy()[:] = coefficients[:, k]
        fields.append(gf)
    joule = coefficients.T @ _csr(loss.mat) @ coefficients
    return LoopCurrentBasis(current, tuple(fields), topology, coefficients.copy(), flux.copy(),
                             joule, div_defect, stationarity, dimension, curl_rank, rank_tolerance)
