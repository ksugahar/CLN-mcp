"""Gmsh-free absolute air cohomology with natural magnetic boundary."""
import hashlib,math
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import ngsolve as ng
import radia.cohomology as rc


def natural_harmonic(mesh,representative,order=2,weight=1.):
    scalar=ng.H1(mesh,order=order)
    dx=ng.dx(intrules={ng.ET.TET:ng.IntegrationRule(ng.ET.TET,order=16)})
    u,v=scalar.TnT();form=ng.BilinearForm(scalar);form+=weight*ng.grad(u)*ng.grad(v)*dx;form.Assemble()
    rhs=ng.LinearForm(scalar);rhs+=-weight*representative*ng.grad(v)*dx;rhs.Assemble()
    # One constant gauge only; no outer Dirichlet magnetic condition.
    free=np.array(list(scalar.FreeDofs()),bool);free[0]=False
    matrix=sp.csr_matrix(form.mat.CSR(),shape=(scalar.ndof,scalar.ndof))[free][:,free].tocsc()
    phi=ng.GridFunction(scalar);phi.vec.FV().NumPy()[free]=sla.spsolve(matrix,rhs.vec.FV().NumPy()[free])
    field=representative+ng.grad(phi)
    residual=matrix@phi.vec.FV().NumPy()[free]-rhs.vec.FV().NumPy()[free]
    return field,phi,float(np.linalg.norm(residual)/max(np.linalg.norm(rhs.vec.FV().NumPy()[free]),1e-300))


def air_generators(mesh):
    basis,b1,space,context,loops=rc.cohomology_basis(mesh,unit_circulation=True)
    d0,d1,edge_map,edges,nv,ne,nf=context
    assert abs(d1@d0).sum()<1e-10
    return dict(basis=basis,b1=b1,space=space,context=context,loops=loops,
                module_sha256=hashlib.sha256(Path(rc.__file__).read_bytes()).hexdigest())
