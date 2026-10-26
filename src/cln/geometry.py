"""Test geometries."""
from netgen.occ import Cylinder, Z, OCCGeometry
from ngsolve import Mesh, TaskManager


def cylinder_mesh(radius=0.01, length=0.01, maxh=None, maxh_conductor=None, curve=3):
    """Conductor-only cylinder along z. Boundaries: 'conductorBND' (lateral), 'in', 'out'."""
    cyl = Cylinder((0, 0, 0), Z, r=radius, h=length)
    cyl.faces.name = "conductorBND"
    cyl.faces.Max(Z).name = "out"
    cyl.faces.Min(Z).name = "in"
    cyl.mat("sig")
    cyl.maxh = maxh_conductor or radius/10
    with TaskManager():
        return Mesh(OCCGeometry(cyl).GenerateMesh(maxh=maxh or radius/2)).Curve(curve)
