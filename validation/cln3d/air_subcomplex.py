"""Material air subcomplex and normalized Radia cotree-period transfer."""
from collections import deque
import numpy as np
import ngsolve as ng
from netgen.meshing import Mesh as NetgenMesh,MeshPoint,Point3d,Element3D
from air_cohomology import air_generators


def extract_air(base,alternate=False):
 elements=[e for e in base.mesh.Elements(ng.VOL) if e.mat=='air']
 vertices=sorted({int(v.nr) for e in elements for v in e.vertices})
 if alternate:vertices=vertices[::2]+vertices[1::2]
 mapping={v:i+1 for i,v in enumerate(vertices)};points=np.array(base.geometry_arrays['points'])
 mesh=NetgenMesh(dim=3)
 for v in vertices:mesh.Add(MeshPoint(Point3d(*points[v])))
 for e in elements:mesh.Add(Element3D(index=1,vertices=[mapping[int(v.nr)] for v in e.vertices]))
 mesh.SetMaterial(1,'air')
 return ng.Mesh(mesh),vertices


def period_row(vertex_count,edges,selected):
 adjacency=[[] for _ in range(vertex_count)]
 for i,(lo,hi) in enumerate(edges):adjacency[lo].append((hi,i));adjacency[hi].append((lo,i))
 visited=set();tree=[[] for _ in range(vertex_count)]
 for root in range(vertex_count):
  if root in visited:continue
  visited.add(root);queue=deque([root])
  while queue:
   v=queue.popleft()
   for w,e in adjacency[v]:
    if w not in visited:
     visited.add(w);queue.append(w);tree[v].append((w,e));tree[w].append((v,e))
 lo,hi=edges[selected];queue=deque([hi]);parent={hi:None}
 while queue and lo not in parent:
  v=queue.popleft()
  for w,e in tree[v]:
   if w not in parent:parent[w]=(v,e);queue.append(w)
 row=np.zeros(len(edges));row[selected]=1.;vertex=lo
 while vertex!=hi:
  previous,e=parent[vertex]
  # Return along the tree from hi to lo, opposite the cotree edge.
  row[e]+=1. if edges[e]==(previous,vertex) else -1.
  vertex=previous
 return row


def transfer_generator(base,global_space,alternate=False):
 mesh,vertices=extract_air(base,alternate=alternate);data=air_generators(mesh);assert data['b1']==1
 d0,d1,edge_map,edges,nv,ne,nf=data['context']
 childgf=data['basis'][0];childspace=data['space'];child_dofs={tuple(sorted(int(v.nr) for v in e.vertices)):childspace.GetDofNrs(e)[0] for e in mesh.edges}
 coefficients=np.array([childgf.vec.FV().NumPy()[child_dofs[edge]] for edge in edges])
 selected=data['loops'][0];cycle=period_row(nv,edges,selected)
 assert abs(cycle@coefficients-1)<1e-10
 global_dofs={tuple(sorted(int(v.nr) for v in e.vertices)):global_space.GetDofNrs(e)[0] for e in base.mesh.edges}
 h=np.zeros(global_space.ndof);period=np.zeros(global_space.ndof)
 for i,(lo,hi) in enumerate(edges):
  pair=(vertices[lo],vertices[hi]);canonical=tuple(sorted(pair));sign=1. if pair==canonical else -1.
  gdof=global_dofs[canonical];h[gdof]=sign*coefficients[i];period[gdof]=sign*cycle[i]
 assert abs(period@h-1)<1e-10
 return dict(h=h,period=period,b1=data['b1'],vertices=vertices,edges=edges,cochains=coefficients,cycle=cycle,
             module_sha256=data['module_sha256'],closure=float(np.linalg.norm(d1@coefficients)))
