import numpy as np
import sys
sys.path.append("src")
from generate_data import generate_torus_data, generate_sphere_data
from TDC import reconstruct_surface_tdc

def check_manifold(triangles):
    edges = {}
    for t in triangles:
        tri_edges = [tuple(sorted([t[0], t[1]])), 
                     tuple(sorted([t[1], t[2]])), 
                     tuple(sorted([t[2], t[0]]))]
        for e in tri_edges:
            edges[e] = edges.get(e, 0) + 1
            
    counts = list(edges.values())
    print(f"Total edges: {len(counts)}")
    print(f"Edges with 1 face (boundary): {counts.count(1)}")
    print(f"Edges with 2 faces (manifold): {counts.count(2)}")
    print(f"Edges with >2 faces (non-manifold): {sum(1 for c in counts if c > 2)}")

print("--- Sphere ---")
X_s = generate_sphere_data(500, 1.0)
tri_s = reconstruct_surface_tdc(X_s)
check_manifold(tri_s)

print("\n--- Torus ---")
X_t = generate_torus_data(500, 2.0, 0.8)
tri_t = reconstruct_surface_tdc(X_t)
check_manifold(tri_t)

