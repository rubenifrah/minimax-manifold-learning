import numpy as np
import sys
sys.path.append("src")
from generate_data import generate_torus_data
from TDC import reconstruct_surface_tdc
import potpourri3d as pp3d

X = generate_torus_data(100, 2.0, 0.8)
triangles = reconstruct_surface_tdc(X)

print("Triangles:", len(triangles))

try:
    solver = pp3d.MeshHeatMethodDistanceSolver(X, triangles)
    dist = solver.compute_distance(0)
    print("Heat method success, dist to 1:", dist[1])
except Exception as e:
    print("Heat method failed:", e)

try:
    dist_exact = pp3d.compute_distance(X, triangles, 0)
    print("Exact distance success, dist to 1:", dist_exact[1])
except Exception as e:
    print("Exact distance failed:", e)
