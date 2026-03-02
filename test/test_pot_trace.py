import numpy as np
import sys
sys.path.append("src")
from generate_data import generate_torus_data
from TDC import reconstruct_surface_tdc
import potpourri3d as pp3d

X = generate_torus_data(100, 2.0, 0.8)
triangles = reconstruct_surface_tdc(X)

print("Testing tracer...")
try:
    tracer = pp3d.GeodesicTracer(X, triangles)
    path = tracer.trace_vertex_to_vertex(1, 0)
    print("Trace success, path len:", len(path))
except Exception as e:
    print("Trace failed:", e)
