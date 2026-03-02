import numpy as np
import sys
sys.path.append("src")
from generate_data import generate_torus_data
from TDC import reconstruct_surface_tdc
import pygeodesic.geodesic as geodesic

R = 2.0
r = 0.8
n_points = 100

print("Generating data...")
X = generate_torus_data(n_points, R, r)
print("Reconstructing surface...")
triangles = reconstruct_surface_tdc(X)
print(f"Extracted {len(triangles)} triangles")

print("Initializing Exact Geodesic...")
try:
    # Set a timeout? pygeodesic is C++ bound, might not be interruptible easily.
    geoalg = geodesic.PyGeodesicAlgorithmExact(X, triangles)
    print("Initialization finished!")
except Exception as e:
    print(f"Failed: {e}")
