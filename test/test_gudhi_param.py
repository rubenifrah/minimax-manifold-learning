import numpy as np
import sys
sys.path.append("src")
from generate_data import generate_torus_data
import gudhi

def check_manifold(triangles):
    edges = {}
    for t in triangles:
        tri_edges = [tuple(sorted([t[0], t[1]])), 
                     tuple(sorted([t[1], t[2]])), 
                     tuple(sorted([t[2], t[0]]))]
        for e in tri_edges:
            edges[e] = edges.get(e, 0) + 1
    counts = list(edges.values())
    nm = sum(1 for c in counts if c > 2)
    return nm

X = generate_torus_data(1000, 2.0, 0.8)

max_edges = [None, 0.5, 1.0, 2.0, 5.0]

for me in max_edges:
    tc = gudhi.TangentialComplex(intrisic_dim=2, points=X)
    if me is not None:
        tc.set_max_squared_edge_length(me)
    tc.compute_tangential_complex()
    st = tc.create_simplex_tree()
    tri = np.array([s for s, _ in st.get_simplices() if len(s) == 3])
    print(f"Max Edge Sq = {me} | Triangles = {len(tri)} | Non-manifold edges = {check_manifold(tri)}")

print("\nTesting Perturbation Fix:")
tc2 = gudhi.TangentialComplex(intrisic_dim=2, points=X)
tc2.compute_tangential_complex()
tc2.fix_inconsistencies_using_perturbation(0.01)
st2 = tc2.create_simplex_tree()
tri2 = np.array([s for s, _ in st2.get_simplices() if len(s) == 3])
print(f"Perturbation | Triangles = {len(tri2)} | Non-manifold edges = {check_manifold(tri2)}")
