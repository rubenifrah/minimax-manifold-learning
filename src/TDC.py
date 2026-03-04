import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gudhi
import os
import pygeodesic.geodesic as geodesic

from generate_data import generate_torus_data, generate_sphere_data, generate_tubular_knot_surface, generate_swiss_data 


def reconstruct_surface_tdc(points, intrinsic_dim=2, max_edge_length_squared=None):
    """
    Reconstruct a surface from a point cloud using Tangential Delaunay Complex.
    """
    print("Computing Tangential Complex...")
    
    # Apply microscopic noise to uniquely perturb coordinates. GUDHI's C++ solver 
    # loops infinitely on perfectly uniform/co-circular geometric degeneracies.
    np.random.seed(42)
    jitter = np.random.normal(scale=1e-5, size=points.shape)
    perturbed_points = points + jitter
    
    tc = gudhi.TangentialComplex(intrisic_dim=intrinsic_dim, points=perturbed_points)
    if max_edge_length_squared is not None:
        tc.set_max_squared_edge_length(max_edge_length_squared)
    tc.compute_tangential_complex()
    
    # Export to a SimplexTree
    st = tc.create_simplex_tree()
    
    # Extract raw triangles
    raw_triangles = []
    for simplex, filtration in st.get_simplices():
        if len(simplex) == 3:
            raw_triangles.append(simplex)
            
    print(f"Extracted {len(raw_triangles)} raw triangles from the complex.")
    
    # Greedy topological pruning to enforce a strict 2-manifold 
    # (required for mathematically generic C++ Geodesic Solvers like CGAL)
    edge_counts = {}
    for t in raw_triangles:
        for i in range(3):
            e = tuple(sorted((t[i], t[(i+1)%3])))
            edge_counts[e] = edge_counts.get(e, 0) + 1
            
    current_triangles = list(raw_triangles)
    dropped = 0
    
    while True:
        non_manifold_edges = {e: c for e, c in edge_counts.items() if c > 2}
        if not non_manifold_edges:
            break
            
        worst_edge = max(non_manifold_edges, key=non_manifold_edges.get)
        
        # Find the worst triangles
        bad_tris_indices = []
        for idx, t in enumerate(current_triangles):
            for i in range(3):
                e = tuple(sorted((t[i], t[(i+1)%3])))
                if e == worst_edge:
                    bad_tris_indices.append(idx)
                    break
                    
        drop_idx = bad_tris_indices[-1]
        drop_t = current_triangles.pop(drop_idx)
        
        for i in range(3):
            e = tuple(sorted((drop_t[i], drop_t[(i+1)%3])))
            edge_counts[e] -= 1
            if edge_counts[e] == 0:
                del edge_counts[e]
                
        dropped += 1

    if dropped > 0:
        print(f"Pruned exactly {dropped} overlapping triangles to enforce strict 2-manifold geometry.")

    return np.array(current_triangles)


class TDCDistanceSolver:
    def __init__(self, points, triangles):
        self.points = points
        self.triangles = triangles
        
        # Build scipy graph for robust Dijkstra path fallback tracing
        from scipy.sparse import csr_matrix
        row, col, data = [], [], []
        for t in triangles:
            for i in range(3):
                u, v = t[i], t[(i+1)%3]
                d = np.linalg.norm(points[u] - points[v])
                row.extend([u, v])
                col.extend([v, u])
                data.extend([d, d])
        self.graph = csr_matrix((data, (row, col)), shape=(len(points), len(points)))
        
    def geodesicDistance(self, start_idx, end_idx):
        from scipy.sparse.csgraph import dijkstra
        dist, pred = dijkstra(self.graph, indices=start_idx, return_predecessors=True)
        path = []
        curr = end_idx
        while curr != -9999 and curr >= 0:
            path.append(curr)
            if curr == start_idx:
                break
            curr = pred[curr]
            
        path = path[::-1]
        path_points = self.points[path] if (len(path) > 0 and path[0] == start_idx) else []
        return 0, path_points


def compute_tdc_distances(points, triangles):
    """
    Compute exactly intrinsic geodesic distances across the faces 
    of the TDC reconstructed mesh using the Exact Geodesic algorithm.
    """
    n_points = len(points)
    
    if len(triangles) == 0:
        print("Warning: No triangles found. Returning infinite distances.")
        return np.full((n_points, n_points), np.inf), None

    # Exact Geodesic C++ solvers (CGAL, pygeodesic) mathematically crash/hang on non-manifold meshes.
    # We must explicitly check the mesh and abort gracefully if the user has not tuned max_edge_length_squared enough.
    edge_counts = {}
    for t in triangles:
        for i in range(3):
            e = tuple(sorted((t[i], t[(i+1)%3])))
            edge_counts[e] = edge_counts.get(e, 0) + 1
    
    non_manifold_edges = sum(1 for c in edge_counts.values() if c > 2)
    if non_manifold_edges > 0:
        print(f"\n[CRITICAL ERROR] The generated TDC mesh fundamentally contains {non_manifold_edges} non-manifold edges!!")
        print("Exact Surface Geodesic solvers CANNOT mathematically unroll overlapping/intersecting faces and will hang.")
        print("-> You MUST tune your --max_edge parameter to prevent GUDHI from creating these intersections!\n")
        return np.full((n_points, n_points), np.inf), None

    # Initialize the Exact Polyhedral Geodesic algorithm over the strict manifold mesh
    geoalg = geodesic.PyGeodesicAlgorithmExact(points, triangles)
    
    dist_matrix = np.zeros((n_points, n_points))
    
    print("Computing exact surface geodesics...")
    for i in range(n_points):
        distances, _ = geoalg.geodesicDistances(np.array([i]))
        dist_matrix[i, :] = distances
        
    disconnected_mask = dist_matrix > 1e15
    if disconnected_mask.any():
        max_dist = dist_matrix[~disconnected_mask].max()
        if max_dist == 0:
            max_dist = 1.0 
        dist_matrix[disconnected_mask] = max_dist * 10
        
    return dist_matrix, geoalg


def plot_and_save(points, triangles, name, output_dir="images"):
    fig = plt.figure(figsize=(14, 6))
    
    # Original Point Cloud
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c='blue', alpha=0.5)
    ax1.set_title("Original Point Cloud"+name)
    ax1.set_box_aspect([1, 1, 0.5])
    
    # Reconstructed Surface
    ax2 = fig.add_subplot(122, projection='3d')
    if len(triangles) > 0:
        ax2.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], 
                         triangles=triangles, cmap='viridis', edgecolor='none', alpha=0.8)
    else:
        print("Warning: No triangles found to plot.")
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c='red')
        
    ax2.set_title("Tangential Delaunay Complex Reconstruction")
    ax2.set_box_aspect([1, 1, 0.5])
    
    plt.tight_layout()
    
    # Sauvegarde
    filename = f"{name.lower().replace(' ', '_')}.pdf"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150)
    plt.show()
    plt.close(fig)




if __name__ == '__main__':
    n_points = 2500
    OUTPUT_FOLDER = "images"
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    generators = {
        "Torus": (generate_torus_data, {'R': 2.0, 'r': 0.8}),
        "Sphere": (generate_sphere_data, {'radius': 1.5}),
        "Swiss": (generate_swiss_data,  {'t_max' : 2*np.pi, 'h_max': 2*np.pi}),
        "Knot_3_7": (generate_tubular_knot_surface, {'p': 3, 'q': 7, 'r0': 4.0, 'tube_r': 0.5}),
        "Knot_2_3": (generate_tubular_knot_surface, {'p': 2, 'q': 3, 'r0': 3.0, 'tube_r': 0.3}),
        "Knot_3_2": (generate_tubular_knot_surface, {'p': 3, 'q': 2, 'r0': 3.0, 'tube_r': 0.3})
    }

    for name, (func, params) in generators.items():
        points = func(n_points=n_points, **params)
        # It seems like adding slight noise can sometimes help TDC avoid degenerate cases, 
        # but first we try without
    
        # Reconstruct the surface
        triangles = reconstruct_surface_tdc(points) #
        
        # Plotting
        plot_and_save(points, triangles, name, OUTPUT_FOLDER)

