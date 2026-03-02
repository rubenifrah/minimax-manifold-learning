import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.csgraph import dijkstra
import os

from generate_data import generate_sphere_data, generate_torus_data
from ISOMAP import k_neighbors_graph
from TDC import reconstruct_surface_tdc, compute_tdc_distances

def get_shortest_path(predecessors, start_idx, end_idx):
    """ Backtrack through the predecessors matrix to find the exact trajectory indices. """
    path = []
    curr = end_idx
    while curr != -9999 and curr >= 0:
        path.append(curr)
        if curr == start_idx:
            break
        curr = predecessors[start_idx, curr]
    return path[::-1] # reverse so it goes start -> end

def plot_geodesic_comparison(X, start_idx, end_idx, k=8, true_dist=None, true_path=None, method='both', title="Geodesic Estimation", save_name="geodesic_path.pdf"):
    """
    General function to compute ISOMAP and/or TDC geodesics, plotting them against an optional true path.
    method: 'isomap', 'tdc', or 'both'
    """
    p0 = X[start_idx]
    p1 = X[end_idx]
    
    run_isomap = method in ['isomap', 'both']
    run_tdc = method in ['tdc', 'both']
    
    has_iso_path = False
    has_tdc_path = False
    
    print(f"\n--- {title} ---")
    if true_dist is not None:
        print(f"True Geodesic Distance: {true_dist:.4f}")
        
    iso_dist, tdc_dist = None, None
    iso_path_points, tdc_path_points = [], []
    triangles = []
    
    if run_isomap:
        # Compute ISOMAP graph & shortest path
        iso_graph = k_neighbors_graph(X, n_neighbors=k)
        iso_dist_matrix, iso_predecessors = dijkstra(csgraph=iso_graph, directed=False, return_predecessors=True)
        iso_dist = iso_dist_matrix[start_idx, end_idx]
        iso_path_indices = get_shortest_path(iso_predecessors, start_idx, end_idx)
        
        has_iso_path = (len(iso_path_indices) > 0 and iso_path_indices[0] == start_idx)
        iso_path_points = X[iso_path_indices] if has_iso_path else []
        
        print(f"ISOMAP Estimated Distance (k={k}): {iso_dist:.4f}")
        if true_dist is not None:
            err_iso = abs(iso_dist - true_dist)
            print(f"ISOMAP Absolute Error: {err_iso:.4f}")

    if run_tdc:
        # Compute TDC Surface & exact shortest path
        triangles = reconstruct_surface_tdc(X)
        dist_matrix, geoalg = compute_tdc_distances(X, triangles)
        if geoalg is not None:
            tdc_dist = dist_matrix[start_idx, end_idx]
            print(f"TDC Estimated Distance: {tdc_dist:.4f}")
            if true_dist is not None:
                err_tdc = abs(tdc_dist - true_dist)
                print(f"TDC Absolute Error: {err_tdc:.4f}")
                
            # Trace the exact continuous path across the faces
            _, path_points = geoalg.geodesicDistance(start_idx, end_idx)
            tdc_path_points = np.array(path_points)
            has_tdc_path = len(tdc_path_points) > 0
        else:
            print("WARNING: TDC Surface Reconstruction Failed (no faces returned).")

    print("-" * (8 + len(title)))
    
    if run_isomap and not has_iso_path:
        print(f"WARNING: No ISOMAP path found between points at k={k}. Graph is disconnected.")
    if run_tdc and not has_tdc_path:
        print(f"WARNING: No TDC path found between points. Surface might be disconnected.")
        
    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if run_tdc and len(triangles) > 0:
        # Render the reconstructed complex surface mesh and the points
        ax.plot_trisurf(X[:, 0], X[:, 1], X[:, 2], 
                         triangles=triangles, cmap='viridis', edgecolor='none', alpha=0.7, zorder=1)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='gray', s=5, alpha=0.3, label="Manifold points", zorder=2)
    else:
        # full point cloud
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='gray', s=5, alpha=0.3, label="Manifold points", zorder=2)
    
    # true mathematical geodesic
    if true_path is not None:
        ax.plot(true_path[:, 0], true_path[:, 1], true_path[:, 2], 
                color='red', linewidth=3, label="True Geodesic", zorder=10)
    
    # ISOMAP shortest path
    if has_iso_path:
        ax.plot(iso_path_points[:, 0], iso_path_points[:, 1], iso_path_points[:, 2], 
                color='blue', linewidth=2, linestyle='--', marker='o', markersize=4, 
                label=f'ISOMAP Path (k={k})', zorder=11)
                
    # TDC shortest path
    if has_tdc_path:
        ax.plot(tdc_path_points[:, 0], tdc_path_points[:, 1], tdc_path_points[:, 2], 
                color='magenta', linewidth=2, linestyle='-', marker='s', markersize=3, 
                label=f'TDC Path', zorder=12)
            
    # two target points
    ax.scatter(*p0, color='green', s=100, marker='*', label=f'Start Point ({start_idx})', zorder=15)
    ax.scatter(*p1, color='orange', s=100, marker='*', label=f'End Point ({end_idx})', zorder=15)
    
    # Build subtitle
    subtitles = []
    if true_dist is not None:
        subtitles.append(f"True: {true_dist:.4f}")
    if run_isomap:
        subtitles.append(f"ISOMAP: {iso_dist:.4f}")
    if run_tdc:
        subtitles.append(f"TDC: {tdc_dist:.4f}" if tdc_dist is not None else "TDC: Failed")
        
    subtitle = " | ".join(subtitles)
    ax.set_title(f"{title}\n{subtitle}")
    
    # axis aspect
    ax.set_box_aspect([1, 1, 1])
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    # Save the plot
    output_dir = "images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    
    plt.show()

def sphere():
    radius = 1.0
    n_points = 100
    
    # base sphere point cloud
    X_base = generate_sphere_data(n_points, radius)
    
    # target points defined by angle difference
    angle = np.pi / 2
    # p0 = North pole
    p0 = np.array([0, 0, radius])
    # target point
    p1 = np.array([radius * np.sin(angle), 0, radius * np.cos(angle)])
    
    # Combine data + two points s.t. targets are exactly at indices 0 and 1
    X = np.vstack([p0, p1, X_base])
    
    # True Geodesic Distance
    true_dist = radius * angle
    
    # true mathematical geodesic (the Great Circle arc)
    t = np.linspace(0, angle, 50)
    arc_x = radius * np.sin(t)
    arc_y = np.zeros_like(t)
    arc_z = radius * np.cos(t)
    true_path = np.vstack([arc_x, arc_y, arc_z]).T
    
    # Compare both
    plot_geodesic_comparison(
        X=X, 
        start_idx=0, 
        end_idx=1,
        k=8,
        true_dist=true_dist, 
        true_path=true_path,
        method='both', # Try 'both', 'isomap', or 'tdc'
        title="Sphere: True Geodesic vs ISOMAP vs TDC",
        save_name="geodesic_path_sphere_unified.pdf"
    )

def torus():
    R = 2.0
    r = 0.8
    n_points = 1000
    
    # base torus point cloud
    X_base = generate_torus_data(n_points, R, r)
    
    # Let's fix a starting point and an ending point
    # Intrinsic coordinates (theta, phi)
    theta0, phi0 = 0.0, 0.0
    # Destination is rotated fully around the small circle (pi) and halfway around the big circle (pi/2)
    theta1, phi1 = np.pi, np.pi / 2
    
    p0 = np.array([(R + r * np.cos(theta0)) * np.cos(phi0), 
                   (R + r * np.cos(theta0)) * np.sin(phi0), 
                   r * np.sin(theta0)])
                   
    p1 = np.array([(R + r * np.cos(theta1)) * np.cos(phi1), 
                   (R + r * np.cos(theta1)) * np.sin(phi1), 
                   r * np.sin(theta1)])
                   
    # Combine data
    X = np.vstack([p0, p1, X_base])
    
    plot_geodesic_comparison(
        X=X, 
        start_idx=0, 
        end_idx=1, 
        k=12, 
        true_dist=None, 
        true_path=None,
        method='both', # Try 'both', 'isomap', or 'tdc'
        title="Torus: ISOMAP vs TDC Geodesic Estimation",
        save_name="geodesic_path_torus_unified.pdf"
    )

if __name__ == "__main__":
    torus()
