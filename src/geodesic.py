import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.csgraph import dijkstra

from generate_data import generate_sphere_data
from ISOMAP import k_neighbors_graph
import os

def get_shortest_path(predecessors, start_idx, end_idx):
    """ Backtrack through the predecessors matrix to find the exact trajectory indices. """
    # predecessors[i,j] contains the index of the previous point on the path from i to j
    # start_idx is the index of the start point
    # end_idx is the index of the end point
    path = []
    curr = end_idx
    while curr != -9999 and curr >= 0:
        path.append(curr)
        if curr == start_idx:
            break
        curr = predecessors[start_idx, curr]
    return path[::-1] # reverse so it goes start -> end

def main():
    radius = 1.0
    n_points = 1500
    
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
    print(f"True Geodesic Distance (angle = pi/4): {true_dist:.4f}")
    
    # Compute ISOMAP graph
    k = 8
    graph = k_neighbors_graph(X, n_neighbors=k)
    
    # Dijkstra. We set return_predecessors=True to be able to physically plot the graph path taken
    dist_matrix, predecessors = dijkstra(csgraph=graph, directed=False, return_predecessors=True)
    
    iso_dist = dist_matrix[0, 1]
    print(f"ISOMAP Estimated Distance (k={k}):     {iso_dist:.4f}")
    
    error = abs(iso_dist - true_dist)
    print(f"Absolute Error:                        {error:.4f}")
    
    # ISOMAP path coordinates
    path_indices = get_shortest_path(predecessors, 0, 1)
    
    # Check if a path actually exists between them in the graph
    if len(path_indices) == 0 or path_indices[0] != 0:
        print(f"WARNING: No path found between points at k={k}. Graph is disconnected.")
        return
        
    path_points = X[path_indices]
    
    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # full sphere point cloud
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='gray', s=5, alpha=0.3, label="Manifold points")
    
    # true mathematical geodesic (the Great Circle arc)
    t = np.linspace(0, angle, 50)
    arc_x = radius * np.sin(t)
    arc_y = np.zeros_like(t)
    arc_z = radius * np.cos(t)
    ax.plot(arc_x, arc_y, arc_z, color='red', linewidth=3, label="True Geodesic (Great Circle Arc)")
    
    # ISOMAP shortest path through the generated graph
    ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], 
            color='blue', linewidth=2, linestyle='--', marker='o', markersize=5, 
            label=f'ISOMAP Path (k={k})')
            
    # two target points
    ax.scatter(*p0, color='green', s=100, marker='*', label='Start Point (P0)')
    ax.scatter(*p1, color='orange', s=100, marker='*', label='End Point (P1)')
    
    ax.set_title(f"True Geodesic vs ISOMAP Graph Estimation\n"
                 f"Estimated: {iso_dist:.4f} | True: {true_dist:.4f} | Error: {error:.4f}")
    
    ax.set_box_aspect([1, 1, 1])
    ax.legend()

    # Save the plot
    output_dir = "images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, "geodesic_path_sphere.pdf")
    plt.savefig(save_path, dpi=150)
    print(f"Saved figure to {save_path}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
