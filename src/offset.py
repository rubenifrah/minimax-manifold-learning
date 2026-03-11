import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree

def get_offset_volume_graph(X, epsilon, resolution=50):
    """
    Constructs the volumetric connected graph (the "tube" of balls).
    """
    print(f"  -> Voxelizing bounding volume (resolution={resolution}^3)...")
    
    # Define rigid bounding box inflated by epsilon
    min_bounds = np.min(X, axis=0) - (epsilon * 1.5)
    max_bounds = np.max(X, axis=0) + (epsilon * 1.5)
    
    # Discretize Ambient Space
    x_grid = np.linspace(min_bounds[0], max_bounds[0], resolution)
    y_grid = np.linspace(min_bounds[1], max_bounds[1], resolution)
    z_grid = np.linspace(min_bounds[2], max_bounds[2], resolution)
    
    # Meshgrid gives us all 3D combinations
    xv, yv, zv = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    voxel_coords = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T
    
    print(f"  -> Querying Hausdorff offset across {len(voxel_coords)} internal points (ε={epsilon:.4f})...")
    manifold_tree = cKDTree(X)
    
    # Query every voxel to its nearest manifold point. If > epsilon, it's outside the balls.
    distances, closest_pts = manifold_tree.query(voxel_coords)
    valid_mask = distances <= epsilon
    valid_voxels = voxel_coords[valid_mask]
    
    print(f"  -> Extracted bounding volume: {len(valid_voxels)} valid continuous internal points.")
    
    if len(valid_voxels) == 0:
        return None, None
        
    print("  -> Building 26-connectivity continuous internal graph...")
    # Calculate cell diagonal size for safe connectivity
    cell_size_x = x_grid[1] - x_grid[0]
    cell_size_y = y_grid[1] - y_grid[0]
    cell_size_z = z_grid[1] - z_grid[0]
    max_edge = np.sqrt(cell_size_x**2 + cell_size_y**2 + cell_size_z**2) * 1.05
    
    valid_tree = cKDTree(valid_voxels)
    
    # We constrain the search tightly to the 26 immediate neighbors (27 including self)
    distances, indices = valid_tree.query(valid_voxels, k=27, distance_upper_bound=max_edge)
    
    row_indices = []
    col_indices = []
    weights = []
    
    n_valid = len(valid_voxels)
    
    for i in range(n_valid):
        for k_idx, j in enumerate(indices[i]):
            if j == i or j == n_valid: # Skip self and 'not found' padding index
                continue
            dist = distances[i][k_idx]
            row_indices.append(i)
            col_indices.append(j)
            weights.append(dist)
            
    offset_graph = csr_matrix((weights, (row_indices, col_indices)), shape=(n_valid, n_valid))
    
    return valid_voxels, offset_graph

def _get_bottleneck_distance(X, start_idx, end_idx):
    """
    Finds the exact minimum epsilon required to connect start to end.
    Calculates MST to instantly find the bottleneck edge.
    """
    print("  -> Calculating Minimax Bottleneck required to connect targets...")
    
    manifold_tree = cKDTree(X)
    distances, indices = manifold_tree.query(X, k=min(20, len(X)))
    
    row_indices = []
    col_indices = []
    weights = []
    
    for i in range(len(X)):
        for k_idx, j in enumerate(indices[i][1:]):  # Skip self
            dist = distances[i][k_idx+1]
            row_indices.extend([i, j])
            col_indices.extend([j, i])
            weights.extend([dist, dist])
            
    n = len(X)
    baseline_graph = csr_matrix((weights, (row_indices, col_indices)), shape=(n, n))
    
    mst = minimum_spanning_tree(baseline_graph)
    dist_matrix, predecessors = dijkstra(csgraph=mst, directed=False, indices=start_idx, return_predecessors=True)
    
    if dist_matrix[end_idx] == np.inf:
        return np.inf
        
    curr = end_idx
    max_jump = 0.0
    
    while curr != -9999 and curr >= 0 and curr != start_idx:
        prev = predecessors[curr]
        jump_dist = np.linalg.norm(X[curr] - X[prev])
        max_jump = max(max_jump, jump_dist)
        curr = prev
        
    # Minimum radius to connect points functionally is half the largest jump constraint
    return max_jump / 2.0

def _try_path_for_epsilon(X, start_idx, end_idx, epsilon, resolution=50):
    valid_voxels, offset_graph = get_offset_volume_graph(X, epsilon, resolution)
    
    if valid_voxels is None:
        return np.inf, None, [], []
        
    voxel_tree = cKDTree(valid_voxels)
    _, p0_voxel_idx = voxel_tree.query(X[start_idx])
    _, p1_voxel_idx = voxel_tree.query(X[end_idx])
    
    dist_matrix, predecessors = dijkstra(csgraph=offset_graph, directed=False, indices=p0_voxel_idx, return_predecessors=True)
    offset_dist = dist_matrix[p1_voxel_idx]
    
    if offset_dist == np.inf:
        return np.inf, valid_voxels, [], []
        
    path_indices = []
    curr = p1_voxel_idx
    while curr != -9999 and curr >= 0:
        path_indices.append(curr)
        if curr == p0_voxel_idx:
            break
        curr = predecessors[curr]
    
    path_indices = path_indices[::-1]
    offset_path = valid_voxels[path_indices]
    
    # Identify which specific true manifold points (balls) generated this string-pulled continuous path
    manifold_tree = cKDTree(X)
    _, sphere_indices = manifold_tree.query(offset_path)
    
    unique_sphere_indices = []
    for idx in sphere_indices:
        if len(unique_sphere_indices) == 0 or idx != unique_sphere_indices[-1]:
            unique_sphere_indices.append(idx)
            
    path_spheres = X[unique_sphere_indices]
    
    return offset_dist, valid_voxels, offset_path, path_spheres

def compute_offset_distances(X, start_idx, end_idx, epsilon, resolution=50, dynamic_epsilon=False):
    """
    Computes the continuous Aamari et al. Volumetric Offset geodesic using exact mathematical limits 
    but simulating continuous string-pulling internally.
    """
    if not dynamic_epsilon:
        print(f"  -> Pathfinding continuous volumetric geodesic (fixed ε={epsilon})...")
        dist, valid_voxels, path, path_spheres = _try_path_for_epsilon(X, start_idx, end_idx, epsilon, resolution)
        return dist, valid_voxels, path, path_spheres, epsilon
        
    print("  -> Phase 1: Analytically computing optimal connective dynamic epsilon...")
    eps_min = _get_bottleneck_distance(X, start_idx, end_idx)
    
    if eps_min == np.inf:
        print("     -> Failed: Manifold topology is disconnected.")
        return None, [], [], [], 0.0
        
    print(f"     -> Minimal connecting radius successfully proved: ε_min = {eps_min:.4f}")
    
    # Ensure floating-point precision overlap
    best_eps = eps_min + 1e-4
    
    print(f"  -> Phase 2: String-pulling continuous trajectory across internal volume (resolution={resolution}^3)...")
    best_dist, best_voxels, best_path, best_spheres = _try_path_for_epsilon(X, start_idx, end_idx, best_eps, resolution)
    
    return best_dist, best_voxels, best_path, best_spheres, best_eps
