import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra

# implementation of ISOMAP algorithm from scratch

def k_neighbors_graph(X, n_neighbors):
    """
    Construct the k-nearest neighbors graph.
    Returns a sparse adjacency matrix with Euclidean distances as weights.
    """
    return kneighbors_graph(X, n_neighbors=n_neighbors, mode='distance')

def compute_geodesic_distances(graph):
    """
    Compute shortest paths between all pairs of nodes using Dijkstra.
    """
    dist_matrix = dijkstra(csgraph=graph, directed=False)
    # Handle infinite distances (disconnected components) by replacing with the max finite distance
    if np.isinf(dist_matrix).any():
        max_dist = dist_matrix[~np.isinf(dist_matrix)].max()
        dist_matrix[np.isinf(dist_matrix)] = max_dist * 10 # Penalize disconnected parts
    return dist_matrix

def mds(D, n_components=2):
    """
    Apply Multi-Dimensional Scaling on the distance matrix D.
    """
    n = D.shape[0]
    
    # Centering matrix H = I - 1/n * J
    H = np.eye(n) - np.ones((n, n)) / n
    
    # Compute inner product matrix B = -1/2 * H * D^2 * H
    D_squared = D ** 2
    B = -0.5 * H.dot(D_squared).dot(H)
    
    # Eigenvalue decomposition of B
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top n_components
    lambda_top = eigenvalues[:n_components]
    V_top = eigenvectors[:, :n_components]
    
    # Filter out any negative eigenvalues that arise due to numerical issues
    lambda_top[lambda_top < 0] = 0
    
    # Compute the final embedding Y = V_top * Lambda_top^(1/2)
    Y = V_top.dot(np.diag(np.sqrt(lambda_top)))
    
    return Y

def custom_isomap(X, n_neighbors=5, n_components=2):
    """
    Step-by-step ISOMAP algorithm.
    """
    # Step 1: Neighborhood graph
    graph = k_neighbors_graph(X, n_neighbors)
    
    # Step 2: Compute geodesic distances (shortest paths)
    geodesic_distances = compute_geodesic_distances(graph) # shape (N, N)
    
    # Step 3: Multi-Dimensional Scaling
    embedding = mds(geodesic_distances, n_components)
    
    return embedding

if __name__ == '__main__':
    from generate_data import generate_swiss_data
    from sklearn.decomposition import PCA
    
    print("Generating Swiss roll data...")
    # Generate data
    X = generate_swiss_data(1000)
    
    # Calculate a color map based on the 't' parameter from generation (roughly distance from origin in x-y plane)
    colors = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    
    print("Running Custom ISOMAP...")
    X_isomap = custom_isomap(X, n_neighbors=10, n_components=2)
    
    print("Running PCA for comparison...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Plotting
    fig = plt.figure(figsize=(15, 5))
    
    # Original Data
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors, cmap=plt.cm.Spectral)
    ax1.set_title("Original Data (Swiss Roll)")
    
    # PCA Projection
    ax2 = fig.add_subplot(132)
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap=plt.cm.Spectral)
    ax2.set_title("PCA Projection")
    
    # ISOMAP Projection
    ax3 = fig.add_subplot(133)
    ax3.scatter(X_isomap[:, 0], X_isomap[:, 1], c=colors, cmap=plt.cm.Spectral)
    ax3.set_title("Custom ISOMAP Projection")
    
    plt.tight_layout()
    plt.show()
