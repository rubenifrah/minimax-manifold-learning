import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gudhi

def reconstruct_surface_tdc(points, intrinsic_dim=2):
    """
    Reconstruct a surface from a point cloud using Tangential Delaunay Complex.
    """
    print("Computing Tangential Complex...")
    
    # Intrinsic dimension is 2 for a surface embedded in 3D
    tc = gudhi.TangentialComplex(intrisic_dim=intrinsic_dim, points=points)
    tc.compute_tangential_complex()
    
    # Export to a SimplexTree
    st = tc.create_simplex_tree()
    
    # Extract triangles (simplices of dimension 2, i.e., 3 vertices)
    triangles = []
    for simplex, filtration in st.get_simplices():
        if len(simplex) == 3:
            triangles.append(simplex)
            
    print(f"Extracted {len(triangles)} triangles from the complex.")
    return np.array(triangles)

if __name__ == '__main__':
    from generate_data import generate_torus_data, generate_sphere_data
    
    # Generate data
    print("Generating Torus point cloud...")
    n_points = 500
    points = generate_torus_data(n_points, R=2.0, r=0.8)
    
    # Adding slight noise can sometimes help TDC avoid degenerate cases, 
    # but we will try with exact points first.
    
    # Reconstruct surface
    triangles = reconstruct_surface_tdc(points, intrinsic_dim=2)
    
    # Plotting
    fig = plt.figure(figsize=(14, 6))
    
    # Original Point Cloud
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c='blue', alpha=0.5)
    ax1.set_title("Original Point Cloud (Torus)")
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
    plt.show()
