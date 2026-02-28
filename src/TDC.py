import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gudhi
import os

from generate_data import generate_torus_data, generate_sphere_data, generate_tubular_knot_surface, generate_swiss_data 


def reconstruct_surface_tdc(points, intrinsic_dim=2):
    """
    Reconstruct a surface from a point cloud using Tangential Delaunay Complex.
    """
    print("Computing Tangential Complex...")
    
    # we will probably work only with intrinsic dimension is 2 for a surface embedded in 3D
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

