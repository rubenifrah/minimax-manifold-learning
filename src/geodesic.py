import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
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

def plot_geodesic_comparison(X, start_idx, end_idx, k=8, max_edge=None, true_dist=None, true_path=None, method='both', plot_engine='both', title="Geodesic Estimation", save_name="geodesic_path"):
    """
    General function to compute ISOMAP and/or TDC geodesics, plotting them against an optional true path.
    method: 'isomap', 'tdc', or 'both'
    plot_engine: 'plotly', 'matplotlib', or 'both'
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
        triangles = reconstruct_surface_tdc(X, max_edge_length_squared=max_edge)
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
        
    # Build subtitle
    subtitles = []
    if true_dist is not None:
        subtitles.append(f"True: {true_dist:.4f}")
    if run_isomap:
        subtitles.append(f"ISOMAP: {iso_dist:.4f}")
    if run_tdc:
        subtitles.append(f"TDC: {tdc_dist:.4f}" if tdc_dist is not None else "TDC: Failed")
    subtitle = " | ".join(subtitles)

    # Save and plotting logic
    output_dir = "images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if plot_engine in ['plotly', 'both']:
        # Plotting using Plotly
        fig = go.Figure()
        
        if run_tdc and len(triangles) > 0:
            # Render the reconstructed complex surface mesh
            fig.add_trace(go.Mesh3d(
                x=X[:, 0], y=X[:, 1], z=X[:, 2],
                i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                color='lightgreen', opacity=0.4,
                name='TDC Mesh', hoverinfo='skip'
            ))
            # Points
            fig.add_trace(go.Scatter3d(
                x=X[:, 0], y=X[:, 1], z=X[:, 2],
                mode='markers', marker=dict(size=2, color='gray', opacity=0.3),
                name="Manifold points"
            ))
        else:
            # Full point cloud
            fig.add_trace(go.Scatter3d(
                x=X[:, 0], y=X[:, 1], z=X[:, 2],
                mode='markers', marker=dict(size=2, color='gray', opacity=0.5),
                name="Manifold points"
            ))
        
        # True mathematical geodesic
        if true_path is not None:
            fig.add_trace(go.Scatter3d(
                x=true_path[:, 0], y=true_path[:, 1], z=true_path[:, 2],
                mode='lines', line=dict(color='red', width=6),
                name="True Geodesic"
            ))
        
        # ISOMAP shortest path
        if has_iso_path:
            fig.add_trace(go.Scatter3d(
                x=iso_path_points[:, 0], y=iso_path_points[:, 1], z=iso_path_points[:, 2],
                mode='lines+markers', 
                line=dict(color='blue', width=5, dash='dash'), 
                marker=dict(size=4, color='blue'),
                name=f'ISOMAP Path (k={k})'
            ))
                    
        # TDC shortest path
        if has_tdc_path:
            fig.add_trace(go.Scatter3d(
                x=tdc_path_points[:, 0], y=tdc_path_points[:, 1], z=tdc_path_points[:, 2],
                mode='lines+markers', 
                line=dict(color='magenta', width=6), 
                marker=dict(size=3, color='magenta', symbol='square'),
                name='TDC Path'
            ))
                
        # Two target points
        fig.add_trace(go.Scatter3d(
            x=[p0[0]], y=[p0[1]], z=[p0[2]],
            mode='markers', marker=dict(size=8, color='green', symbol='circle'),
            name=f'Start Point ({start_idx})'
        ))
        fig.add_trace(go.Scatter3d(
            x=[p1[0]], y=[p1[1]], z=[p1[2]],
            mode='markers', marker=dict(size=8, color='orange', symbol='circle'),
            name=f'End Point ({end_idx})'
        ))
        
        fig.update_layout(
            title=f"{title}<br><sup>{subtitle}</sup>",
            scene=dict(aspectmode='data'),
            legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1),
            margin=dict(l=0, r=0, b=0, t=50)
        )

        save_path_html = os.path.join(output_dir, f"{save_name}.html")
        fig.write_html(save_path_html)
        print(f"Saved interactive Plotly figure to {save_path_html}")
        
        # Automatically open interactive rendering
        if plot_engine == 'plotly':
            fig.show()
            
    if plot_engine in ['matplotlib', 'both']:
        # Plotting using Matplotlib
        fig_mpl = plt.figure(figsize=(10, 8))
        ax = fig_mpl.add_subplot(111, projection='3d')
        
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
        
        ax.set_title(f"{title}\n{subtitle}")
        
        # axis aspect
        ax.set_box_aspect([1, 1, 1])
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

        # Save the plot
        save_path_pdf = os.path.join(output_dir, f"{save_name}.pdf")
        plt.savefig(save_path_pdf, dpi=150, bbox_inches='tight')
        print(f"Saved static Matplotlib PDF to {save_path_pdf}")
        
        if plot_engine == 'matplotlib':
            plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Geodesic Path Estimation (ISOMAP vs TDC)")
    parser.add_argument('--manifold', type=str, choices=['sphere', 'torus', 'swiss', 'knot'], default='sphere', help="Manifold type")
    parser.add_argument('--n_points', type=int, default=1000, help="Number of points to sample")
    parser.add_argument('--k', type=int, default=8, help="Number of neighbors for ISOMAP")
    parser.add_argument('--max_edge', type=float, default=None, help="Maximum squared edge length for TDC")
    parser.add_argument('--method', type=str, choices=['isomap', 'tdc', 'both'], default='both', help="Method(s) to compute geodesic")
    parser.add_argument('--plot_engine', type=str, choices=['plotly', 'matplotlib', 'both'], default='both', help="Which plotting engine to use for generation")
    parser.add_argument('--points', type=str, choices=['fixed', 'random'], default='fixed', help="How to choose start and end points")
    
    args = parser.parse_args()
    
    true_dist = None
    true_path = None
    
    if args.manifold == 'sphere':
        radius = 1.0
        X_base = generate_sphere_data(args.n_points, radius)
        
        if args.points == 'fixed':
            angle = np.pi / 2
            p0 = np.array([0, 0, radius])
            p1 = np.array([radius * np.sin(angle), 0, radius * np.cos(angle)])
            true_dist = radius * angle
            
            t = np.linspace(0, angle, 50)
            arc_x = radius * np.sin(t)
            arc_y = np.zeros_like(t)
            arc_z = radius * np.cos(t)
            true_path = np.vstack([arc_x, arc_y, arc_z]).T
        else: # random
            idx0, idx1 = np.random.choice(len(X_base), 2, replace=False)
            p0 = X_base[idx0]
            p1 = X_base[idx1]
            
            # calculate true distance on sphere using arc length
            dot_product = np.clip(np.dot(p0, p1) / (radius**2), -1.0, 1.0)
            angle = np.arccos(dot_product)
            true_dist = radius * angle
            
            # SLERP for the true geodesic path interpolation
            omega = angle
            if omega > 1e-5:
                t = np.linspace(0, 1, 50)
                true_path = np.array([(np.sin((1-t_val)*omega)/np.sin(omega))*p0 + (np.sin(t_val*omega)/np.sin(omega))*p1 for t_val in t])
            else:
                true_path = np.vstack([p0, p1])
                
        if args.points == 'random':
            X_base = np.delete(X_base, [idx0, idx1], axis=0)
            
    elif args.manifold == 'torus':
        R = 2.0
        r = 0.8
        X_base = generate_torus_data(args.n_points, R, r)
        
        if args.points == 'fixed':
            theta0, phi0 = 0.0, 0.0
            theta1, phi1 = np.pi, np.pi / 2
            p0 = np.array([(R + r * np.cos(theta0)) * np.cos(phi0), 
                           (R + r * np.cos(theta0)) * np.sin(phi0), 
                           r * np.sin(theta0)])
            p1 = np.array([(R + r * np.cos(theta1)) * np.cos(phi1), 
                           (R + r * np.cos(theta1)) * np.sin(phi1), 
                           r * np.sin(theta1)])
        else:
            idx0, idx1 = np.random.choice(len(X_base), 2, replace=False)
            p0 = X_base[idx0]
            p1 = X_base[idx1]
            
        if args.points == 'random':
            X_base = np.delete(X_base, [idx0, idx1], axis=0)

    elif args.manifold == 'swiss':
        from generate_data import generate_swiss_data
        X_base = generate_swiss_data(args.n_points)
        if args.points == 'fixed':
            p0 = X_base[0]
            p1 = X_base[-1]
            X_base = X_base[1:-1]
        else:
            idx0, idx1 = np.random.choice(len(X_base), 2, replace=False)
            p0 = X_base[idx0]
            p1 = X_base[idx1]
            X_base = np.delete(X_base, [idx0, idx1], axis=0)

    elif args.manifold == 'knot':
        from generate_data import generate_tubular_knot_surface
        X_base = generate_tubular_knot_surface(args.n_points)
        if args.points == 'fixed':
            p0 = X_base[0]
            p1 = X_base[int(len(X_base)/2)]
            X_base = np.delete(X_base, [0, int(len(X_base)/2)], axis=0)
        else:
            idx0, idx1 = np.random.choice(len(X_base), 2, replace=False)
            p0 = X_base[idx0]
            p1 = X_base[idx1]
            X_base = np.delete(X_base, [idx0, idx1], axis=0)
            
    # Combine the start/end points exactly at indices 0 and 1
    X = np.vstack([p0, p1, X_base])
    
    plot_geodesic_comparison(
        X=X, 
        start_idx=0, 
        end_idx=1,
        k=args.k,
        max_edge=args.max_edge,
        true_dist=true_dist, 
        true_path=true_path,
        method=args.method,
        plot_engine=args.plot_engine,
        title=f"{args.manifold.capitalize()} ({args.points}): ISOMAP vs TDC",
        save_name=f"geodesic_path_{args.manifold}_{args.points}"
    )

if __name__ == "__main__":
    main()
