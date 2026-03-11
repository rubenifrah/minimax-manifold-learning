import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from scipy.sparse.csgraph import dijkstra
import os
import io

from PIL import Image

from generate_data import generate_sphere_data, generate_torus_data
from ISOMAP import k_neighbors_graph
from TDC import reconstruct_surface_tdc, compute_tdc_distances
from offset import compute_offset_distances

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


def save_plotly_rotation_gif(fig, save_path, radius=1.8, z_height=0.6, angle_step=1.5,
                             width=800, height=640, fps=15):
    """Render a full camera orbit of a Plotly 3D figure to an animated GIF."""
    frames = []
    frame_duration_ms = max(1, int(round(1000 / fps)))

    try:
        import kaleido
    except ImportError as exc:
        raise RuntimeError(
            "Plotly GIF export requires the 'kaleido' package. Install dependencies from "
            "requirements.txt and rerun with --save_gif."
        ) from exc

    for angle in np.arange(0, 360, angle_step):
        theta = np.deg2rad(angle)
        fig.update_layout(scene_camera=dict(
            eye=dict(x=radius * np.cos(theta), y=radius * np.sin(theta), z=z_height)
        ))
        try:
            img_bytes = fig.to_image(format="png", width=width, height=height)
        except Exception as exc:
            raise RuntimeError(
                "Plotly static image export failed. Ensure Chrome or Chromium is installed "
                "for Kaleido, then retry. If needed, run `plotly_get_chrome`."
            ) from exc
        frames.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

    if not frames:
        raise RuntimeError("No frames were generated for the Plotly GIF export.")

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0
    )


def save_plotly_static_pdf(fig, save_path, width=900, height=700):
    """Export a Plotly figure to a static PDF via Kaleido."""
    try:
        import kaleido  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Plotly PDF export requires the 'kaleido' package. Install dependencies from "
            "requirements.txt and rerun."
        ) from exc

    try:
        fig.write_image(save_path, format="pdf", width=width, height=height)
    except Exception as exc:
        raise RuntimeError(
            "Plotly PDF export failed. Ensure Chrome or Chromium is installed for Kaleido, "
            "then retry. If needed, run `plotly_get_chrome`."
        ) from exc


def set_equal_3d_axes(ax, points, padding_ratio=0.05):
    """Force equal data scaling across x, y, and z without drawing a 3D box."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    spans = maxs - mins
    max_span = np.max(spans)
    half_range = 0.5 * max_span

    if max_span <= 0:
        half_range = 0.5
    else:
        half_range *= 1.0 + padding_ratio

    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)
    ax.set_box_aspect((1, 1, 1))


def slerp_path_on_sphere(p0, p1, radius, n_samples=200):
    """Sample the exact great-circle geodesic between two sphere points."""
    dot_product = np.clip(np.dot(p0, p1) / (radius ** 2), -1.0, 1.0)
    angle = np.arccos(dot_product)

    if angle < 1e-8:
        return 0.0, np.vstack([p0, p1])

    t_values = np.linspace(0, 1, n_samples)
    sin_angle = np.sin(angle)
    path = np.array([
        (np.sin((1 - t) * angle) / sin_angle) * p0 + (np.sin(t * angle) / sin_angle) * p1
        for t in t_values
    ])
    return radius * angle, path


def make_near_antipodal_sphere_points(radius, polar_offset_deg, azimuth_deg):
    """
    Construct a nearly antipodal pair on the sphere.

    The start point is the north pole and the end point is moved slightly above the
    south pole with an azimuthal shift to force a unique true geodesic.
    """
    polar_offset = np.deg2rad(polar_offset_deg)
    azimuth = np.deg2rad(azimuth_deg)

    p0 = np.array([0.0, 0.0, radius])
    theta = np.pi - polar_offset
    p1 = radius * np.array([
        np.sin(theta) * np.cos(azimuth),
        np.sin(theta) * np.sin(azimuth),
        np.cos(theta)
    ])

    true_dist, true_path = slerp_path_on_sphere(p0, p1, radius)
    return p0, p1, true_dist, true_path

def plot_geodesic_comparison(X, start_idx, end_idx, k=8, max_edge=None, epsilon=0.2, resolution=50, dynamic_epsilon=False, true_dist=None, true_path=None, method='all', plot_engine='both', save_gif=False, title="Geodesic Estimation", save_name="geodesic_path"):
    """
    General function to compute ISOMAP, TDC, and Offset geodesics, plotting them against an optional true path.
    """
    p0 = X[start_idx]
    p1 = X[end_idx]
    
    run_isomap = method in ['isomap', 'all']
    run_tdc = method in ['tdc', 'all']
    run_offset = method in ['offset', 'all']
    
    has_iso_path = False
    has_tdc_path = False
    has_off_path = False
    
    print(f"\n--- {title} ---")
    if true_dist is not None:
        print(f"True Geodesic Distance: {true_dist:.4f}")
        
    iso_dist, tdc_dist, off_dist = None, None, None
    iso_path_points, tdc_path_points, off_path_points = [], [], []
    valid_voxels, path_spheres = [], []
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

    if run_offset:
        # Compute Aamari et al. D-D Volumetric Offset Geodesic
        off_dist, valid_voxels, off_path_points, path_spheres, final_eps = compute_offset_distances(X, start_idx, end_idx, epsilon, resolution, dynamic_epsilon)
        if len(off_path_points) > 0:
            has_off_path = True
            print(f"Offset Estimated Distance (eps={final_eps:.4f}): {off_dist:.4f}")
            if true_dist is not None:
                err_off = abs(off_dist - true_dist)
                print(f"Offset Absolute Error: {err_off:.4f}")
        else:
            print(f"WARNING: Offset Volume Reconstruction Failed (Path disconnected).")

    print("-" * (8 + len(title)))
    
    if run_isomap and not has_iso_path:
        print(f"WARNING: No ISOMAP path found between points at k={k}. Graph is disconnected.")
    if run_tdc and not has_tdc_path:
        print(f"WARNING: No TDC path found between points. Surface might be disconnected.")
    if run_offset and not has_off_path:
        print(f"WARNING: No Offset path found. Increase --epsilon or --resolution.")
        
    # Build subtitle
    subtitles = []
    if true_dist is not None:
        subtitles.append(f"True: {true_dist:.4f}")
    if run_isomap:
        subtitles.append(f"ISO: {iso_dist:.4f}")
    if run_tdc:
        subtitles.append(f"TDC: {tdc_dist:.4f}" if tdc_dist is not None else "TDC: Fail")
    if run_offset:
        subtitles.append(f"OFF: {off_dist:.4f}" if off_dist is not None else "OFF: Fail")
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
        # Offset shortest path and voxel rendering
        if run_offset:
            # Generate parameterized unit sphere for mathematical ball rendering
            u = np.linspace(0, 2 * np.pi, 15)
            v = np.linspace(0, np.pi, 15)
            x_unit = np.outer(np.cos(u), np.sin(v))
            y_unit = np.outer(np.sin(u), np.sin(v))
            z_unit = np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Draw the exact mathematical epsilon-balls that generated the path
            for sphere_center in path_spheres:
                sx = sphere_center[0] + final_eps * x_unit
                sy = sphere_center[1] + final_eps * y_unit
                sz = sphere_center[2] + final_eps * z_unit
                
                # We use Surface to strictly visualize the overlapping spheres
                fig.add_trace(go.Surface(
                    x=sx, y=sy, z=sz,
                    colorscale=[[0, 'orange'], [1, 'orange']],
                    opacity=0.15,
                    showscale=False,
                    name=f'Union of ε-Balls (ε={final_eps:.3f})',
                    hoverinfo='skip'
                ))

        if has_off_path:
            fig.add_trace(go.Scatter3d(
                x=off_path_points[:, 0], y=off_path_points[:, 1], z=off_path_points[:, 2],
                mode='lines+markers', 
                line=dict(color='yellow', width=6), 
                marker=dict(size=4, color='orange', symbol='cross'),
                name='Geodesic through Balls'
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

        # Avoid blocking batch GIF export on an interactive viewer.
        if plot_engine in ['plotly', 'both'] and not save_gif:
            fig.show()

        save_path_html = os.path.join(output_dir, f"{save_name}.html")
        fig.write_html(save_path_html)
        print(f"Saved interactive Plotly figure to {save_path_html}")

        save_path_plotly_pdf = os.path.join(output_dir, f"{save_name}_plotly.pdf")
        try:
            save_plotly_static_pdf(fig, save_path_plotly_pdf)
            print(f"Saved static Plotly PDF to {save_path_plotly_pdf}")
        except Exception as e:
            print(f"Plotly PDF export failed: {e}")
        
        if save_gif:
            print(f"Generating 360-degree Plotly GIF animation (this may take a minute)...")
            try:
                save_path_gif_plotly = os.path.join(output_dir, f"{save_name}.gif")
                save_plotly_rotation_gif(fig, save_path_gif_plotly)
                print(f"Saved animated Plotly GIF to {save_path_gif_plotly}")
            except Exception as e:
                import traceback
                print(f"Plotly GIF generation failed: {e}")
                traceback.print_exc()

            
    if plot_engine in ['matplotlib', 'both']:
        # Plotting using Matplotlib
        fig_mpl = plt.figure(figsize=(10, 8))
        ax = fig_mpl.add_subplot(111, projection='3d')
        
        # --- MAKE MATPLOTLIB LOOK AS CLEAN AS PLOTLY ---
        # Plotly has a transparent, floating canvas feel. We disable all the noisy Matplotlib framing.
        ax.set_axis_off() 
        fig_mpl.patch.set_facecolor('white')
        ax.set_facecolor('white')
        fig_mpl.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # -----------------------------------------------
        
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
        # Offset shortest path and mathematical sphere rendering
        if run_offset:
            # Generate unit sphere geometry for Matplotlib 3D surfaces
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x_unit = np.outer(np.cos(u), np.sin(v))
            y_unit = np.outer(np.sin(u), np.sin(v))
            z_unit = np.outer(np.ones(np.size(u)), np.cos(v))
            
            # Explicitly plot the solid bounding spheres defining the union-of-balls topology
            for sphere_center in path_spheres:
                sx = sphere_center[0] + final_eps * x_unit
                sy = sphere_center[1] + final_eps * y_unit
                sz = sphere_center[2] + final_eps * z_unit
                ax.plot_surface(sx, sy, sz, color='orange', alpha=0.1, zorder=0)

            # We add a hidden scatter purely to generate the legend entry for the spheres
            ax.scatter([], [], [], color='orange', alpha=0.4, label=f'Union of ε-Balls ({final_eps:.3f})')

        if has_off_path:
            ax.plot(off_path_points[:, 0], off_path_points[:, 1], off_path_points[:, 2], 
                    color='yellow', linewidth=3, linestyle='-', marker='X', markersize=4, 
                    label=f'Geodesic through Balls', zorder=13)
                
        # two target points
        ax.scatter(*p0, color='green', s=100, marker='*', label=f'Start Point ({start_idx})', zorder=15)
        ax.scatter(*p1, color='orange', s=100, marker='*', label=f'End Point ({end_idx})', zorder=15)

        set_equal_3d_axes(ax, X)
        
        ax.set_title(f"{title}\n{subtitle}")
        
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
    parser.add_argument('--epsilon', type=float, default=0.2, help="Hausdorff bounding radius for the Volumetric Offset")
    parser.add_argument('--dynamic_epsilon', action='store_true', help="Iteratively grow epsilon from tiny radius until connected")
    parser.add_argument('--resolution', type=int, default=50, help="Grid density for the Volumetric Offset (N^3 voxels)")
    parser.add_argument('--method', type=str, choices=['isomap', 'tdc', 'offset', 'all'], default='all', help="Method(s) to compute geodesic")
    parser.add_argument('--plot_engine', type=str, choices=['plotly', 'matplotlib', 'both'], default='both', help="Which plotting engine to use for generation")
    parser.add_argument('--save_gif', action='store_true', help="Additionally save a 360-degree animated GIF rendered from the Plotly figure")
    parser.add_argument('--points', type=str, choices=['fixed', 'random', 'unstable'], default='fixed', help="How to choose start and end points")
    parser.add_argument('--sphere_polar_offset_deg', type=float, default=8.0, help="For sphere unstable mode: lift the target this many degrees above the antipode")
    parser.add_argument('--sphere_azimuth_deg', type=float, default=18.0, help="For sphere unstable mode: rotate the target around the vertical axis by this many degrees")
    
    args = parser.parse_args()
    
    true_dist = None
    true_path = None
    comparison_label = f"{args.manifold.capitalize()} ({args.points})"
    save_name = f"geodesic_path_{args.manifold}_{args.points}"
    
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
        elif args.points == 'random':
            idx0, idx1 = np.random.choice(len(X_base), 2, replace=False)
            p0 = X_base[idx0]
            p1 = X_base[idx1]
            true_dist, true_path = slerp_path_on_sphere(p0, p1, radius)
        else:
            p0, p1, true_dist, true_path = make_near_antipodal_sphere_points(
                radius=radius,
                polar_offset_deg=args.sphere_polar_offset_deg,
                azimuth_deg=args.sphere_azimuth_deg
            )
            comparison_label = (
                f"Sphere (unstable, polar+{args.sphere_polar_offset_deg:.1f}deg, "
                f"azimuth {args.sphere_azimuth_deg:.1f}deg)"
            )
            save_name = (
                "geodesic_path_sphere_unstable_"
                f"polar_{args.sphere_polar_offset_deg:.1f}_azimuth_{args.sphere_azimuth_deg:.1f}"
            ).replace('.', 'p')

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
        epsilon=args.epsilon,
        resolution=args.resolution,
        dynamic_epsilon=args.dynamic_epsilon,
        true_dist=true_dist, 
        true_path=true_path,
        method=args.method,
        plot_engine=args.plot_engine,
        save_gif=args.save_gif,
        title=f"{comparison_label}: true geodesic vs estimators",
        save_name=save_name
    )

if __name__ == "__main__":
    main()
