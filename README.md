# Minimax Manifold Learning: Distance Estimation on a Surface

This project is a study and computational extension of the methodologies presented in the research article **"Minimax Estimation of Distances on a Surface and Minimax Manifold Learning in the Isometric-to-Convex Setting"**. 

The core focus of this repository is to practically implement, evaluate, and visualize the theoretical claims made in the paper regarding intrinsic distance estimation. Specifically, it compares two fundamental approaches to geodesic distance estimation on sampled manifolds:

1. **Tangential Delaunay Complex (TDC):** Generating an exact polyhedral surface reconstruction from the point cloud and computing the exact shortest path across its contiguous faces (using an algorithm analogous to Chen and Han).
2. **ISOMAP (k-Nearest Neighbors Graph):** A widely-used baseline that approximates the manifold using a discrete connectivity graph and solves for the shortest path using Dijkstra's algorithm.

The paper establishes that surface-based estimators (like TDC) achieve optimally fast minimax convergence rates under certain geometric assumptions, avoiding the "short-circuiting" phenomenon that plagues k-NN graphs in highly curved spaces.

## Repository Structure

The project relies on a modular Python architecture located in the `src/` directory:

- **`src/generate_data.py`**
  Handles the synthetic generation of diverse 3D manifolds to test the algorithms under varying curvatures and topologies. Supported datasets include:
  - `sphere`: A simple, constant positive curvature manifold.
  - `torus`: A manifold with variable curvature (positive, zero, negative).
  - `swiss`: The classic Swiss Roll, testing algorithm robustness to tightly coiled adjacent sheets.
  - `knot`: A complex tubular knot surface that aggressively penalizes metric short-circuits.

- **`src/TDC.py`**
  The core of the paper's advanced methodology. It takes a raw 3D point cloud, reconstructs the Tangential Delaunay Complex using the `gudhi` library, and applies strict topological 2-manifold pruning to eliminate intersecting faces. Finally, it calculates the mathematically exact intrinsic shortest path across the resulting polyhedral surface using `pygeodesic`.

- **`src/ISOMAP.py`**
  The baseline implementation. It computes a standard k-Nearest Neighbors adjacency graph and leverages `scipy.sparse.csgraph.dijkstra` to find the shortest discrete path. 

- **`src/geodesic.py`**
  The primary execution script. It unifies the generation, TDC estimation, and ISOMAP baseline. Crucially, it provides a high-performance visualizer capable of generating both static Matplotlib `.pdf` reports and fully interactive WebGL Plotly `.html` environments for inspecting the paths and mesh topology.

## Prerequisites & Installation

The project requires several low-level computational geometry libraries to accurately replicate the paper's exact geodesic mathematics.

Install the dependencies:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- `gudhi`: For building the Tangential Delaunay Complex.
- `pygeodesic`: For Exact Polyhedral Geodesics (Mitchell-Mount-Papadopoulou algorithm).
- `scikit-learn` & `scipy`: For the ISOMAP K-NN baseline.
- `plotly` & `matplotlib`: For 3D visualization.

## Usage

The primary entry point is `src/geodesic.py`. It is driven entirely by command-line arguments to allow rapid experimentation with parameters discussed in the paper (such as sampling density, graph connectivity, and maximum edge constraints).

```bash
python src/geodesic.py --manifold torus --n_points 1500 --points random --k 12 --max_edge 0.5
```

### Command Line Arguments:
- `--manifold` : The dataset to test (`sphere`, `torus`, `swiss`, `knot`).
- `--n_points` : The sampling density size ($N$) of the point cloud.
- `--k` : The number of neighbors for the ISOMAP graph construction.
- `--max_edge` : The critical *Maximum Squared Edge Length* parameter for the TDC builder. As emphasized in the paper, this must be empirically tuned based on sampling density to prevent GUDHI from connecting disparate topological sheets (e.g., across the inner radius of a torus).
- `--method` : Choose whether to run `isomap`, `tdc`, or `both`.
- `--plot_engine`: Choose the rendering engine (`plotly` for interactive HTML, `matplotlib` for static PDF, or `both`).
- `--points` : Strategy for choosing the starting and ending target points (`fixed` or `random`).

### Output Visualizations
Outputs are saved automatically to the `images/` directory:
- **Interactive (`.html`)**: Opens instantly in your web browser. It allows you to freely rotate, zoom, and inspect the topological continuity of the TDC mesh and exactly verify the geodesic paths.
- **Static (`.pdf`)**: A publication-ready flat plot generated via Matplotlib.

## Tuning the `--max_edge` Parameter

As detailed in the paper, creating an Exact Polyhedral surface from raw points is mathematically rigorous. If `--max_edge` is too large (or unspecified), GUDHI will bridge massive topological gaps, creating hundreds of intersecting/overlapping 3D triangles. Exact path solvers cannot operate on non-manifold geometries and will fail.

The script implements a robust topological greedy pruning algorithm that mathematically deletes overlapping faces before feeding the mesh to the solver. However, to prevent your surface from becoming disconnected during pruning, you must proactively tune `--max_edge` down until the mesh perfectly traces the manifold, or increase `--n_points` to give the algorithm denser context.
