# How does the Tangential Delaunay Complex (TDC) Algorithm Work?
The TDC algorithm solves a very hard problem: If I just give you a cloud of points in 3D space, how do you know which points should be connected with triangles to form a smooth surface without leaving holes or creating overlapping, messy triangles?

step-by-step of how the GUDHI algorithm operates globally:

Step A: Estimate the Tangent Space (Local Geometry)
For every single point in the point cloud, the algorithm looks at its nearest neighbors. Using Principal Component Analysis (PCA) on those local neighbors, it estimates the Tangent Plane passing through that point.

Intuition: Imagine standing on the Earth (a sphere). The ground around you looks flat (the 2D tangent plane), even though the Earth is 3D.
Step B: Local Delaunay Triangulation
For a given point, the algorithm takes its nearest neighbors in 3D space and perfectly projects them down onto that flat 2D tangent plane we just found. Once they are flat on the 2D plane, it computes a standard 2D Delaunay Triangulation (a well-known algorithm that connects dots into nice, non-skinny triangles). It then maps those 2D triangles back up into 3D space. It repeats this for every single point, creating a "star" of triangles around every point.

Step C: Global Gluing and Inconsistencies
Now we have overlapping "stars" (local patches of triangles) for every point. The hard part is that the star computed for Point A and the star computed for Point B might not perfectly agree on how they should connect. The algorithm "glues" these patches together. If two neighboring points disagree on a triangle, they are flagged as an "inconsistency."

Step D: Perturbation (Fixing the mesh)
To fix the disagreements, the TDC algorithm applies a clever trick: it slightly perturbates (jiggles) the points. By moving the points by microscopically small amounts, it reshuffles the geometry just enough that the local triangulations suddenly snap into agreement. Once all inconsistencies are resolved, you are left with a perfect, water-tight 2-dimensional manifold!