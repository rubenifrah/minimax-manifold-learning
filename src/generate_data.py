# file to generate point clouds corresponding to classical manifolds 
# in the isometric-to-convex setting

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# sphere
def generate_sphere_data(n_points, radius=1.0):
    # generate random points on the unit sphere
    points = np.random.randn(n_points, 3)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    # scale the points to the desired radius
    points = radius * points
    return points


# torus
def generate_torus_data(n_points, R=1.0, r=0.5):
    # generate random points on the torus
    points = np.zeros((n_points, 3))
    for i in range(n_points):
        theta = 2 * np.pi * np.random.rand()
        phi = 2 * np.pi * np.random.rand()
        points[i, 0] = (R + r * np.cos(theta)) * np.cos(phi)
        points[i, 1] = (R + r * np.cos(theta)) * np.sin(phi)
        points[i, 2] = r * np.sin(theta)
    return points

# Swiss 
def generate_swiss_data(n_points, t_max=2*np.pi, h_max=2*np.pi):
    # generate random points on the swiss roll
    # equation
    # x(t,h) = t cos(t)
    # y(t,h) = t sin(t)
    # z(t,h) = h
    # t represents the angle of rotation (1 rotation <=> t = 2pi)
    # h represents the height of the manifold
    
    points = np.zeros((n_points, 3))
    for i in range(n_points):
        t = t_max * np.random.rand()
        h = h_max * np.random.rand()
        points[i, 0] = t * np.cos(t)
        points[i, 1] = t * np.sin(t)
        points[i, 2] = h
    return points

def save_point_cloud(points, filename, data_dir='data', params=None):
    np.savetxt(f'{data_dir}/{filename}_{params}.txt', points)

def load_point_cloud(filename, data_dir='data', params=None):
    return np.loadtxt(f'{data_dir}/{filename}_{params}.txt')

def main():
    # generate sphere data
    n_points = 1000
    radius = 1.0
    points = generate_sphere_data(n_points, radius)
    # save the point cloud
    data_dir = 'data'
    save_point_cloud(points, 'sphere_data', data_dir, params=f'radius_{radius}_n_points_{n_points}')
    # load the point cloud
    points = load_point_cloud('sphere_data', data_dir, params=f'radius_{radius}_n_points_{n_points}')
    # plot the point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_title(f'Sphere Data - Radius: {radius}, N Points: {n_points}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    plt.show()
    
if __name__ == '__main__':
    main()