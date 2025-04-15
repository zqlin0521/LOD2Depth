#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Module: Responsible for 3D mesh visualization
"""

import numpy as np
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    print("Warning: matplotlib not installed. Visualization will not be available. Please install with pip install matplotlib.")

def visualize_mesh(vertices, faces, color='lightblue', alpha=0.7, figsize=(10, 8)):
    """
    Visualize 3D mesh using Matplotlib
    
    Parameters:
        vertices (numpy.ndarray): Vertex coordinates array
        faces (list): Face indices list
        color (str): Mesh color
        alpha (float): Transparency
        figsize (tuple): Figure size
    """
    try:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create polygon collection
        mesh = Poly3DCollection([vertices[face] for face in faces], alpha=alpha)
        mesh.set_facecolor(color)
        ax.add_collection3d(mesh)
        
        # Set axis limits
        all_vertices = np.vstack([vertices[face] for face in faces])
        x_min, x_max = np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])
        y_min, y_max = np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])
        z_min, z_max = np.min(all_vertices[:, 2]), np.max(all_vertices[:, 2])
        
        # Ensure equal axis scales
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
        
        # Set title and labels
        ax.set_title('3D Mesh Visualization')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        
        plt.tight_layout()
        plt.show()
    except NameError:
        print("Error: matplotlib could not be properly imported. Visualization is not available.")