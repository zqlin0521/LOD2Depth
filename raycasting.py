#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Raycasting Module: Generate depth and normal maps using Open3D
"""

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image

def create_mesh_scene(vertices, faces, triangles=None):
    """
    Create Open3D mesh scene for raycasting
    
    Parameters:
        vertices (numpy.ndarray): Vertex coordinates array
        faces (list): Face indices list
        triangles (numpy.ndarray, optional): Triangles array if already available
    
    Returns:
        open3d.geometry.RaycastingScene: Open3D scene for raycasting
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    if triangles is not None:
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
    else:
        # Convert faces to triangles
        triangles = []
        for face in faces:
            if len(face) == 3:
                triangles.append(face)
            elif len(face) > 3:
                # Triangulate n-gon (simple fan triangulation)
                for i in range(1, len(face) - 1):
                    triangles.append([face[0], face[i], face[i+1]])
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # Compute vertex normals for normal map generation
    mesh.compute_vertex_normals()
    
    # Create a scene with the mesh for raycasting
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)
    
    return scene

def generate_depth_normal_maps(scene, camera_intrinsics, camera_extrinsics, width, height):
    """
    Generate depth and normal maps using raycasting
    
    Parameters:
        scene (open3d.t.geometry.RaycastingScene): Raycasting scene
        camera_intrinsics (numpy.ndarray): 3x3 intrinsic matrix
        camera_extrinsics (numpy.ndarray): 4x4 extrinsic matrix
        width (int): Image width
        height (int): Image height
        
    Returns:
        tuple: (depth_map, normal_map, hit_map) as numpy arrays
    """
    # Create rays for raycasting
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=camera_intrinsics,
        extrinsic_matrix=camera_extrinsics,
        width_px=width,
        height_px=height
    )
    
    # Cast rays
    ray_cast_results = scene.cast_rays(rays)
    
    # Extract depth and hit information
    depth_map = ray_cast_results['t_hit'].numpy()
    hit_map = ray_cast_results['primitive_ids'].numpy() != -1
    
    # Extract normals (note: these are in world coordinates)
    normal_map = ray_cast_results['primitive_normals'].numpy()
    
    # Set invalid depths to NaN for easier filtering later
    depth_map[~hit_map] = np.nan
    
    # Set invalid normals to zero
    normal_map[~hit_map] = 0
    
    return depth_map, normal_map, hit_map

def save_depth_normal_maps(depth_map, normal_map, hit_map, output_dir, image_name):
    """
    Save depth and normal maps to files (both raw .npy files and visualized .png files)
    
    Parameters:
        depth_map (numpy.ndarray): Depth map array
        normal_map (numpy.ndarray): Normal map array
        hit_map (numpy.ndarray): Boolean array indicating valid hits
        output_dir (str): Base output directory path
        image_name (str): Original image name for naming output files
    """
    # Create output directories for different types of outputs
    raw_depth_dir = os.path.join(output_dir, 'raw_depth')
    raw_normal_dir = os.path.join(output_dir, 'raw_normal')
    vis_depth_dir = os.path.join(output_dir, 'vis_depth')
    vis_normal_dir = os.path.join(output_dir, 'vis_normal')
    
    os.makedirs(raw_depth_dir, exist_ok=True)
    os.makedirs(raw_normal_dir, exist_ok=True)
    os.makedirs(vis_depth_dir, exist_ok=True)
    os.makedirs(vis_normal_dir, exist_ok=True)
    
    # Get base name without extension
    base_name = os.path.splitext(image_name)[0]
    
    # Save raw depth map as .npy file (preserving actual depth values)
    raw_depth_file = os.path.join(raw_depth_dir, f"{base_name}.npy")
    np.save(raw_depth_file, depth_map)
    
    # Save raw normal map as .npy file
    raw_normal_file = os.path.join(raw_normal_dir, f"{base_name}.npy")
    np.save(raw_normal_file, normal_map)
    
    # Path for depth visualization
    vis_depth_file = os.path.join(vis_depth_dir, f"{base_name}.png")
    
    # Only consider valid depths for visualization
    valid_hits = np.sum(hit_map)
    
    if valid_hits > 0:
        # Extract valid depths (avoiding inf, nan, and extreme values)
        valid_depths = depth_map[hit_map]
        
        # Handle extreme values - filter out inf and very large values
        valid_mask = np.isfinite(valid_depths) & (valid_depths < 1e6)
        
        if np.sum(valid_mask) > 0:
            # Get min and max of valid finite depths
            min_depth = np.min(valid_depths[valid_mask])
            max_depth = np.max(valid_depths[valid_mask])
            
            # Simple linear normalization to [0, 1]
            normalized_depth = np.zeros_like(depth_map)
            
            # Filter out inf and nan when normalizing
            norm_mask = hit_map & np.isfinite(depth_map) & (depth_map < 1e6)
            depth_range = max_depth - min_depth
            
            if depth_range > 0:  # Avoid division by zero
                normalized_depth[norm_mask] = (depth_map[norm_mask] - min_depth) / depth_range
            else:
                normalized_depth[norm_mask] = 0.5  # Set to mid-value if all depths are the same
            
            # Create colored depth map using turbo colormap (better visual distribution for depth)
            from matplotlib import cm
            # Get colormap - turbo is better than viridis for depth perception
            colored_depth = cm.turbo(normalized_depth)
            # Convert to uint8 RGB (removing alpha channel)
            colored_depth_rgb = (colored_depth[:, :, :3] * 255).astype(np.uint8)
            # Save directly as image
            Image.fromarray(colored_depth_rgb).save(vis_depth_file)
        else:
            # Create blank image if no valid finite depths
            blank = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
            Image.fromarray(blank).save(vis_depth_file)
            print(f"Warning: No valid finite depth values for {image_name}")
    else:
        # Create blank image if no valid hits
        blank = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
        Image.fromarray(blank).save(vis_depth_file)
        print(f"Warning: No valid ray hits for {image_name}")
    
    # Create and save normal map visualization
    # Normalize the normal vectors
    normal_length = np.sqrt(np.sum(normal_map**2, axis=2, keepdims=True))
    normal_length[normal_length == 0] = 1  # Avoid division by zero
    normalized_normals = normal_map / normal_length
    
    # Convert to RGB (mapping from [-1, 1] to [0, 255])
    normal_rgb = ((normalized_normals + 1) * 127.5).astype(np.uint8)
    
    # Save normal RGB image directly
    vis_normal_file = os.path.join(vis_normal_dir, f"{base_name}.png")
    Image.fromarray(normal_rgb).save(vis_normal_file)
    
    print(f"Saved depth data to: {raw_depth_file}")
    print(f"Saved depth visualization to: {vis_depth_file}")
    print(f"Saved normal data to: {raw_normal_file}")
    print(f"Saved normal visualization to: {vis_normal_file}")