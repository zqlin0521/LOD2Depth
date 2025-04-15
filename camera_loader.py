#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camera Loader Module: Load camera parameters from COLMAP format
"""

import os
import numpy as np
from collections import namedtuple

# Define camera and image data structures
CameraModel = namedtuple('CameraModel', ['id', 'model', 'width', 'height', 'params'])
ImageData = namedtuple('ImageData', ['id', 'qvec', 'tvec', 'camera_id', 'name', 'points2D'])

def load_cameras(cameras_file):
    """
    Load camera intrinsics from COLMAP cameras.txt file
    
    Parameters:
        cameras_file (str): Path to the cameras.txt file
        
    Returns:
        dict: Dictionary mapping camera_id to CameraModel
    """
    cameras = {}
    with open(cameras_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            
            cameras[camera_id] = CameraModel(
                id=camera_id,
                model=model,
                width=width,
                height=height,
                params=params
            )
    
    return cameras

def load_images(images_file):
    """
    Load camera extrinsics from COLMAP images.txt file
    
    Parameters:
        images_file (str): Path to the images.txt file
        
    Returns:
        dict: Dictionary mapping image_id to ImageData
    """
    images = {}
    with open(images_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            
            parts = line.split()
            if len(parts) < 10:
                continue  # Skip invalid lines
                
            image_id = int(parts[0])
            qw, qx, qy, qz = [float(q) for q in parts[1:5]]
            tx, ty, tz = [float(t) for t in parts[5:8]]
            camera_id = int(parts[8])
            name = parts[9]
            
            # Read next line for points2D
            points2D = []
            line = f.readline().strip()
            if line and not line.startswith('#'):
                points_parts = line.split()
                for i in range(0, len(points_parts), 3):
                    if i + 2 < len(points_parts):
                        x, y, point3D_id = float(points_parts[i]), float(points_parts[i+1]), int(points_parts[i+2])
                        points2D.append((x, y, point3D_id))
            
            images[image_id] = ImageData(
                id=image_id,
                qvec=[qw, qx, qy, qz],
                tvec=[tx, ty, tz],
                camera_id=camera_id,
                name=name,
                points2D=points2D
            )
    
    return images

def get_camera_intrinsics(camera):
    """
    Get camera intrinsic matrix based on COLMAP camera model
    
    Parameters:
        camera (CameraModel): Camera model from COLMAP
        
    Returns:
        numpy.ndarray: 3x3 camera intrinsic matrix
    """
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        return K
    elif camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
        return K
    else:
        raise ValueError(f"Camera model {camera.model} not supported yet")

def quaternion_to_rotation_matrix(qvec):
    """
    Convert quaternion to rotation matrix
    
    Parameters:
        qvec (list): Quaternion [w, x, y, z]
        
    Returns:
        numpy.ndarray: 3x3 rotation matrix
    """
    w, x, y, z = qvec
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    return R

def get_camera_extrinsics(image_data):
    """
    Get camera extrinsic matrix based on COLMAP image data
    
    Parameters:
        image_data (ImageData): Image data from COLMAP
        
    Returns:
        numpy.ndarray: 4x4 camera extrinsic matrix (world to camera)
    """
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(image_data.qvec)
    
    # Create translation vector
    t = np.array(image_data.tvec).reshape(3, 1)
    
    # Create 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    
    # COLMAP uses world to camera transform, so we need to invert it
    # to get camera to world transform
    return T