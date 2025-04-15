#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reference Frame Processor: Load scene reference frame and apply transformations
"""

import json
import numpy as np

def load_reference_frame(file_path):
    """
    Load scene reference frame from a JSON file
    
    Parameters:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Reference frame data
    """
    with open(file_path, 'r') as f:
        reference_frame = json.load(f)
    return reference_frame

def apply_transformation(vertices, scale, shift, swap_xy=False):
    """
    Apply transformation to vertices
    
    Parameters:
        vertices (numpy.ndarray): Vertex coordinates array (N, 3)
        scale (list): Scale factors [sx, sy, sz]
        shift (list): Translation vector [tx, ty, tz]
        swap_xy (bool): Whether to swap X and Y coordinates
        
    Returns:
        numpy.ndarray: Transformed vertex coordinates array (N, 3)
    """
    # Convert to numpy arrays for matrix operations
    vertices = np.array(vertices)
    scale = np.array(scale)
    shift = np.array(shift)
    
    # Swap X and Y coordinates if needed
    if swap_xy:
        vertices = vertices[:, [1, 0, 2]]
    
    # Apply scaling
    vertices = vertices * scale
    
    # Apply translation (negative because we need to transform from base to canonical)
    # Note: According to the reference frame definition, shift is from canonical to base,
    # so we need to negate it to go from base to canonical
    vertices = vertices + shift
    
    return vertices