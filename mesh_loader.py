#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mesh Loader: Responsible for loading 3D meshes from OBJ files
"""

import numpy as np

def load_obj(file_path):
    """
    Load vertices, faces, normals, and texture coordinates from an OBJ file
    
    Parameters:
        file_path (str): Path to the OBJ file
        
    Returns:
        tuple: (vertices list, faces list, normals list, texture coordinates list)
    """
    vertices = []
    faces = []
    normals = []
    texcoords = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):  # Skip comments
                continue
                
            values = line.split()
            if not values:
                continue
                
            if values[0] == 'v':  # Vertex
                vertices.append([float(x) for x in values[1:4]])
            elif values[0] == 'vn':  # Normal
                normals.append([float(x) for x in values[1:4]])
            elif values[0] == 'vt':  # Texture coordinate
                texcoords.append([float(x) for x in values[1:3]])
            elif values[0] == 'f':  # Face
                # OBJ file face format can be f v1 v2 v3 or f v1/vt1 v2/vt2 v3/vt3 or f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                # Here we only extract vertex indices
                face = []
                for v in values[1:]:
                    # Extract vertex index (first number)
                    # OBJ indices start at 1, so subtract 1 to convert to 0-based indexing
                    w = v.split('/')
                    face.append(int(w[0]) - 1)
                faces.append(face)
    
    return np.array(vertices), faces, np.array(normals), np.array(texcoords)