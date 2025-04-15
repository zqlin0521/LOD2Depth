#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main entry point: Load 3D mesh model, apply transformation from scene reference frame,
generate depth and normal maps using raycasting, and save results
"""

import os
import argparse
from mesh_loader import load_obj
from reference_frame import load_reference_frame, apply_transformation
from visualization import visualize_mesh
from mesh_saver import save_obj
from camera_loader import load_cameras, load_images, get_camera_intrinsics, get_camera_extrinsics
from raycasting import create_mesh_scene, generate_depth_normal_maps, save_depth_normal_maps

def main():
    """
    Main function: Process input arguments and call functionality from other modules
    """
    parser = argparse.ArgumentParser(description='3D Mesh Transformation and Depth/Normal Map Generation Tool')

    parser.add_argument('--mesh_path', type=str, default='./data/Mesh/TUM_LoD2.obj',
                        help='Input OBJ file path')
    parser.add_argument('--reference_frame_path', type=str, 
                        default='./data/Campus_1179/scene_reference_frame.json',
                        help='Scene reference frame JSON file path')
    parser.add_argument('--z_offset', type=float, default=46.55,
                        help='Additional Z-axis offset')
    parser.add_argument('--visualize', action='store_true', help='Visualize transformed mesh')
    parser.add_argument('--colmap_dir', type=str, 
                        default='./data/Campus_1179/undistorted/sparse_txt',
                        help='Directory containing COLMAP files (cameras.txt, images.txt)')
    parser.add_argument('--generate_maps', action='store_true',
                        help='Generate depth and normal maps using raycasting')


    parser.add_argument('--building_name', type=str, default='building1',
                        help='Building dataset name (e.g., building2, building3)')
    
    base_path = './data'
    
    parser.add_argument('--subset_images_dir', type=str, 
                        default=f'{base_path}/{{args.building_name}}/undistorted/images',
                        help='Directory containing subset images to process')
    parser.add_argument('--reference_frame_path_building', type=str, 
                        default=f'{base_path}/{{args.building_name}}/scene_reference_frame.json',
                        help='Scene reference frame (for building subset) JSON file path')
    parser.add_argument('--output_path', type=str, 
                        default=f'{base_path}/{{args.building_name}}/transformed_mesh.obj',
                        help='Output path for transformed OBJ file')
    parser.add_argument('--output_building_path', type=str, 
                        default=f'{base_path}/{{args.building_name}}/transformed_mesh_{{args.building_name}}.obj',
                        help='Output path for transformed building OBJ file')
    parser.add_argument('--depth_normal_dir', type=str, 
                        default=f'./output/{{args.building_name}}',
                        help='Base output directory for depth and normal maps')
    
    args = parser.parse_args()

    for arg_name, arg_value in vars(args).items():
        if isinstance(arg_value, str) and '{args.building_name}' in arg_value:
            setattr(args, arg_name, arg_value.format(args=args))
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if args.generate_maps:
        os.makedirs(args.depth_normal_dir, exist_ok=True)
    
    # Load OBJ file
    print(f"Loading OBJ file: {args.mesh_path}")
    vertices, faces, normals, texcoords = load_obj(args.mesh_path)
    print(f"Mesh loaded with {len(vertices)} vertices and {len(faces)} faces")
    
    # Load reference frame
    print(f"Loading scene reference frame: {args.reference_frame_path}")
    reference_frame = load_reference_frame(args.reference_frame_path)
    
    # Apply transformation
    print("Applying transformation...")
    # Get transformation parameters from reference frame
    scale = reference_frame["base_to_canonical"]["scale"]
    shift = reference_frame["base_to_canonical"]["shift"]
    # Modify Z-axis offset
    modified_shift = [shift[0], shift[1], shift[2] + args.z_offset]
    swap_xy = reference_frame["base_to_canonical"]["swap_xy"]
    
    # Apply transformation to mesh
    transformed_vertices = apply_transformation(vertices, scale, modified_shift, swap_xy)
    print("Transformation completed")
    
    # Save transformed OBJ
    print(f"Saving transformed OBJ file: {args.output_path}")
    save_obj(args.output_path, transformed_vertices, faces, normals, texcoords)
    print("Saving completed")
    
    # Visualize if requested
    if args.visualize:
        print("Visualizing transformed mesh...")
        visualize_mesh(transformed_vertices, faces)
    
    # Generate depth and normal maps if requested
    if args.generate_maps:
        print("Generating depth and normal maps using raycasting...")
        
        # Load camera information from COLMAP
        cameras_file = os.path.join(args.colmap_dir, 'cameras.txt')
        images_file = os.path.join(args.colmap_dir, 'images.txt')
        
        print(f"Loading camera parameters from: {cameras_file}")
        cameras = load_cameras(cameras_file)
        
        print(f"Loading image parameters from: {images_file}")
        images = load_images(images_file)
        
        # Create mesh scene for raycasting
        print("Creating mesh scene for raycasting...")
        scene = create_mesh_scene(transformed_vertices, faces)
        
        subset_image_files = set()
        if args.subset_images_dir and os.path.exists(args.subset_images_dir):
            print(f"Loading image list from subset directory: {args.subset_images_dir}")
            for file in os.listdir(args.subset_images_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    subset_image_files.add(file)
            print(f"Found {len(subset_image_files)} images in subset directory")
        else:
            print(f"Warning: Subset images directory not found or not specified: {args.subset_images_dir}")
        
        # Process each camera/image
        print(f"Processing images for depth and normal map generation...")
        processed_count = 0
        skipped_count = 0
        for image_id, image_data in images.items():

            if subset_image_files and image_data.name not in subset_image_files:
                skipped_count += 1
                continue
            
            camera = cameras[image_data.camera_id]
            camera_intrinsics = get_camera_intrinsics(camera)
            camera_extrinsics = get_camera_extrinsics(image_data)
            
            print(f"Generating maps for image: {image_data.name}")
            
            # Generate depth and normal maps
            depth_map, normal_map, hit_map = generate_depth_normal_maps(
                scene, camera_intrinsics, camera_extrinsics, camera.width, camera.height)
            
            # Save depth and normal maps (both raw and visualized)
            save_depth_normal_maps(
                depth_map, normal_map, hit_map, args.depth_normal_dir, image_data.name)
            
            processed_count += 1
        
        print(f"Depth and normal map generation completed: {processed_count} images processed, {skipped_count} images skipped")

    print(f"Loading building reference frame: {args.reference_frame_path_building}")
    building_reference_frame = load_reference_frame(args.reference_frame_path_building)
    
    print("Applying building transformation to original mesh...")

    building_scale = building_reference_frame["base_to_canonical"]["scale"]
    building_shift = building_reference_frame["base_to_canonical"]["shift"]
    # Modify Z-axis offset
    modified_building_shift = [building_shift[0], building_shift[1], building_shift[2] + args.z_offset]
    building_swap_xy = building_reference_frame["base_to_canonical"]["swap_xy"]
    
    building_transformed_vertices = apply_transformation(vertices, building_scale, modified_building_shift, building_swap_xy)
    print("Building transformation completed")
    
    print(f"Saving building transformed OBJ file: {args.output_building_path}")
    save_obj(args.output_building_path, building_transformed_vertices, faces, normals, texcoords)
    print("Building transformation saving completed")
    
    print("All operations completed")

if __name__ == "__main__":
    main()