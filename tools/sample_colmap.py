import os
import shutil
from pathlib import Path
import re

def read_cameras_txt(file_path):
    """Read cameras.txt file"""
    cameras = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        parts = line.split()
        camera_id = int(parts[0])
        cameras[camera_id] = line
    
    return cameras

def read_images_txt(file_path):
    """Read images.txt file"""
    images = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('#') or not line:
            i += 1
            continue
        
        # First line contains image information
        parts = line.split()
        image_id = int(parts[0])
        image_name = parts[-1]  # Last element is the image name
        
        # Second line contains 2D point information
        points2d_line = ""
        if i + 1 < len(lines):
            points2d_line = lines[i + 1].strip()
        
        images[image_id] = {
            'image_line': line,
            'points2d_line': points2d_line,
            'image_name': image_name
        }
        
        i += 2  # Skip two lines
    
    return images

def read_points3d_txt(file_path):
    """Read points3D.txt file"""
    points3d = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue
        parts = line.split()
        point3d_id = int(parts[0])
        points3d[point3d_id] = line
    
    return points3d

def get_subset_image_names(subset_images_dir):
    """Get list of image filenames in the subset directory"""
    subset_dir = Path(subset_images_dir)
    if not subset_dir.exists():
        raise FileNotFoundError(f"Subset images directory does not exist: {subset_images_dir}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_names = []
    
    for file_path in subset_dir.iterdir():
        if file_path.suffix in image_extensions:
            image_names.append(file_path.name)
    
    print(f"Found {len(image_names)} subset images")
    return set(image_names)

def extract_points2d_ids(points2d_line):
    """Extract POINT3D_ID from points2D line"""
    if not points2d_line:
        return set()
    
    parts = points2d_line.split()
    point3d_ids = set()
    
    # points2D format: X Y POINT3D_ID, every 3 numbers form a group
    for i in range(2, len(parts), 3):
        if i < len(parts):
            point3d_id = int(parts[i])
            if point3d_id != -1:  # -1 indicates no corresponding 3D point
                point3d_ids.add(point3d_id)
    
    return point3d_ids

def filter_points3d_by_visibility(points3d, visible_point3d_ids):
    """Filter 3D points by visibility and update track information"""
    filtered_points3d = {}
    
    for point3d_id, line in points3d.items():
        if point3d_id not in visible_point3d_ids:
            continue
            
        parts = line.split()
        if len(parts) < 8:
            continue
            
        # Extract track information (starting from 8th element, every 2 numbers form a pair IMAGE_ID POINT2D_IDX)
        track_parts = parts[8:]
        new_track = []
        
        for i in range(0, len(track_parts), 2):
            if i + 1 < len(track_parts):
                image_id = int(track_parts[i])
                point2d_idx = track_parts[i + 1]
                # Here we simplify the processing and keep all track information
                # In practice, you might need to further filter based on subset_image_ids
                new_track.extend([str(image_id), point2d_idx])
        
        # Reconstruct points3D line
        base_info = parts[:8]  # POINT3D_ID, X, Y, Z, R, G, B, ERROR
        new_line = ' '.join(base_info + new_track)
        filtered_points3d[point3d_id] = new_line
    
    return filtered_points3d

def create_colmap_subset(original_sparse_dir, subset_images_dir, output_dir):
    """Create a subset of COLMAP results"""
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read original COLMAP files
    cameras_file = Path(original_sparse_dir) / 'cameras.txt'
    images_file = Path(original_sparse_dir) / 'images.txt'
    points3d_file = Path(original_sparse_dir) / 'points3D.txt'
    
    print("Reading original COLMAP files...")
    cameras = read_cameras_txt(cameras_file)
    images = read_images_txt(images_file)
    points3d = read_points3d_txt(points3d_file)
    
    print(f"Original data: {len(cameras)} cameras, {len(images)} images, {len(points3d)} 3D points")
    
    # Get subset image names
    subset_image_names = get_subset_image_names(subset_images_dir)
    
    # Filter images
    subset_images = {}
    used_camera_ids = set()
    visible_point3d_ids = set()
    
    for image_id, image_info in images.items():
        if image_info['image_name'] in subset_image_names:
            subset_images[image_id] = image_info
            
            # Extract used camera IDs
            parts = image_info['image_line'].split()
            camera_id = int(parts[8])
            used_camera_ids.add(camera_id)
            
            # Extract visible 3D point IDs
            point3d_ids = extract_points2d_ids(image_info['points2d_line'])
            visible_point3d_ids.update(point3d_ids)
    
    print(f"Subset contains: {len(subset_images)} images")
    print(f"Using {len(used_camera_ids)} cameras")
    print(f"Visible {len(visible_point3d_ids)} 3D points")
    
    # Filter cameras
    subset_cameras = {cam_id: cameras[cam_id] for cam_id in used_camera_ids if cam_id in cameras}
    
    # Filter 3D points
    subset_points3d = filter_points3d_by_visibility(points3d, visible_point3d_ids)
    
    # Write new files
    print("Writing subset files...")
    
    # Write cameras.txt
    with open(output_path / 'cameras.txt', 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(subset_cameras)}\n")
        for camera_id in sorted(subset_cameras.keys()):
            f.write(subset_cameras[camera_id] + '\n')
    
    # Write images.txt
    with open(output_path / 'images.txt', 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(subset_images)}, mean observations per image: N/A\n")
        for image_id in sorted(subset_images.keys()):
            f.write(subset_images[image_id]['image_line'] + '\n')
            f.write(subset_images[image_id]['points2d_line'] + '\n')
    
    # Write points3D.txt
    with open(output_path / 'points3D.txt', 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(subset_points3d)}, mean track length: N/A\n")
        for point3d_id in sorted(subset_points3d.keys()):
            f.write(subset_points3d[point3d_id] + '\n')
    
    print(f"Subset COLMAP files saved to: {output_dir}")
    print(f"Final statistics: {len(subset_cameras)} cameras, {len(subset_images)} images, {len(subset_points3d)} 3D points")

def convert_to_binary_format(sparse_txt_dir, colmap_executable="colmap"):
    """Convert COLMAP text format to binary format using colmap model_converter"""
    
    # Create sparse/0 directory in the same parent directory as sparse_txt
    sparse_txt_path = Path(sparse_txt_dir)
    parent_dir = sparse_txt_path.parent
    sparse_binary_dir = parent_dir / "sparse" / "0"
    
    # Create the directory structure
    sparse_binary_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting COLMAP format from TXT to BIN...")
    print(f"Input (TXT): {sparse_txt_dir}")
    print(f"Output (BIN): {sparse_binary_dir}")
    
    # Construct the colmap command
    cmd = [
        colmap_executable,
        "model_converter",
        "--input_path", str(sparse_txt_path),
        "--output_path", str(sparse_binary_dir),
        "--output_type", "BIN"
    ]
    
    try:
        # Run the colmap command
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("COLMAP model conversion completed successfully!")
        print(f"Binary files saved to: {sparse_binary_dir}")
        return str(sparse_binary_dir)
        
    except subprocess.CalledProcessError as e:
        print(f"Error running COLMAP model_converter:")
        print(f"Command: {' '.join(cmd)}")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    except FileNotFoundError:
        print(f"Error: COLMAP executable not found. Please ensure 'colmap' is installed and in your PATH.")
        print(f"Tried to run: {colmap_executable}")
        print(f"You can specify a custom path by setting the colmap_executable parameter.")
        raise

# Usage example
if __name__ == "__main__":
    # Set paths
    original_sparse_dir = "path_to_original_sparse_txt"
    subset_images_dir = "path_to_smaller_subset_images"
    output_dir = "path_to_save_smaller_subset_sparse_txt"
    
    try:
        # Step 1: Create COLMAP subset
        create_colmap_subset(original_sparse_dir, subset_images_dir, output_dir)
        print("Subset extraction completed!")
        
        # Step 2: Convert to binary format
        binary_output_dir = convert_to_binary_format(output_dir)
        print(f"Complete pipeline finished! Binary COLMAP files available at: {binary_output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()