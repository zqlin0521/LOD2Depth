#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to verify the alignment between depth maps and original RGB images
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def verify_alignment(rgb_path, depth_path, output_dir=None, alpha=0.5, save_result=True):
    """
    Verify alignment between RGB image and depth map through visualization
    
    Parameters:
        rgb_path (str): Path to original RGB image
        depth_path (str): Path to depth map visualization
        output_dir (str, optional): Directory to save verification results
        alpha (float): Blending factor for overlay (0-1)
        save_result (bool): Whether to save the visualization results
    """
    # Create output directory if needed
    if save_result and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    print(f"Loading RGB image: {rgb_path}")
    rgb_img = np.array(Image.open(rgb_path))
    
    print(f"Loading depth map: {depth_path}")
    depth_img = np.array(Image.open(depth_path))
    
    # Get image names for output
    rgb_name = os.path.basename(rgb_path)
    depth_name = os.path.basename(depth_path)
    
    # Handle different image sizes (resize depth to match RGB if needed)
    if rgb_img.shape[:2] != depth_img.shape[:2]:
        print(f"Warning: Image sizes don't match. RGB: {rgb_img.shape[:2]}, Depth: {depth_img.shape[:2]}")
        print("Resizing depth map to match RGB image dimensions...")
        depth_img = cv2.resize(depth_img, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # Original RGB
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title('Original RGB Image')
    axes[0, 0].axis('off')
    
    # Depth visualization
    axes[0, 1].imshow(depth_img)
    axes[0, 1].set_title('Depth Map Visualization')
    axes[0, 1].axis('off')
    
    # Overlay - Method 1: Alpha blending
    blended = cv2.addWeighted(rgb_img, alpha, depth_img, 1-alpha, 0)
    axes[1, 0].imshow(blended)
    axes[1, 0].set_title(f'Overlay (Alpha Blending, α={alpha})')
    axes[1, 0].axis('off')
    
    # Overlay - Method 2: Checkerboard pattern
    checkerboard = create_checkerboard_overlay(rgb_img, depth_img)
    axes[1, 1].imshow(checkerboard)
    axes[1, 1].set_title('Checkerboard Pattern Overlay')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save result
    if save_result:
        if output_dir:
            out_path = os.path.join(output_dir, f"alignment_verification_{os.path.splitext(rgb_name)[0]}.png")
        else:
            out_path = f"alignment_verification_{os.path.splitext(rgb_name)[0]}.png"
        
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved verification image to: {out_path}")
    
    # Display the result
    plt.show()
    
    # Create interactive slider for alpha blending (separate window)
    create_interactive_blending(rgb_img, depth_img, rgb_name, depth_name, output_dir, save_result)
    
    return blended, checkerboard

def create_checkerboard_overlay(img1, img2, tile_size=64):
    """
    Create a checkerboard pattern overlay of two images
    
    Parameters:
        img1 (numpy.ndarray): First image
        img2 (numpy.ndarray): Second image
        tile_size (int): Size of checkerboard tiles
        
    Returns:
        numpy.ndarray: Checkerboard overlay image
    """
    # Ensure same shape
    assert img1.shape == img2.shape, "Images must have the same shape"
    
    # Create output image
    result = np.zeros_like(img1)
    
    # Create checkerboard pattern
    for i in range(0, img1.shape[0], tile_size):
        for j in range(0, img1.shape[1], tile_size):
            # Alternate between images based on checkerboard pattern
            if ((i // tile_size) + (j // tile_size)) % 2 == 0:
                result[i:i+tile_size, j:j+tile_size] = img1[i:i+tile_size, j:j+tile_size]
            else:
                result[i:i+tile_size, j:j+tile_size] = img2[i:i+tile_size, j:j+tile_size]
    
    return result

def create_interactive_blending(rgb_img, depth_img, rgb_name, depth_name, output_dir=None, save_result=True):
    """
    Create interactive slider for alpha blending
    
    Parameters:
        rgb_img (numpy.ndarray): RGB image
        depth_img (numpy.ndarray): Depth image
        rgb_name (str): RGB image name
        depth_name (str): Depth image name
        output_dir (str, optional): Directory to save results
        save_result (bool): Whether to save the visualization
    """
    # Create figure for interactive blending
    plt.figure(figsize=(12, 10))
    
    # Initial alpha value
    alpha = 0.5
    
    # Display initial blend
    ax = plt.subplot(1, 1, 1)
    blended = cv2.addWeighted(rgb_img, alpha, depth_img, 1-alpha, 0)
    img_display = ax.imshow(blended)
    ax.set_title(f'Interactive Blending (α={alpha:.2f})\nAdjust slider to change blend')
    ax.axis('off')
    
    # Create slider axes
    ax_slider = plt.axes([0.25, 0.02, 0.5, 0.03])
    from matplotlib.widgets import Slider
    slider = Slider(ax_slider, 'Alpha', 0.0, 1.0, valinit=alpha)
    
    # Update function for slider
    def update(val):
        alpha = slider.val
        blended = cv2.addWeighted(rgb_img, alpha, depth_img, 1-alpha, 0)
        img_display.set_data(blended)
        ax.set_title(f'Interactive Blending (α={alpha:.2f})\nAdjust slider to change blend')
        plt.draw()
    
    slider.on_changed(update)
    
    # Save button
    ax_button = plt.axes([0.8, 0.02, 0.1, 0.03])
    from matplotlib.widgets import Button
    button = Button(ax_button, 'Save')
    
    def save_current(event):
        if save_result and output_dir:
            current_alpha = slider.val
            blended = cv2.addWeighted(rgb_img, current_alpha, depth_img, 1-current_alpha, 0)
            out_path = os.path.join(output_dir, f"blended_alpha{current_alpha:.2f}_{os.path.splitext(rgb_name)[0]}.png")
            plt.imsave(out_path, blended)
            print(f"Saved current blend (α={current_alpha:.2f}) to: {out_path}")
    
    button.on_clicked(save_current)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

def process_directory(rgb_dir, depth_dir, output_dir, pattern=None):
    """
    Process all matching images in directories
    
    Parameters:
        rgb_dir (str): Directory containing RGB images
        depth_dir (str): Directory containing depth visualizations
        output_dir (str): Directory to save verification results
        pattern (str, optional): File pattern to match
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get RGB files
    rgb_files = [f for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if pattern:
        rgb_files = [f for f in rgb_files if pattern in f]
    
    # Process each file
    for rgb_file in rgb_files:
        base_name = os.path.splitext(rgb_file)[0]
        depth_file = f"{base_name}.png"  # Assume depth files are .png
        
        rgb_path = os.path.join(rgb_dir, rgb_file)
        depth_path = os.path.join(depth_dir, depth_file)
        
        if os.path.exists(depth_path):
            print(f"\nProcessing pair: {rgb_file} and {depth_file}")
            verify_alignment(rgb_path, depth_path, output_dir)
        else:
            print(f"Warning: No matching depth file for {rgb_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify alignment between RGB images and depth maps')
    parser.add_argument('--rgb', type=str, default='./data/building1/undistorted/images/DJI_20241217084245_0006_D.JPG',
                        help='Path to RGB image or directory of RGB images')
    parser.add_argument('--depth', type=str, default='./output/building1/vis_depth/DJI_20241217084245_0006_D.png',
                        help='Path to depth map visualization or directory of depth maps')
    parser.add_argument('--output', type=str, default='./output/alignment_verification',
                        help='Directory to save verification results')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha blending factor (0-1)')
    parser.add_argument('--pattern', type=str, help='Pattern to match filenames (when processing directories)')
    parser.add_argument('--batch', action='store_true', help='Process directories in batch mode')
    
    args = parser.parse_args()
    
    # Check if directories or individual files
    is_rgb_dir = os.path.isdir(args.rgb)
    is_depth_dir = os.path.isdir(args.depth)
    
    if args.batch and is_rgb_dir and is_depth_dir:
        # Process directories
        process_directory(args.rgb, args.depth, args.output, args.pattern)
    else:
        # Process single pair
        verify_alignment(args.rgb, args.depth, args.output, args.alpha)