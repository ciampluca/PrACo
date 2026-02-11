"""
Density Map Generation for Multiclass Dataset

This script generates density maps for each class in each image of the multiclass dataset.
The density maps are created using a Gaussian filter, similar to FSC147 format.

Usage:
    python generate_density_maps.py [--output_dir DIR] [--save_png] [--png_dir DIR] [--sigma SIGMA] [--sigma_config FILE]

Arguments:
    --output_dir: Directory to save .npy density map files (default: density_maps)
    --save_png: Flag to save PNG visualizations
    --png_dir: Directory to save PNG files (default: density_maps_png)
    --sigma: Default Gaussian filter sigma value (default: 1.0, used if no sigma_config)
    --sigma_config: JSON file with per-class sigma values (default: densitymap_sigmas_per_class.json)
    
Example sigma config file format (densitymap_sigmas_per_class.json):
{
    "person": 1.5,
    "car": [1.0, 1.5],
    "bicycle": 1.0
}
Note: Sigma can be a single value (scalar) or a list of two values [sigma_y, sigma_x] for anisotropic filtering.
"""

import json
import os
import argparse
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# Images with letterboxing (white bands at top/bottom)
LETTERBOXED_IMAGES = [
    'mcac_22.jpg'
]


def detect_and_crop_letterbox(img_array, threshold=240):
    """
    Detect and return the crop bounds for letterboxed images.
    
    Args:
        img_array: numpy array of image (H x W x C)
        threshold: brightness threshold for white detection (default 240)
    
    Returns:
        (top, bottom): row indices for cropping (top inclusive, bottom exclusive)
    """
    height = img_array.shape[0]
    row_brightness = img_array.mean(axis=(1, 2))  # Average brightness per row
    
    # Find top content boundary
    top = 0
    for i in range(height):
        if row_brightness[i] < threshold:
            top = i
            break
    
    # Find bottom content boundary
    bottom = height
    for i in range(height - 1, -1, -1):
        if row_brightness[i] < threshold:
            bottom = i + 1
            break
    
    return top, bottom


def create_density_map(points, original_height, original_width, sigma=1.0, target_height=384):
    """
    Create a density map from point annotations using Gaussian filtering.
    Points are assumed to be annotated at 384px height scale.
    
    Args:
        points: List of [x, y] coordinates (in 384px height scale)
        original_height: Original image height (only used to calculate aspect ratio)
        original_width: Original image width (only used to calculate aspect ratio)
        sigma: Gaussian filter sigma value (can be scalar or tuple)
        target_height: Target height for density map (default 384, matching annotation scale)
    
    Returns:
        density_map: numpy array of shape (target_height, target_width)
    """
    # Calculate target dimensions maintaining aspect ratio of ORIGINAL image
    aspect_ratio = original_width / original_height
    target_width = int(16 * round((target_height * aspect_ratio) / 16))  # Round to multiple of 16
    
    # Points are already at 384px height scale, no need to scale them
    # But we need to scale them to target dimensions if target_height != 384
    scale_factor = target_height / 384.0
    
    # Initialize density map with zeros
    density_map = np.zeros((target_height, target_width), dtype='float32')
    
    # Place points on the density map
    for point in points:
        x, y = point
        # Scale points if target is not 384
        x_scaled = x * scale_factor
        y_scaled = y * scale_factor
        
        # Ensure coordinates are within bounds
        x_int = min(target_width - 1, max(0, int(round(x_scaled))))
        y_int = min(target_height - 1, max(0, int(round(y_scaled))))
        density_map[y_int, x_int] += 1.0
    
    # Apply Gaussian filter (sigma can be scalar or tuple)
    density_map = ndimage.gaussian_filter(density_map, sigma=sigma, order=0)
    
    return density_map


def save_density_map_png(density_map, output_path, class_name, image_path=None):
    """
    Save density map as PNG visualization with optional image background.
    
    Args:
        density_map: numpy array of density values
        output_path: Path to save PNG file
        class_name: Class name for title
        image_path: Optional path to original image for background
    """
    plt.figure(figsize=(12, 10))
    
    # Get density map dimensions
    height, width = density_map.shape
    
    # If image path provided, load and display the original image
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path)
        # Always resize image to match density map dimensions
        img = img.resize((width, height), Image.LANCZOS)
        plt.imshow(img, alpha=0.6, origin='upper')
    
    # Overlay density map with same origin
    plt.imshow(density_map, cmap='jet', alpha=0.7, origin='upper')
    plt.colorbar(label='Density')
    plt.title(f'Density Map - {class_name}', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_composite_density_map(density_maps_dict, output_path, image_path=None):
    """
    Save composite visualization with all class density maps in different colors.
    
    Args:
        density_maps_dict: Dictionary mapping class names to density maps
        output_path: Path to save PNG file
        image_path: Optional path to original image for background
    """
    # Define a colormap for different classes
    colors = plt.cm.tab20(np.linspace(0, 1, 20))  # 20 distinct colors
    if len(density_maps_dict) > 20:
        # If more than 20 classes, extend with more colormaps
        colors = np.vstack([
            plt.cm.tab20(np.linspace(0, 1, 20)),
            plt.cm.tab20b(np.linspace(0, 1, 20)),
            plt.cm.tab20c(np.linspace(0, 1, 20))
        ])
    
    plt.figure(figsize=(14, 12))
    
    # Get density map dimensions
    first_density_map = list(density_maps_dict.values())[0]
    height, width = first_density_map.shape
    
    # If image path provided, load and display the original image
    if image_path and os.path.exists(image_path):
        img = Image.open(image_path)
        # Always resize image to match density map dimensions
        img = img.resize((width, height), Image.LANCZOS)
        plt.imshow(img, alpha=0.5, origin='upper')
    
    # Create composite density map with different colors for each class
    composite = np.zeros((height, width, 4))  # RGBA
    
    for idx, (class_name, density_map) in enumerate(density_maps_dict.items()):
        # Normalize density map to [0, 1]
        if density_map.max() > 0:
            normalized = density_map / density_map.max()
        else:
            normalized = density_map
        
        # Apply class-specific color
        color = colors[idx % len(colors)]
        for c in range(3):  # RGB channels
            composite[:, :, c] = np.maximum(composite[:, :, c], normalized * color[c])
        composite[:, :, 3] = np.maximum(composite[:, :, 3], normalized * 0.7)  # Alpha
    
    plt.imshow(composite, origin='upper')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[idx % len(colors)], label=class_name) 
                      for idx, class_name in enumerate(density_maps_dict.keys())]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
              fontsize=10, framealpha=0.9)
    
    plt.title('Composite Density Map - All Classes', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate density maps for multiclass dataset')
    parser.add_argument('--output_dir', type=str, default='density_maps',
                        help='Directory to save .npy density map files')
    parser.add_argument('--save_png', action='store_true',
                        help='Save PNG visualizations of density maps')
    parser.add_argument('--png_dir', type=str, default='density_maps_png',
                        help='Directory to save PNG files')
    parser.add_argument('--composite_png_dir', type=str, default='density_maps_composite_png',
                        help='Directory to save composite PNG files with all classes')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Default Gaussian filter sigma value (used if no sigma_config)')
    parser.add_argument('--sigma_config', type=str, default='densitymap_sigmas_per_class.json',
                        help='JSON file with per-class sigma values')
    parser.add_argument('--target_height', type=int, default=384,
                        help='Target height for density maps (default: 384, matching annotation scale)')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='Directory containing annotations.json and class_mapping.json')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_png:
        os.makedirs(args.png_dir, exist_ok=True)
        os.makedirs(args.composite_png_dir, exist_ok=True)
    
    # Load annotations and class mapping
    annotations_path = os.path.join(args.data_dir, 'annotations.json')
    class_mapping_path = os.path.join(args.data_dir, 'class_mapping.json')
    sigma_config_path = os.path.join(args.data_dir, args.sigma_config)
    
    print(f"Loading annotations from {annotations_path}...")
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    print(f"Loading class mapping from {class_mapping_path}...")
    with open(class_mapping_path, 'r') as f:
        class_mapping = json.load(f)
    
    # Load per-class sigma configuration if available
    sigma_per_class = {}
    if os.path.exists(sigma_config_path):
        print(f"Loading sigma configuration from {sigma_config_path}...")
        with open(sigma_config_path, 'r') as f:
            sigma_config = json.load(f)
            # Convert sigma values to appropriate format (scalar or tuple)
            for class_name, sigma_value in sigma_config.items():
                if isinstance(sigma_value, list):
                    sigma_per_class[class_name] = tuple(sigma_value)
                else:
                    sigma_per_class[class_name] = sigma_value
        print(f"  Loaded sigma values for {len(sigma_per_class)} classes")
    else:
        print(f"Sigma config file not found at {sigma_config_path}, using default sigma={args.sigma}")
    
    print(f"\nDataset statistics:")
    print(f"  Total images: {len(annotations)}")
    print(f"  Total classes: {len(class_mapping)}")
    print(f"  Target density map height: {args.target_height}px")
    print(f"  Default Gaussian sigma: {args.sigma}")
    if sigma_per_class:
        print(f"  Per-class sigmas loaded: {len(sigma_per_class)} classes")
    print(f"  Output directory: {args.output_dir}")
    if args.save_png:
        print(f"  PNG directory: {args.png_dir}")
        print(f"  Composite PNG directory: {args.composite_png_dir}")
    
    # Statistics
    total_density_maps = 0
    images_processed = 0
    
    # Process each image
    print("\nGenerating density maps...")
    for img_name, img_data in tqdm(annotations.items(), desc="Processing images"):
        # NOTE: W and H in annotations are original image dimensions, NOT scaled dimensions
        # Points are at 384px scale, so we need to get actual image dimensions
        img_path = os.path.join(args.data_dir, 'images', img_name)
        
        if os.path.exists(img_path):
            img = Image.open(img_path)
            
            # Handle letterboxed images
            if img_name in LETTERBOXED_IMAGES:
                img_array = np.array(img)
                top, bottom = detect_and_crop_letterbox(img_array)
                img = Image.fromarray(img_array[top:bottom, :, :])
                # Use cropped dimensions
                width, height = img.size
            else:
                width, height = img.size  # Get actual original image dimensions
        else:
            # Fallback to annotation dimensions if image not found
            height = img_data['H']
            width = img_data['W']
            img_path = None
        
        points = img_data['points']
        classes = img_data['classes']
        
        # Group points by class
        class_points = defaultdict(list)
        for point, class_id in zip(points, classes):
            class_points[class_id].append(point)
        
        # Dictionary to store density maps for composite visualization
        density_maps_for_composite = {}
        
        # Generate density map for each class
        for class_id, class_pts in class_points.items():
            class_name = class_mapping[str(class_id)]
            
            # Get sigma for this class (use per-class if available, otherwise default)
            sigma = sigma_per_class.get(class_name, args.sigma)
            
            # Create density map (points are at 384px scale)
            density_map = create_density_map(class_pts, height, width, sigma=sigma, target_height=args.target_height)
            
            # Save as .npy file
            img_base = os.path.splitext(img_name)[0]
            npy_filename = f"{img_base}_{class_name}.npy"
            npy_path = os.path.join(args.output_dir, npy_filename)
            np.save(npy_path, density_map)
            
            # Store for composite visualization
            density_maps_for_composite[class_name] = density_map
            
            # Optionally save as PNG with image background
            if args.save_png:
                png_filename = f"{img_base}_{class_name}.png"
                png_path = os.path.join(args.png_dir, png_filename)
                save_density_map_png(density_map, png_path, class_name, image_path=img_path)
            
            total_density_maps += 1
        
        # Save composite visualization with all classes
        if args.save_png and len(density_maps_for_composite) > 0:
            composite_filename = f"{img_base}_composite.png"
            composite_path = os.path.join(args.composite_png_dir, composite_filename)
            save_composite_density_map(density_maps_for_composite, composite_path, image_path=img_path)
        
        images_processed += 1
    
    # Print summary
    print("\n" + "="*60)
    print("DENSITY MAP GENERATION SUMMARY")
    print("="*60)
    print(f"Images processed: {images_processed}")
    print(f"Total density maps generated: {total_density_maps}")
    print(f"Average density maps per image: {total_density_maps / images_processed:.2f}")
    print(f"\nDensity maps saved to: {args.output_dir}")
    if args.save_png:
        print(f"Per-class PNG visualizations saved to: {args.png_dir}")
        print(f"Composite PNG visualizations saved to: {args.composite_png_dir}")
    print("\nDensity map generation complete!")


if __name__ == '__main__':
    main()
