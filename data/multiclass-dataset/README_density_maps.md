# Density Map Generation for Multiclass Dataset

## Overview
The `generate_density_maps.py` script creates Gaussian-filtered density maps for each class in each image of the multiclass dataset. These density maps are useful for training and evaluating object counting models.

## Files

### Input Files
- **annotations.json**: Contains image annotations with point coordinates and class IDs
- **class_mapping.json**: Maps class IDs to class names
- **densitymap_sigmas_per_class.json**: (Optional) Per-class sigma values for Gaussian filtering

### Output Files
- **density_maps/*.npy**: Density map files in NumPy format (one per class per image)
- **density_maps_png/*.png**: (Optional) PNG visualizations of density maps

## Usage

### Basic Usage
Generate density maps with default sigma (1.0) for all classes:
```bash
python generate_density_maps.py
```

### Generate with PNG Visualizations
```bash
python generate_density_maps.py --save_png
```

### Custom Output Directories
```bash
python generate_density_maps.py --output_dir my_density_maps --png_dir my_visualizations --save_png
```

### Custom Default Sigma
```bash
python generate_density_maps.py --sigma 1.5
```

### Using Per-Class Sigma Configuration
The script will automatically use `densitymap_sigmas_per_class.json` if it exists in the data directory. You can also specify a custom config file:
```bash
python generate_density_maps.py --sigma_config my_custom_sigmas.json
```

## Sigma Configuration File Format

The `densitymap_sigmas_per_class.json` file allows you to specify different Gaussian filter sigma values for each class. This is useful when different object types require different levels of smoothing.

### Format
```json
{
  "class_name_1": sigma_value,
  "class_name_2": [sigma_y, sigma_x],
  "class_name_3": sigma_value
}
```

### Examples

**Scalar sigma (isotropic Gaussian):**
```json
{
  "people": 1.5,
  "cars": 2.0,
  "bicycles": 1.0
}
```

**Tuple sigma (anisotropic Gaussian):**
```json
{
  "people": [1.5, 1.0],
  "cars": [2.0, 1.5]
}
```
The tuple format `[sigma_y, sigma_x]` allows different smoothing in vertical and horizontal directions.

**Mixed format:**
```json
{
  "people": 1.5,
  "cars": [2.0, 1.5],
  "bicycles": 1.0
}
```

### Editing Sigma Values

1. Open `densitymap_sigmas_per_class.json`
2. Modify the sigma values for specific classes
3. Save the file
4. Run the density map generation script

Example modifications:
```json
{
  "plates": 1.2,
  "bowls": 1.5,
  "cups": 1.0,
  "people": [2.0, 1.5],
  "cars": [2.5, 2.0]
}
```

## Output File Naming

Density maps are saved with the format:
```
{image_name_without_extension}_{class_name}.npy
{image_name_without_extension}_{class_name}.png  (if --save_png is used)
```

### Example
For image `mcac_1.jpg` with classes "people" and "cars":
- `mcac_1_people.npy`
- `mcac_1_cars.npy`
- `mcac_1_people.png` (if --save_png)
- `mcac_1_cars.png` (if --save_png)

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| --output_dir | str | density_maps | Directory for .npy files |
| --save_png | flag | False | Save PNG visualizations |
| --png_dir | str | density_maps_png | Directory for PNG files |
| --sigma | float | 1.0 | Default sigma value |
| --sigma_config | str | densitymap_sigmas_per_class.json | Sigma config file |
| --data_dir | str | . | Data directory |

## Notes

- The script uses `scipy.ndimage.gaussian_filter` for density map generation
- Density maps preserve the original image dimensions (H x W)
- Points are placed at integer coordinates (rounded) before Gaussian filtering
- If a sigma config file exists, per-class values override the default --sigma
- Classes without entries in the sigma config will use the default --sigma value
