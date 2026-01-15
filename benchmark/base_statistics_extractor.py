import os
import numpy as np
from scipy import ndimage


class BaseStatisticsExtractor:
    """Base class containing shared functionality for statistics extraction with localized metrics."""
    
    def __init__(self, gt_density_maps_dir, pred_density_maps_dir):
        """
        Initialize the base statistics extractor.
        
        Args:
            gt_density_maps_dir: Path to ground truth density maps directory
            pred_density_maps_dir: Path to predicted density maps directory
        """
        self.gt_density_maps_dir = gt_density_maps_dir
        self.pred_density_maps_dir = pred_density_maps_dir
    
    def _load_density_map(self, img_filename, class_name=None, is_gt=True):
        """
        Load a density map from disk.
        
        Args:
            img_filename: Image filename (with extension)
            class_name: Class name (optional, used for multiclass)
            is_gt: Whether to load ground truth or predicted density map
            
        Returns:
            Density map as numpy array, or None if not found
        """
        img_base = os.path.splitext(img_filename)[0]
        
        if class_name is not None:
            # Multiclass case
            filename = f"{img_base}_{class_name}.npy"
        else:
            # Single class case
            filename = f"{img_base}.npy"
        
        density_map_dir = self.gt_density_maps_dir if is_gt else self.pred_density_maps_dir
        density_map_path = os.path.join(density_map_dir, filename)
        
        if not os.path.exists(density_map_path):
            return None
        
        return np.load(density_map_path)
    
    def _partition_density_map(self, density_map, divisions):
        """
        Partition a density map into a grid.
        
        Args:
            density_map: 2D numpy array
            divisions: Number of divisions (0=whole, 1=2x2, 2=4x4, etc.)
            
        Returns:
            List of tuples (partition_array, row_idx, col_idx)
        """
        if divisions == 0:
            return [(density_map, 0, 0)]
        
        n_parts = 2 ** divisions
        h, w = density_map.shape
        part_h = h // n_parts
        part_w = w // n_parts
        
        partitions = []
        for i in range(n_parts):
            for j in range(n_parts):
                start_h = i * part_h
                end_h = (i + 1) * part_h if i < n_parts - 1 else h
                start_w = j * part_w
                end_w = (j + 1) * part_w if j < n_parts - 1 else w
                
                partition = density_map[start_h:end_h, start_w:end_w]
                partitions.append((partition, i, j))
        
        return partitions
    
    def _resize_density_map(self, density_map, target_shape):
        """
        Resize a density map to match target shape while preserving count.
        
        Args:
            density_map: Source density map
            target_shape: Target (height, width)
            
        Returns:
            Resized density map with preserved total count
        """
        original_count = density_map.sum()
        
        zoom_factors = (target_shape[0] / density_map.shape[0], 
                       target_shape[1] / density_map.shape[1])
        
        resized = ndimage.zoom(density_map, zoom_factors, order=1)
        
        # Preserve count
        if resized.sum() > 0:
            resized = resized * (original_count / resized.sum())
        
        return resized
    
    def _compute_partition_metrics(self, gt_partition, pred_partition):
        """
        Compute metrics for a single partition.
        
        Args:
            gt_partition: Ground truth partition
            pred_partition: Predicted partition
            
        Returns:
            Dictionary with mae, tp, fp, gt_count
        """
        gt_count = gt_partition.sum()
        pred_count = pred_partition.sum()
        
        mae = abs(pred_count - gt_count)
        tp = min(pred_count, gt_count)
        fp = max(0, pred_count - gt_count)
        
        return {
            'mae': mae,
            'tp': tp,
            'fp': fp,
            'gt_count': gt_count
        }
