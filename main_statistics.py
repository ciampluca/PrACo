from benchmark.statistics_extractor import StatisticsExtractor
import os
from tqdm import tqdm
import argparse
import pandas as pd


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run model benchmark tests.")
parser.add_argument('--data_dir', type=str, default="./data", help="Directory containing the data files.")
parser.add_argument('--img_to_exclude_txt', type=str, default=None, help="Path of the txt file containing images to exclude")
parser.add_argument('--split', type=str, default="test", help="Split to be considered")
parser.add_argument('--compute_localized', default=True, action='store_true', help="Compute localized metrics using density map partitions")
parser.add_argument('--gt_density_maps_dir', type=str, default="./data/gt_density_maps_FSC", help="Directory containing ground truth density maps (optional)")
parser.add_argument('--collage_type', type=str, default="vertical", choices=['vertical', 'horizontal'], help="Type of mosaic collage for test2: 'vertical' (default) or 'horizontal'")
parser.add_argument('--model', type=str, choices=['CounTX', 'CLIP-Count', 'TFPOC', 'VLCounter', 'DAVE', 'ZSC', 'PseCo', 'GroundingREC', 'CountGD', 'FixedPointPromptCounting', 'all'], 
                    help="Choose the model to use: Options: 'CounTX', 'CLIP-Count', 'TFPOC', 'VLCounter', 'DAVE', 'ZSC', 'PseCo', 'GroundingREC', 'CountGD', 'FixedPointPromptCounting', 'all'", default='all')
args = parser.parse_args()

# Set up directories and file names based on the arguments
data_dir = args.data_dir

output_csv_path = f'./final_metrics_{args.split}.csv'

#data_dir = "../CounTX/data/FSC/FSC_147"
gt_json_filename = "annotation_FSC147_384.json"
img_class_txt = "ImageClasses_FSC147.txt"
split_classes_file = "Split_Classes_FSC147.json"

# List of model names to evaluate
if args.model == 'all':
    model_names = ["CounTX", "CLIP-Count", "VLCounter", "TFPOC", "DAVE", "ZSC", "PseCo", "GroundingREC", "CountGD", "FixedPointPromptCounting"]
else:
    model_names = [args.model]

# Loop through the model names
stats = []

if os.path.exists(output_csv_path):
    prev_stats = pd.read_csv(output_csv_path, index_col=0)#, index_col='Model'
    stats.append(prev_stats)

for model_name in tqdm(model_names, desc="Evaluating Models"):
    m = "DAVE" if "DAVE" in model_name else model_name
    test_csv_filenames = {
        'test1': f'Inference_Test1_{m}_{args.split}.csv',
        'upper_test2': f'Inference_Test2_Upper_{m}_{args.split}.csv',
        'lower_test2': f'Inference_Test2_Lower_{m}_{args.split}.csv'
    }
    test_csv_dir = os.path.join('benchmark_results', model_name)
    
    # Initialize the StatisticsExtractor for the current model
    stats_extractor = StatisticsExtractor(
        model_name, data_dir, test_csv_dir, test_csv_filenames, 
        gt_json_filename, img_class_txt, split_classes_file, 
        img_to_exclude_txt=args.img_to_exclude_txt,
        gt_density_maps_dir=args.gt_density_maps_dir
    )
    stats_extractor.load_data()
    
    # Process Test1 data
    stats_extractor.process_test1_data()
    statistics_data_test1 = stats_extractor.evaluate_test1_metrics()
    
    # Process Test2 data
    stats_extractor.process_test2_data()
    statistics_data_test2 = stats_extractor.evaluate_test2_metrics()

    # Compute localized metrics if requested
    if args.compute_localized:
        print(f"Computing localized metrics for {model_name}...")
        
        for divisions in [1, 2]:  # 2x2 and 4x4 grids
            n_partitions = (2 ** divisions) ** 2  # 4 or 16 partitions
            
            # Compute localized metrics for test2
            localized_df = stats_extractor.compute_test2_localized_metrics(
                divisions=divisions, 
                collage_type=args.collage_type
            )
            
            # Filter to only images with density maps
            localized_df_valid = localized_df[localized_df['Has Density Maps'] == True].copy()
            
            if len(localized_df_valid) == 0:
                print(f"Warning: No valid density maps found for {model_name} with {n_partitions} partitions")
                continue
            
            # Compute macro-averaged metrics (per-image average first, then global average)
            # This matches the standard test2 computation: mean per negative class, then mean per image
            metrics = ['Total MAE', 'Recall', 'Precision', 'F-score']
            per_image_avg = localized_df_valid.groupby('Image Name')[metrics].mean()
            macro_metrics = per_image_avg.mean()
            
            # Compute micro-averaged metrics
            # Two interpretations:
            # 1. TRUE MICRO: Recompute from pooled TP/FP/GT (most statistically correct)
            # 2. DIRECT AVERAGE: Simple mean of all (image, neg_class) values (simpler)
            # 
            # For coherence with standard test2 which uses mean(mean()), we offer both:
            
            # Option 1: Direct average across all (image, negative_class) pairs
            # This is simpler but weights each pair equally regardless of GT count
            micro_metrics_direct = localized_df_valid[metrics].mean()
            
            # Option 2: True micro-averaging (recompute from pooled values)
            # This weights by GT count for Recall and (TP+FP) for Precision
            total_tp = localized_df_valid['Total TP'].sum()
            total_fp = localized_df_valid['Total FP'].sum()
            total_gt = localized_df_valid['GT Count'].sum()
            
            micro_recall_pooled = total_tp / total_gt if total_gt > 0 else 0
            micro_precision_pooled = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            
            if micro_precision_pooled > 0 and micro_recall_pooled > 0:
                micro_fscore_pooled = 2 * (micro_precision_pooled * micro_recall_pooled) / (micro_precision_pooled + micro_recall_pooled)
            else:
                micro_fscore_pooled = 0
            
            # Use direct average for consistency with standard test2 approach
            # (which also uses mean of means, not pooled recomputation)
            micro_metrics = micro_metrics_direct
            
            # Add to statistics_data_test2
            for metric in metrics:
                metric_name = metric.replace(' ', '_')  # e.g., 'Total MAE' -> 'Total_MAE'
                statistics_data_test2[f'Localized_{metric_name}_macro_div{n_partitions}'] = [round(macro_metrics[metric], stats_extractor.metric_precision)]
                statistics_data_test2[f'Localized_{metric_name}_micro_div{n_partitions}'] = [round(micro_metrics[metric], stats_extractor.metric_precision)]

    # merge the two dictionaries
    statistics = {**statistics_data_test1, **statistics_data_test2}
    statistics = pd.DataFrame.from_dict(statistics)#.set_index('Model')

    stats.append(statistics)

# Combine statistics for all models
stats = pd.concat(stats, axis=0, ignore_index=True)
print(f"Saving statistics to {output_csv_path}")
stats.to_csv(output_csv_path)