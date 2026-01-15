from tqdm import tqdm
import argparse
import pandas as pd
import os

from models.loader import MULTICLASS_IMPLEMENTED_MODELS
from benchmark.multiclass_statistics_extractor import MulticlassStatisticsExtractor

parser = argparse.ArgumentParser(description="Run model multiclass benchmark tests.")
parser.add_argument('--data_dir', type=str, default="data/multiclass-dataset", help="Directory containing the multiclass dataset files.")
parser.add_argument('--split', type=str, default="test", help="Split to be considered", choices=['test'])
parser.add_argument('--model', type=str, default='all', help="Choose the model to use or 'all' to run all models.")
parser.add_argument('--benchmark_inference_dir', type=str, default="multiclass_benchmark_results", help="Directory containing benchmark results.")
parser.add_argument('--compute_localized', default=True, action='store_true', help="Compute localized metrics based on density map partitions.")

args = parser.parse_args()

data_dir = args.data_dir
output_csv_path = f'./multiclass_final_metrics_{args.split}.csv'
gt_json_count_per_class_filename = "multiclass_class_counts_per_image.json"
img_class_txt = "multiclass_image_classes.txt"
split_classes_filename = "multiclass_split_classes.json"

#gt_json_count_per_class_path = os.path.join(data_dir, gt_json_count_per_class_filename)
#split_classes_path = os.path.join(data_dir, split_classes_filename)

# List of model names to evaluate
if args.model == 'all':
    model_names = MULTICLASS_IMPLEMENTED_MODELS
elif "[" in args.model and "]" in args.model:
    # Parse list of models from string
    print("Parsing multiple models from input string.")
    model_names = [m.strip().replace("'", "").replace('"', '') for m in args.model.strip("[]").split(",")]
    print(f"Models to evaluate: {model_names}")
else:
    model_names = [args.model]

# Loop through the model names
stats = []

if os.path.exists(output_csv_path):
    prev_stats = pd.read_csv(output_csv_path, index_col=0)#, index_col='Model'
    stats.append(prev_stats)

for model_name in tqdm(model_names, desc="Evaluating Models on Multiclass Benchmark"):
    test_csv_filenames = {
        "test1" : f'multiclass_{model_name}_test1.csv',
    }

    test_csv_dir = os.path.join(args.benchmark_inference_dir, model_name)

    if not os.path.exists(test_csv_dir):
        print(f"Skipping model {model_name} as no benchmark results found in {test_csv_dir}.")
        continue

    stats_extractor = MulticlassStatisticsExtractor(model_name, data_dir, test_csv_dir, test_csv_filenames, gt_json_count_per_class_filename, img_class_txt, split_classes_filename, img_to_exclude_txt=None)

    stats_extractor.load_data()

    # Process Test1 data
    stats_extractor.process_test1_data()
    statistics_data_test1 = stats_extractor.evaluate_test1_metrics()

    # Compute localized metrics if requested
    if args.compute_localized:
        for divisions in [1, 2]:  # 2x2 and 4x4 partitions
            localized_df = stats_extractor.compute_localized_metrics(
                divisions=divisions, 
                positive_classes_only=True
            )
            
            # Filter only rows with density maps available
            localized_df_valid = localized_df[localized_df['Has Density Maps'] == True]
            
            if len(localized_df_valid) > 0:
                # Compute macro averages (per-image average first, then global average)
                # Group by image to get per-image averages across all positive classes
                per_image_avg = localized_df_valid.groupby('Image Name')[['Total MAE', 'Recall', 'Precision', 'F-score']].mean()
                mae_macro = per_image_avg['Total MAE'].mean()
                recall_macro = per_image_avg['Recall'].mean()
                precision_macro = per_image_avg['Precision'].mean()
                fscore_macro = per_image_avg['F-score'].mean()
                
                # Compute micro averages (directly average all (image, class) values)
                mae_micro = localized_df_valid['Total MAE'].mean()
                recall_micro = localized_df_valid['Recall'].mean()
                precision_micro = localized_df_valid['Precision'].mean()
                fscore_micro = localized_df_valid['F-score'].mean()
                
                # Add to statistics
                n_parts = 2 ** ( divisions * 2 ) 
                statistics_data_test1[f'Localized_MAE_macro_div{n_parts}'] = [round(mae_macro, stats_extractor.metric_precision)]
                statistics_data_test1[f'Localized_MAE_micro_div{n_parts}'] = [round(mae_micro, stats_extractor.metric_precision)]
                statistics_data_test1[f'Localized_Recall_macro_div{n_parts}'] = [round(recall_macro, stats_extractor.metric_precision)]
                statistics_data_test1[f'Localized_Recall_micro_div{n_parts}'] = [round(recall_micro, stats_extractor.metric_precision)]
                statistics_data_test1[f'Localized_Precision_macro_div{n_parts}'] = [round(precision_macro, stats_extractor.metric_precision)]
                statistics_data_test1[f'Localized_Precision_micro_div{n_parts}'] = [round(precision_micro, stats_extractor.metric_precision)]
                statistics_data_test1[f'Localized_Fscore_macro_div{n_parts}'] = [round(fscore_macro, stats_extractor.metric_precision)]
                statistics_data_test1[f'Localized_Fscore_micro_div{n_parts}'] = [round(fscore_micro, stats_extractor.metric_precision)]
            else:
                print(f"Warning: No valid density maps found for {model_name} at division level {divisions}")
    
    # merge statistics in one dict
    combined_stats = {**statistics_data_test1}
    model_stats_df = pd.DataFrame.from_dict(combined_stats)

    stats.append(model_stats_df)

# Combine all statistics and save to CSV
final_stats = pd.concat(stats, axis=0, ignore_index=True)
print(f"Saving final statistics to {output_csv_path}")
final_stats.to_csv(output_csv_path)
print("Done.")