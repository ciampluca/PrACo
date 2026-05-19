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
parser.add_argument('--model', type=str, choices=['CounTX', 'CLIP-Count', 'TFPOC', 'VLCounter', 'DAVE', 'ZSC', 'PseCo', 'GroundingREC', 'CountGD', 'CountGDPlusPlus', 'FixedPointPromptCounting', 'all'], 
                    help="Choose the model to use: Options: 'CounTX', 'CLIP-Count', 'TFPOC', 'VLCounter', 'DAVE', 'ZSC', 'PseCo', 'GroundingREC', 'CountGD', 'CountGDPlusPlus', 'FixedPointPromptCounting', 'all'", default='all')
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
    model_names = ["CounTX", "CLIP-Count", "VLCounter", "TFPOC", "DAVE", "ZSC", "PseCo", "GroundingREC", "CountGD", "CountGDPlusPlus", "FixedPointPromptCounting"]
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
    stats_extractor = StatisticsExtractor(model_name, data_dir, test_csv_dir, test_csv_filenames, gt_json_filename, img_class_txt, split_classes_file, img_to_exclude_txt=args.img_to_exclude_txt)
    stats_extractor.load_data()
    
    # Process Test1 data
    stats_extractor.process_test1_data()
    statistics_data_test1 = stats_extractor.evaluate_test1_metrics()
    
    # Process Test2 data
    stats_extractor.process_test2_data()
    statistics_data_test2 = stats_extractor.evaluate_test2_metrics()

    # merge the two dictionaries
    statistics = {**statistics_data_test1, **statistics_data_test2}
    statistics = pd.DataFrame.from_dict(statistics)#.set_index('Model')

    stats.append(statistics)

# Combine statistics for all models
stats = pd.concat(stats, axis=0, ignore_index=True)
print(f"Saving statistics to {output_csv_path}")
stats.to_csv(output_csv_path)
