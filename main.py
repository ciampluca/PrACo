import argparse
import os
import json
from benchmark.benchmark import Benchmark

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run model benchmark tests.")
parser.add_argument('--model', type=str, choices=['CounTX', 'CLIP-Count', 'TFPOC', 'VLCounter', 'DAVE'], required=True, 
                    help="Choose the model to use: Options: 'CounTX', 'CLIP-Count', 'TFPOC', 'VLCounter', 'DAVE'")
parser.add_argument('--data_dir', type=str, default="../CounTX/data/FSC/FSC_147", help="Directory containing the data files.")
parser.add_argument('--img_directory', type=str, default='../CounTX/data/FSC/images_384_VarV2', help="Directory containing the images.")
parser.add_argument('--split_classes_file', type=str, default="Split_Classes_FSC147.json", help="Filename for the split classes JSON.")
parser.add_argument('--split_images_file', type=str, default="Train_Test_Val_FSC_147.json", help="Filename for the split images JSON.")
parser.add_argument('--img_class_txt', type=str, default="ImageClasses_FSC147.txt", help="Filename for the image classes TXT.")
parser.add_argument('--split', type=str, default="test", help="Split to be considered")
args = parser.parse_args()

# Set up directories and file names based on the arguments
data_dir = args.data_dir
img_directory = args.img_directory
split_classes_file = args.split_classes_file
split_images_file = args.split_images_file
img_class_txt = args.img_class_txt

# Load split classes and images
with open(os.path.join(data_dir, split_classes_file), 'r') as f:
    split_classes = json.load(f)

with open(os.path.join(data_dir, split_images_file), 'r') as f:
    split_images = json.load(f)    
    
img_classes = {}

# Load image classes
with open(os.path.join(data_dir, img_class_txt), 'r') as file:
    for line in file:
        line = line.strip().split('\t')
        if len(line) == 2:
            img_name, label = line
            img_classes[img_name] = label

# Select and initialize the model based on the argument
if args.model == 'CounTX':
    from models.countx_model import CounTXModel
    model = CounTXModel(img_directory, split_images, split_classes)
    output_prefix = 'CounTX'
elif args.model == 'CLIP-Count':
    from models.clipcount_model import CLIPCountModel
    model = CLIPCountModel(img_directory, split_images, split_classes)
    output_prefix = 'CLIP-Count'
elif args.model == 'TFPOC':
    from models.TFPOC_model import ClipSAMModel
    model = ClipSAMModel(img_directory, split_images, split_classes)
    output_prefix = 'TFPOC'
elif args.model == 'VLCounter':
    from models.vlcounter_model import VLCounterModel
    model = VLCounterModel(img_directory, split_images, split_classes)
    output_prefix = 'VLCounter'
elif args.model == 'DAVE':
    from models.dave_model import DAVEModel
    model = DAVEModel(img_directory, split_images, split_classes)
    output_prefix = 'DAVE'

# Run benchmarks
img_class_txt_path = os.path.join(data_dir, 'ImageClasses_FSC147.txt')
benchmark = Benchmark(model, img_class_txt_path)

output_csv = f'Inference_Test1_{output_prefix}_{args.split}.csv'
benchmark.run_negative_label_test(output_csv, split=args.split)

output_upper_csv = f'Inference_Test2_Upper_{output_prefix}_{args.split}.csv'
output_lower_csv = f'Inference_Test2_Lower_{output_prefix}_{args.split}.csv'
benchmark.run_mosaic_test(img_classes, output_upper_csv, output_lower_csv, split=args.split)
