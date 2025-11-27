import argparse
import os
import json
from benchmark.multiclass_benchmark import MulticlassBenchmark

from models.loader import MULTICLASS_IMPLEMENTED_MODELS, load_model

parser = argparse.ArgumentParser(description="Run multiclass benchmark tests.")
parser.add_argument('--model', type=str, required=True, help="Model name", choices=MULTICLASS_IMPLEMENTED_MODELS)
parser.add_argument('--data_dir', type=str, default="data/multiclass-dataset", help="Directory containing the multiclass dataset files.")
parser.add_argument('--img_directory', type=str, default='data/multiclass-dataset/images', help="Directory containing the images.")
parser.add_argument('--split_classes_file', type=str, default="multiclass_split_classes.json", help="Filename for the split classes JSON. It contains a mapping from split names to classes str lists.")
parser.add_argument('--split_images_file', type=str, default="multiclass_split_images.json", help="Filename for the split images JSON. It contains a mapping from split names to image filename str lists.")
parser.add_argument('--img_class_txt', type=str, default="multiclass_image_classes.txt", help="Filename for the image classes TXT. It contains image filename and comma-separated classes per line.")
parser.add_argument('--device', type=str, default="cuda", help="Device to run the model on, e.g., 'cuda' or 'cpu'.")

parser.add_argument('--split', type=str, default="test", help="Split to be considered", choices=['test'])

args = parser.parse_args()


# Set up directories and file names based on the arguments
data_dir = args.data_dir
img_directory = args.img_directory
split_classes_path = os.path.join(data_dir, args.split_classes_file)
split_images_path = os.path.join(data_dir, args.split_images_file)
img_class_txt_path = os.path.join(data_dir, args.img_class_txt)

# Load split classes and images
with open(split_classes_path, 'r') as f:
    split_classes = json.load(f)

with open(split_images_path, 'r') as f:
    split_images = json.load(f)


# img_classes dict will have image filename as key and list of classes as value
img_classes = {}

# Load image classes
with open(img_class_txt_path, 'r') as file:
    for line in file:
        line = line.strip().split('\t')
        if len(line) == 2:
            img_name, classes_str = line
            classes_list = [cls.strip() for cls in classes_str.split(',')]
            img_classes[img_name] = classes_list


# Select and initialize the model based on the argument
print(f"Loading model: {args.model} on device: {args.device}")
model = load_model(
    model_name=args.model,
    img_directory=img_directory,
    split_images=split_images,
    split_classes=split_classes,
    load_filtered_checkpoints=True,
    device=args.device
)
output_prefix = args.model

print(f"Initialized model: {args.model}")

benchmark = MulticlassBenchmark(model, img_class_txt_path, img_class_dict=img_classes)

print(f"Loaded MulticlassBenchmark for the dataset: {data_dir}")
print()
print(f"Running negative label test on split: {args.split}")

benchmark.run_negative_label_test(
    output_csv=f"multiclass_{output_prefix}_{args.split}1.csv",
    split=args.split
)

print("Completed negative label test.")