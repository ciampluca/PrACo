from models.loader import load_model
import json

from PIL import Image


device = "cuda:0"

img_class_txt = "data/ImageClasses_FSC147.txt"
split_classes_file = "data/Split_Classes_FSC147.json"
split_images_file = "data/Train_Test_Val_FSC_147.json"
split = "test"
img_directory = f"data/images_384_VarV2"
gt_json = "data/gt_cnts_FSC147.json"

img_classes = {}
# Load image classes
with open(img_class_txt, 'r') as file:
    for line in file:
        line = line.strip().split('\t')
        if len(line) == 2:
            img_name, label = line
            img_classes[img_name] = label

# Load split classes and images
with open(split_classes_file, 'r') as f:
    split_classes = json.load(f)

with open(split_images_file, 'r') as f:
    split_images = json.load(f)

# Load GT Counts JSON
with open(gt_json, 'r') as file:
    gt_counts_dict = json.load(file)


from table_utils import get_ordered_models_list

models = get_ordered_models_list()

sample_img = split_images[split][0]
print(f"Sample image: {sample_img}")
print(f"GT count for sample image: {gt_counts_dict[sample_img]}")

for model in models:
    print(f"Loading model: {model}")
    loaded_model = load_model(model, img_directory=img_directory, split_images=split_images[split], split_classes=split_classes[split], device=device)
    print(f"Model {model} loaded successfully.\n")

    label = img_classes[sample_img]
    positive_class = label
    positive_prompt = positive_class
    img_path = f"{img_directory}/{sample_img}"
    img = Image.open(img_path)
    img.load()
    positive_prompt = loaded_model.get_text_prompt(positive_prompt)
    if 'DAVE' in model:
        pred_cnt, den_map = loaded_model.infer(img, text=positive_prompt, text_positive=positive_prompt)
    else:
        pred_cnt, den_map = loaded_model.infer(img, text=positive_prompt)

    print(f"Model {model} - Predicted count: {pred_cnt}, GT count: {gt_counts_dict[sample_img]}\n")
