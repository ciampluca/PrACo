import os, pandas as pd
import json
import tqdm

annotations_file_path = os.path.join(os.getcwd(), 'annotations_no_scaling.json')

output_images_folder_path = os.path.join(os.getcwd(), '../export-multiclass-dataset/images')

images_folder_path = os.path.join(os.getcwd(), 'images')

with open(annotations_file_path, 'r') as f:
    annotations = json.load(f)

print(f"Total annotations: {len(annotations)}")

# each key in the annotations dict is an image filename

# iterate over the images and copy each of them to the output folder

for image_filename in tqdm.tqdm(annotations.keys()):
    image_path = os.path.join(images_folder_path, image_filename)
    output_image_path = os.path.join(output_images_folder_path, image_filename)

    if not os.path.exists(image_path):
        print(f"Image file {image_path} does not exist. Skipping.")
        raise FileNotFoundError(f"Image file {image_path} does not exist")

    # copy the image to the output folder
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    os.system(f'cp "{image_path}" "{output_image_path}"')
print("All images copied to the output folder.")
