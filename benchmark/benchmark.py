import os
import random
import tqdm
import pandas as pd
from PIL import Image
import torch
import json

PREDICTION_PRECISION = 2

class Benchmark:
    def __init__(self, model, img_class_txt_path):
        self.model = model
        
        self.model_name = model.model_name
        self.benchmark_results_dir = "benchmark_results"
        
        if not os.path.exists(self.benchmark_results_dir):
            os.makedirs(self.benchmark_results_dir)
        
        if not os.path.exists(os.path.join(self.benchmark_results_dir, self.model_name)):
            os.makedirs(os.path.join(self.benchmark_results_dir, self.model_name))

        # Load Image Class .txt
        self.img_class = {}
        with open(img_class_txt_path, 'r') as file:
            for line in file:
                line = line.strip().split('\t')
                if len(line) == 2:
                    img_name, label = line
                    # img_id = img_name.split(".")[0]
                    self.img_class[img_name] = label

    def run_negative_label_test(self, output_csv, split="test", force=False):
        output_file = os.path.join(self.benchmark_results_dir, self.model_name, output_csv)
        if os.path.exists(output_file) and not force:
            df = pd.read_csv(output_file, index_col=0)
        else:
            df = pd.DataFrame(columns=self.model.split_classes[split], index=self.model.split_images[split])

        for idx, img_filename in enumerate(tqdm.tqdm(self.model.split_images[split])):
            if not df.loc[img_filename].isnull().values.all():
                print(f"Skipping {img_filename} as all predictions are already made")
                continue
            img_path = os.path.join(self.model.img_directory, img_filename)
            img = Image.open(img_path)
            img.load()

            for class_name in self.model.split_classes[split]:
                positive_class = self.img_class[img_filename]
                positive_prompt = self.model.get_text_prompt(positive_class)
                prompt = self.model.get_text_prompt(class_name)
                if 'DAVE' in self.model_name:
                    pred_cnt, _ = self.model.infer(img, text=prompt, text_positive=positive_prompt)
                else:
                    pred_cnt, _ = self.model.infer(img, prompt)
                df.at[img_filename, class_name] = round(pred_cnt, PREDICTION_PRECISION)

            if idx % 10 == 0:
                df.to_csv(output_file)

        df.to_csv(output_file)

    def run_mosaic_test(self, img_classes, output_upper_csv, output_lower_csv, split="test", force=False):
        def create_collage(image1, image2, type):
            
            # Resize images to be the same size
            width, height = min(image1.size[0], image2.size[0]), min(image1.size[1], image2.size[1])
            if type == "vertical":
                width = height
                height = height // 2
                image1 = image1.resize((width, height))
                image2 = image2.resize((width, height))
                collage = Image.new('RGB', (width, height * 2))
                collage.paste(image1, (0, 0))
                collage.paste(image2, (0, height))
                
            elif type == "horizontal":
                image1 = image1.resize((width, height))
                image2 = image2.resize((width, height))
                collage = Image.new('RGB', (width * 2, height))
                collage.paste(image1, (0, 0))
                collage.paste(image2, (width, 0))
                
            return collage

        def extract_random_img(class_name):
            filtered_img_classes = [key for key, value in img_classes.items() if value == class_name]
            img_filename = random.choice(filtered_img_classes)
            return img_filename

        random.seed(123)
        output_upper_file = os.path.join(self.benchmark_results_dir, self.model_name, output_upper_csv)
        output_lower_file = os.path.join(self.benchmark_results_dir, self.model_name, output_lower_csv)
        if os.path.exists(output_upper_file) and os.path.exists(output_lower_file) and not force:
            df_upper = pd.read_csv(output_upper_file, index_col=0)
            df_lower = pd.read_csv(output_lower_file, index_col=0)
        else:
            df_upper = pd.DataFrame(columns=self.model.split_classes[split], index=self.model.split_images[split])
            df_lower = pd.DataFrame(columns=self.model.split_classes[split], index=self.model.split_images[split])
        random_images_list = []

        for idx, img_filename in enumerate(tqdm.tqdm(self.model.split_images[split])):
            img = Image.open(os.path.join(self.model.img_directory, img_filename))
            img.load()

            if not df_upper.loc[img_filename].isnull().values.all() and not df_lower.loc[img_filename].isnull().values.all():
                print(f"Skipping {img_filename} as all predictions are already made")
                continue

            for class_name in self.model.split_classes[split]:
                if class_name == img_classes[img_filename]:
                    continue
                
                img2_filename = extract_random_img(class_name)
                random_images_list.append(img2_filename)
                
                img2 = Image.open(os.path.join(self.model.img_directory, img2_filename))
                img2.load()
                collage = create_collage(img, img2, type="vertical")

                prompt = self.model.get_text_prompt(img_classes[img_filename])
                _, density_map_tensor = self.model.infer(collage, prompt)

                half_height = density_map_tensor.size(dim=0)//2
                upper_density = density_map_tensor[:half_height, :]
                lower_density = density_map_tensor[half_height:, :]

                pred_cnt_up = torch.sum(upper_density).item()
                pred_cnt_low = torch.sum(lower_density).item()

                df_upper.at[img_filename, class_name] = round(pred_cnt_up, PREDICTION_PRECISION)
                df_lower.at[img_filename, class_name] = round(pred_cnt_low, PREDICTION_PRECISION)

            if idx % 10 == 0:
                df_upper.to_csv(output_upper_file)
                df_lower.to_csv(output_lower_file)
                
        df_upper.to_csv(output_upper_file)
        df_lower.to_csv(output_lower_file)
