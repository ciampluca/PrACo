from .benchmark import Benchmark
import os
import tqdm
import pandas as pd
from PIL import Image
import numpy as np

PREDICTION_PRECISION = 2


class MulticlassBenchmark(Benchmark):
    def __init__(self, model, img_class_txt_path, benchmark_results_dir="multiclass_benchmark_results", img_class_dict=None, models_options={}):
        super().__init__(model, img_class_txt_path, benchmark_results_dir, img_class_dict)
        self.models_options = models_options

    def run_negative_label_test(self, output_csv, split="test", force=False, save_every=1):
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
                prompt = self.model.get_text_prompt(class_name)
                if 'DAVE' in self.model_name:
                    # In multiclass, img_class contains comma-separated positive classes
                    positive_classes_str = self.img_class[img_filename]
                    if isinstance(positive_classes_str, list):
                        positive_classes = positive_classes_str
                        if len(positive_classes) == 0: raise Exception("no positive class!")
                    else:
                        positive_classes = [c.strip() for c in positive_classes_str.split(',')]
                    positive_prompts = [self.model.get_text_prompt(c) for c in positive_classes]
                    pred_cnt, density_map = self.model.infer(img, text=prompt, text_positive=positive_prompts, per_cluster_thresh=self.models_options.get("per_cluster_thresh", False), take_min_score=self.models_options.get("take_min_score", False))
                else:
                    pred_cnt, density_map = self.model.infer(img, prompt)
                df.at[img_filename, class_name] = round(pred_cnt, PREDICTION_PRECISION)
                
                assert len(density_map.shape) == 2, f"Density map for {img_filename}, class {class_name} is not 2D. It is {density_map.shape}"
                #assert density_map.shape == img.size[::-1], f"Density map shape {density_map.shape} does not match image shape {img.size[::-1]}"
                # Save density map
                density_map_dir = os.path.join(self.benchmark_results_dir, self.model_name, "density_maps")
                os.makedirs(density_map_dir, exist_ok=True)
                density_map_path = os.path.join(density_map_dir, f"{img_filename.split('.')[0]}_{class_name}.npy")
                np.save(density_map_path, density_map)

            if idx % save_every == 0:
                df.to_csv(output_file)

        df.to_csv(output_file)
        return df
    
    def run_mosaic_test(self, img_classes, output_upper_csv, output_lower_csv, split="test", force=False):
        
        raise NotImplementedError("Mosaic test not needed in MulticlassBenchmark.")