import numpy as np
import torch
from .base_model import BaseModel
import torchvision.ops as vision_ops
import cv2

import sys, os

#groundingdino_path = os.path.join(os.path.dirname(__file__), 'GroundingDINO')
#
#if groundingdino_path not in sys.path:
#    sys.path.append(groundingdino_path)

from groundingdino.util.base_api import load_model, threshold
from groundingdino.datasets import transforms as T


class GroundingRECModel(BaseModel):
    def __init__(self, img_directory, split_images, split_classes, model_ckpt='pretrained_models/GroundingREC_model_best.pth', device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(img_directory, split_images, split_classes)
        self.model_name = "GroundingREC"
        self.CONFIG_PATH = "models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        
        self.device = torch.device(device)
        
        # Load model
        self.model = load_model(self.CONFIG_PATH, model_ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.img_trans = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def get_text_prompt(self, text):
        """
        Implement the specific prompt retrieval logic for the GroundingREC model.
        """
        return [f"{text}."]

    def infer(self, img, text, text_threshold=0.5):
        """
        Perform inference on a single image with a given text prompt.

        Args:
            img: The input image for inference.
            text: The text prompt for GroundingREC model.

        Returns:
            Tuple containing:
                - pred_cnt: Number of predicted masks
                - density_map_tensor: The density mask tensor
        """
        img = img.convert("RGB")
        # h, w, _ = np.asarray(img).shape
        img, _ = self.img_trans(img, None)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        
        # Initialize the density map with zeros (shape: target size 384x384)
        target_height, target_width = 384, 384
        density_map = np.zeros((target_height, target_width), dtype=np.float32)
        
        # Determine the scaling factors
        original_height, original_width = img.shape[2], img.shape[3]
        x_scale = original_width / target_width
        y_scale = original_height / target_height

        with torch.no_grad():
            outputs = self.model(img, captions=text)
            outputs["pred_points"] = outputs["pred_boxes"][:, :, :2]
            results = threshold(outputs, text, self.model.tokenizer, text_threshold=text_threshold)
    
            for b in range(len(results)):
                boxes, logits, phrases = results[b]
                boxes = [box.tolist() for box in boxes]
                logits = logits.tolist()
                points = [[box[0]*original_width, box[1]*original_height] for box in boxes]
                
                # Loop through each bb and add a small circle at its centroid to the density map
            for point in points:
                cX = int(point[0])
                cY = int(point[1])
                
                # Scale the centroid to the target density map size
                cX_resized = int(cX / x_scale)
                cY_resized = int(cY / y_scale)
                
                # Place a value of 1 at the scaled centroid
                if 0 <= cX_resized < target_width and 0 <= cY_resized < target_height:
                    # density_map[cY_resized, cX_resized] += 1
                    overlay = np.zeros_like(density_map)
                    cv2.rectangle(overlay, (cX_resized-2, cY_resized-2), (cX_resized+2, cY_resized+2), 1 / 25, -1)
                    density_map += overlay

        # Convert the density_map to a torch tensor
        density_map_tensor = torch.from_numpy(density_map)

        # Integrate over the density_map tensor
        pred_cnt = torch.sum(density_map_tensor).item()
        
        return pred_cnt, density_map_tensor