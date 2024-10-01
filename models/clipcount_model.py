from abc import ABC, abstractmethod
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from .base_model import BaseModel
from PIL import Image
import sys, os
from .CLIPCount.util import misc
from .CLIPCount import run

SCALE_FACTOR = 60

class CLIPCountModel(BaseModel):
    
    def __init__(self, img_directory, split_images, split_classes, model_ckpt='pretrained_models/clipcount_pretrained.ckpt'):
        super().__init__(img_directory, split_images, split_classes)
        self.model = run.Model.load_from_checkpoint(model_ckpt, strict=False)
        self.model.eval()
        self.model_name = "CLIP-Count"
        
    def get_text_prompt(self, text):
        """
        Implement the specific prompt retrieval logic for the CLIP-Count model.
        """
        return f"{text}"
        
    def infer(self, img, text):
        """
        Implement the specific inference logic for the CLIP-Count model.
        """
        self.model.eval()
        self.model.model = self.model.model.cuda()
        
        if isinstance(img, Image.Image):
            img = np.array(img)  # Convert PIL Image to NumPy array
        
        with torch.no_grad():
            # Reshape height to 384, keep aspect ratio
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
            resized_image = TF.resize(img_tensor, (384))
            
            resized_image = resized_image.float() / 255.0
            resized_image = torch.clamp(resized_image, 0, 1)
            text = [text]
            
            with torch.cuda.amp.autocast():
                raw_h, raw_w = resized_image.shape[2:]
                patches, _ = misc.sliding_window(resized_image, stride=128)
                
                # Convert to batch
                patches = torch.from_numpy(patches).float().to(resized_image.device)
                text = np.repeat(text, patches.shape[0], axis=0)
                
                density_map_tensor = self.model.forward(patches, text)
                density_map_tensor.unsqueeze_(1)
                density_map_tensor = misc.window_composite(density_map_tensor, stride=128)
                density_map_tensor = density_map_tensor.squeeze(1)
                
                # Crop to original width
                density_map_tensor = density_map_tensor[:, :, :raw_w]
            
            density_map_tensor = density_map_tensor[0] / SCALE_FACTOR
            pred_cnt = torch.sum(density_map_tensor).item()
        
        return pred_cnt, density_map_tensor