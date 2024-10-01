from abc import ABC, abstractmethod
import numpy as np
import cv2
import torch
from .base_model import BaseModel
import sys, os
from torchvision import transforms
from PIL import Image
from .TFPOC import clip
from .TFPOC.shi_segment_anything import sam_model_registry
from .TFPOC.shi_segment_anything.automatic_mask_generator_text import SamAutomaticMaskGenerator
from .TFPOC.utils import *

class ClipSAMModel(BaseModel):
    def __init__(self, img_directory, split_images, split_classes, device='cuda:0'):
        super().__init__(img_directory, split_images, split_classes)
        self.model_name = "TFPOC"
        self.device = device
        # Load CLIP model
        self.clip_model, _ = clip.load("CS-ViT-B/16", device=device)
        self.clip_model.eval()
        # Load SAM model
        sam_checkpoint = "pretrained_models/sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(model=self.sam)

    def get_text_prompt(self, text):
        """
        Implement the specific prompt retrieval logic for the CLIP-Count model.
        """
        return f"{text}"

    def infer(self, img, text):
        """
        Perform inference on a single image with a given text prompt.

        Args:
            img: The input image for inference.
            text: The text prompt for CLIP model.

        Returns:
            Tuple containing:
                - pred_cnt: Number of predicted masks
                - density_map_tensor: The density mask tensor
        """
        if isinstance(img, Image.Image):
            img = np.array(img)  # Convert PIL Image to NumPy array
            
        # Generate masks on the original image
        input_prompt = get_clip_bboxs(self.clip_model, img, text, self.device)
        masks = self.mask_generator.generate(img, input_prompt)
        
        # Initialize the density map with zeros (shape: target size 384x384)
        target_height, target_width = 384, 384
        density_map = np.zeros((target_height, target_width), dtype=np.float32)
        
        # Determine the scaling factors
        original_height, original_width = img.shape[:2]
        x_scale = original_width / target_width
        y_scale = original_height / target_height
        
        # Loop through each mask and add a small circle at its centroid to the density map
        for mask in masks:
            segmentation = mask['segmentation']
            
            # Calculate the centroid of the mask
            M = cv2.moments(segmentation.astype(np.uint8))
            if M["m00"] != 0:  # To avoid division by zero
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
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