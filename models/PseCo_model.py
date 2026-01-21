import numpy as np
import torch
from .base_model import BaseModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.ops as vision_ops
import cv2

from .PseCo.ops.foundation_models.segment_anything import build_sam_vit_h
from .PseCo.models import PointDecoder, ROIHeadMLP as ROIHead


class PseCoModel(BaseModel):
    def __init__(self, img_directory, split_images, split_classes, point_decoder_ckpt='pretrained_models/point_decoder_vith.pth', 
                 cls_head_ckpt='pretrained_models/MLP_small_box_w1_zeroshot.tar', clip_text_prompt_ckpt='pretrained_models/clip_text_prompt.pth', device="cuda"):
        super().__init__(img_directory, split_images, split_classes)
        self.model_name = "PseCo"
        self.device = torch.device(device)

        # Load models
        self.sam = build_sam_vit_h()
        self.sam = self.sam.to(self.device)
        self.sam.eval()

        self.point_decoder = PointDecoder(self.sam)
        self.point_decoder = self.point_decoder.to(self.device)
        self.point_decoder.eval()
        state_dict = torch.load(point_decoder_ckpt, map_location='cpu')
        if 'point_decoder' in state_dict:
            print("Loading point decoder state dict from 'point_decoder' key.")
            state_dict = state_dict['point_decoder']
        self.point_decoder.load_state_dict(state_dict)
        self.point_decoder.max_points = 1000
        self.point_decoder.point_threshold = 0.05
        self.point_decoder.nms_kernel_size = 3

        self.cls_head = ROIHead().cuda().eval()
        self.cls_head = self.cls_head.to(self.device)
        self.cls_head.eval()
        cls_head_state_dict = torch.load(cls_head_ckpt, map_location='cpu')
        if 'cls_head' in cls_head_state_dict:
            print("Loading classification head state dict from 'cls_head' key.")
            cls_head_state_dict = cls_head_state_dict['cls_head']
        self.cls_head.load_state_dict(cls_head_state_dict)
        
        self.clip_text_prompts = torch.load(clip_text_prompt_ckpt, map_location='cpu')

        self.img_trans = A.Compose([
            A.LongestMaxSize(1024),
            A.PadIfNeeded(1024, border_mode=0, position="top_left", value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def get_text_prompt(self, text):
        """
        Implement the specific prompt retrieval logic for the PseCo model.
        """
        return f"{text}"

    def infer(self, img, text, threshold=0.34):
        """
        Perform inference on a single image with a given text prompt.

        Args:
            img: The input image for inference.
            text: The text prompt for PseCo model.

        Returns:
            Tuple containing:
                - pred_cnt: Number of predicted masks
                - density_map_tensor: The density mask tensor
        """
        img = np.array(img)
        img = self.img_trans(image=img)['image'].unsqueeze(0)
        img = img.to(self.device)
        
        # Initialize the density map with zeros (shape: target size 384x384)
        target_height, target_width = 384, 384
        density_map = np.zeros((target_height, target_width), dtype=np.float32)
        
        # Determine the scaling factors
        original_height, original_width = img.shape[2], img.shape[3]
        x_scale = original_width / target_width
        y_scale = original_height / target_height
        
        example_features = self.clip_text_prompts[text].unsqueeze(0)
        # example_features = dump_clip_text_features([text, ])[text].unsqueeze(0)
        example_features = example_features.to(self.device)

        with torch.no_grad():
            features = self.sam.image_encoder(img)
            outputs_heatmaps = self.point_decoder(features)
            pred_heatmaps = outputs_heatmaps['pred_heatmaps'].cpu().squeeze().clamp(0, 1)
            pred_points = outputs_heatmaps['pred_points'].squeeze().reshape(-1, 2)
            pred_points_score = outputs_heatmaps['pred_points_score'].squeeze()

            all_pred_boxes = []
            all_pred_ious = []
            cls_outs = []
            for indices in torch.arange(len(pred_points)).split(128):
                outputs_points = self.sam.forward_sam_with_embeddings(features, points=pred_points[indices])
                pred_boxes = outputs_points['pred_boxes']
                pred_logits = outputs_points['pred_ious']

                for anchor_size in [8, ]:
                    anchor = torch.Tensor([[-anchor_size, -anchor_size, anchor_size, anchor_size]]).cuda()
                    anchor_boxes = pred_points[indices].repeat(1, 2) + anchor
                    anchor_boxes = anchor_boxes.clamp(0., 1024.)
                    outputs_boxes = self.sam.forward_sam_with_embeddings(features, points=pred_points[indices], boxes=anchor_boxes)
                    pred_logits = torch.cat([pred_logits, outputs_boxes['pred_ious'][:, 1].unsqueeze(1)], dim=1)
                    pred_boxes = torch.cat([pred_boxes, outputs_boxes['pred_boxes'][:, 1].unsqueeze(1)], dim=1)

                all_pred_boxes.append(pred_boxes)
                all_pred_ious.append(pred_logits)
                cls_outs_ = self.cls_head(features, [pred_boxes, ], [example_features, ] * len(indices))
                cls_outs_ = cls_outs_.sigmoid().view(-1, len(example_features), 5).mean(1)
                pred_logits = cls_outs_ * pred_logits
                cls_outs.append(pred_logits)

            pred_boxes = torch.cat(all_pred_boxes)
            pred_ious = torch.cat(all_pred_ious)
            cls_outs = torch.cat(cls_outs)
            pred_boxes = pred_boxes[torch.arange(len(pred_boxes)), torch.argmax(cls_outs, dim=1)]
            scores = cls_outs.max(1).values
            indices = vision_ops.nms(pred_boxes, scores, 0.5)
            pred_boxes = pred_boxes[indices]
            scores = scores[indices]
            
            final_boxes = []
            for p, (xmin, ymin, xmax, ymax) in zip(scores, pred_boxes.tolist()):
                if p < threshold:
                    continue
                final_boxes.append([xmin, ymin, xmax, ymax])
                
            # Loop through each bb and add a small circle at its centroid to the density map
            for bb in final_boxes:
                # Calculate centroid as the midpoint of the bounding box
                cX = int((bb[0] + bb[2]) / 2)
                cY = int((bb[1] + bb[3]) / 2)
                
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