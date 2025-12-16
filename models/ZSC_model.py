import copy
import numpy as np
import torch
from .base_model import BaseModel
from torchvision import transforms
import torch.nn.functional as F

from .ZSC.utils import *
from .ZSC.models.regressor import get_regressor
from .ZSC.FSC147_dataset import random_aug_boxes, get_image_classes
from .ZSC.config import cfg


class ZSCModel(BaseModel):
    def __init__(self, img_directory, split_images, split_classes, model_ckpt='pretrained_models/zsc_model_best.pth', config="models/ZSC/config/test.yaml", device="cuda"):
        super().__init__(img_directory, split_images, split_classes)
        self.model_name = "ZSC"
        self.img_classes = "data/ImageClasses_FSC147.txt"
        self.cls_list = get_image_classes(self.img_classes)
        cfg.merge_from_file(config)

        self.device = torch.device(device)

        # Load model and regressor
        self.model = build_model(cfg)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_imgnet = copy.deepcopy(self.model)
        checkpoint = torch.load(model_ckpt, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])

        self.regressor = get_regressor(cfg)
        self.regressor.load_state_dict(torch.load('models/ZSC/pretrain/regressor.pth', map_location=self.device)) 
        self.regressor = self.regressor.to(self.device)
        self.regressor.eval()

        self.vae_feats = np.load('models/ZSC/checkpoints/bmnet+_ep3_epoch300_no_refiner/fsc_vae_feats.npy', allow_pickle=True)

        self.img_trans = transforms.Compose([
            transforms.Resize(size=384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.query_trans = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_text_prompt(self, text):
        """
        Implement the specific prompt retrieval logic for the ZSC-Count model.
        """
        label = self.cls_list.index(text)

        return label

    def infer(self, img, text):
        """
        Perform inference on a single image with a given text prompt.

        Args:
            img: The input image for inference.
            text: The text prompt for ZSC model.

        Returns:
            Tuple containing:
                - pred_cnt: Number of predicted masks
                - density_map_tensor: The density mask tensor
        """
        w, h = img.size

        patches = []
        scale_embedding = [] 
        scale_number = 20
        boxes = random_aug_boxes(h, w)
        for box in boxes:
            x1, y1 = box[0].astype(np.int32)
            x2, y2 = box[2].astype(np.int32)
            patch = img.crop((x1, y1, x2, y2))
            patches.append(self.query_trans(patch))
            scale = (x2 - x1) / w * 0.5 + (y2 -y1) / h * 0.5
            scale = scale // (0.5 / scale_number)
            scale = scale if scale < scale_number - 1 else scale_number - 1
            scale_embedding.append(0)
        patches = torch.stack(patches, dim=0)

        img = self.img_trans(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)

        patches = patches.unsqueeze(0)
        scale_embedding = torch.tensor(scale_embedding).unsqueeze(0).to(torch.int64)
        scale_embedding = scale_embedding.to(self.device)
        patches = patches.to(self.device)

        with torch.no_grad():
            ori_features1 = self.model.backbone(img)
            ori_features = self.model.input_proj(ori_features1)
            
            img = F.interpolate(img, [384,384])
            features = self.model.backbone(img)
            features = self.model.input_proj(features)

            vae_feature = self.vae_feats[text]

            patches = patches.flatten(0, 1)
            patch_feature = self.model.backbone(patches) # obtain feature maps for exemplar patches
            vae_sel_idx = select_feats_vae_imgnet(torch.from_numpy(vae_feature).to(self.device), patches, self.model_imgnet)
            patch_feature2 = self.model.EPF_extractor(patch_feature[vae_sel_idx], scale_embedding[:, vae_sel_idx])
            refined_feature, patch_feature2 = self.model.refiner(ori_features, patch_feature2)
            counting_feature, _ = self.model.matcher(refined_feature, patch_feature2)

            feats_all = []
            for m_idx in range(patch_feature2.shape[0]):
                counting_feature, _ = self.model.matcher(features, patch_feature2[[m_idx]])
                feats_all.append(counting_feature)
            counting_feature = torch.stack(feats_all).squeeze(1)
            scores = self.regressor(counting_feature)
            sel_idx = scores.argsort(0)[:3]
            patch_feature3 = patch_feature2[sel_idx[:,0]]
            counting_feature, _ = self.model.matcher(refined_feature, patch_feature3)
            
            density_map_tensor = self.model.counter(counting_feature)
        
        # Integrate over the density_map tensor
        pred_cnt = torch.sum(density_map_tensor).item()
        
        return pred_cnt, density_map_tensor