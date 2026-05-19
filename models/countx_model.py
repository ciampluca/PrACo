import torch
import torch.nn as nn
import argparse
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import open_clip
from .base_model import BaseModel
from .CounTX.util import misc
from .CounTX.util.FSC147 import TTensor
from .CounTX.models_counting_network import CountingNetwork

SCALE_FACTOR = 60

class CounTXModel(BaseModel):
    def __init__(self, img_directory, split_images, split_classes, model_ckpt="pretrained_models/paper-model.pth", device="cuda"):
        super().__init__(img_directory, split_images, split_classes)
        
        class MyArgumentParser(argparse.ArgumentParser):
            def __init__(self, resume, *args, **kwargs):
                self.resume = resume
                super().__init__(*args, **kwargs)

        parser = MyArgumentParser(resume=model_ckpt)
        device = torch.device(device)

        self.model_name = "CounTX"
        self.model = CountingNetwork()
        misc.load_model_FSC(args=parser, model_without_ddp=self.model, device=device)
        self.model.to(device)

        self.model.eval()

        self.device = device
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")

        self.open_clip_vit_b_16_preprocess = transforms.Compose(
            [
                transforms.Resize(
                    size=224,
                    interpolation=InterpolationMode.BICUBIC,
                    max_size=None,
                    antialias="warn",
                ),
                ## transform to tensor
                #transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        
    def get_text_prompt(self, text):
        """
        Implement the specific prompt retrieval logic for the CounTX model.
        """
        return f"the {text}"

    def infer(self, img, text):
        """
        Implement the specific inference logic for the CounTX model.
        """
        text_descriptions = self.clip_tokenizer(text).squeeze(-2)
        text_descriptions = text_descriptions.unsqueeze(0)
        text_descriptions = text_descriptions.to(self.device, non_blocking=True)

        W, H = img.size
        
        # Resize image to have height of exactly 384 (like FSC147 dataset)
        # This ensures slices will be square (384x384) when taking 384-width chunks
        scale = 384 / H
        new_H = 384
        new_W = int(W * scale)
            
        # Round to multiples of 16
        new_H = 16 * int(new_H / 16)
        new_W = 16 * int(new_W / 16)
        
        image_resized = image = transforms.Resize((new_H, new_W))(img)
        resized_image = TTensor(image)
        image = resized_image.unsqueeze(0)
        image = image.to(self.device, non_blocking=True)
        _, _, h, w = image.shape

        density_map = torch.zeros([h, w])
        density_map = density_map.to(self.device, non_blocking=True)
        start = 0
        prev = -1
        with torch.no_grad():
            while start + 383 < w:
                
                #img_slice = image_resized.crop((start, 0, start + 384, new_H)) 
                ## convert img_slice to tensor and preprocess
                ##img_slice_tensor = TTensor(img_slice).unsqueeze(0).to(self.device)
                #img_tensor = (self.open_clip_vit_b_16_preprocess(img_slice)).unsqueeze(0).to(self.device) 
                #output, = self.model(img_tensor, text_descriptions)
                
                (output,) = self.model(
                    self.open_clip_vit_b_16_preprocess(
                        image[:, :, :, start : start + 384]
                    ),
                    text_descriptions,
                )
                
                output = output.squeeze(0)
                b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                d1 = b1(output[:, 0 : prev - start + 1])
                b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                d2 = b2(output[:, prev - start + 1 : 384])

                b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                density_map_l = b3(density_map[:, 0:start])
                density_map_m = b1(density_map[:, start : prev + 1])
                b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                density_map_r = b4(density_map[:, prev + 1 : w])

                density_map = (
                    density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2
                )

                prev = start + 383
                start = start + 128
                if start + 383 >= w:
                    if start == w - 384 + 128:
                        break
                    else:
                        start = w - 384

        density_map = density_map / SCALE_FACTOR
        pred_cnt = torch.sum(density_map).item()

        density_map = density_map.cpu()
        density_map_tensor = density_map

        return pred_cnt, density_map_tensor
