import os
import torch
import torch.nn as nn
import argparse
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import open_clip
from .base_model import BaseModel
import sys
import inflect
import yaml
from dotmap import DotMap

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'VLCounter')))
from .VLCounter.tools.models.VLCounter import Counter
from .VLCounter.tools.dataset.tokenizer import tokenize


SCALE_FACTOR = 60


class VLCounterModel(BaseModel):
    def __init__(self, img_directory, split_images, split_classes, model_ckpt='pretrained_models/182_best.pth', config="models/VLCounter/config_files/FSC.yaml"):
        super().__init__(img_directory, split_images, split_classes)

        with open(config, 'r') as f:
            config = yaml.safe_load(f)
        args = DotMap(config)
        args.config = config
        args.enc = "spt"
        args.num_tokens = 10
        args.patch_size = 16

        self.model_name = "VLCounter"
        self.model = Counter(args=args).cuda()
        checkpoint = torch.load(model_ckpt)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        self.img_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])

        
    def get_text_prompt(self, text, prompt="plural"):
        """
        Implement the specific prompt retrieval logic for the VLCounter model.
        """

        single_plural_prompt_templates = ['A photo of a {}.',
                                        'A photo of a small {}.',
                                        'A photo of a medium {}.',
                                        'A photo of a large {}.',
                                        'This is a photo of a {}.',
                                        'This is a photo of a small {}.',
                                        'This is a photo of a medium {}.',
                                        'This is a photo of a large {}.',
                                        'A {} in the scene.',
                                        'A photo of a {} in the scene.',
                                        'There is a {} in the scene.',
                                        'There is the {} in the scene.',
                                        'This is a {} in the scene.',
                                        'This is the {} in the scene.',
                                        'This is one {} in the scene.',
                                    ]
        multi_plural_prompt_templates = ['a photo of a number of {}.',
                                        'a photo of a number of small {}.',
                                        'a photo of a number of medium {}.',
                                        'a photo of a number of large {}.',
                                        'there are a photo of a number of {}.',
                                        'there are a photo of a number of small {}.',
                                        'there are a photo of a number of medium {}.',
                                        'there are a photo of a number of large {}.',
                                        'a number of {} in the scene.',
                                        'a photo of a number of {} in the scene.',
                                        'there are a number of {} in the scene.',
                                    ]

        engine = inflect.engine()
        if prompt == "plural":
            text = [template.format(engine.plural(text)) if engine.singular_noun(text) == False else template.format(text) for template in multi_plural_prompt_templates]
        elif prompt == "sigle":
            text = [template.format(text) if engine.singular_noun(text) == False else template.format(engine.plural(text)) for template in single_plural_prompt_templates]
        else:
            raise NotImplementedError
        
        return text


    def infer(self, img, text):
        """
        Implement the specific inference logic for the VLCounter model.
        """
        tokenized_text = tokenize(text)
        tokenized_text = tokenized_text.unsqueeze(0)
        tokenized_text = tokenized_text.cuda()

        w, h = img.size
        new_H = 384
        new_W = 16 * int((w / h * 384) / 16)
        img = transforms.Resize((new_H, new_W))(img)
        img = self.img_trans(img)
        img = img.unsqueeze(0)
        img = img.cuda()

        _, _, h, w = img.shape
        density_map = torch.zeros([h, w])
        density_map = density_map.cuda()
        attn_map = torch.zeros([h, w])
        attn_map = attn_map.cuda()

        start = 0
        prev = -1
        with torch.no_grad():
            while start + 383 < w:
                output, attn, _ = self.model(img[:, :, :, start:start + 384], tokenized_text)
                output = output.squeeze(0).squeeze(0)
                attn = attn.squeeze(0).squeeze(0)
                b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                d1 = b1(output[:, 0:prev - start + 1])
                a1 = b1(attn[:, 0:prev - start + 1])
                b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                d2 = b2(output[:, prev - start + 1:384])
                a2 = b2(attn[:, prev - start + 1:384])

                b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                density_map_l = b3(density_map[:, 0:start])
                density_map_m = b1(density_map[:, start:prev + 1])
                attn_map_l = b3(attn_map[:, 0:start])
                attn_map_m = b1(attn_map[:, start:prev + 1])
                b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                density_map_r = b4(density_map[:, prev + 1:w])
                attn_map_r = b4(attn_map[:, prev + 1:w])

                density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2
                attn_map = attn_map_l + attn_map_r + attn_map_m / 2 + a1 / 2 + a2

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
