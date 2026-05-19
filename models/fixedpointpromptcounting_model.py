from .base_model import BaseModel


import os, torch
from PIL import Image
import numpy as np

from .FixedPointPromptCounting.datasets.utils import NormalSample

import clip
import cv2
from .FixedPointPromptCounting.data_fsc147 import llama as llama
import spacy

import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
_transform = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def imgprocess(img, patch_size=[16, 16], scale_factor=1):
    w, h = img.size
    ph, pw = patch_size
    nw = int(w * scale_factor / pw + 0.5) * pw
    nh = int(h * scale_factor / ph + 0.5) * ph

    ResizeOp = Resize((nh, nw), interpolation=InterpolationMode.BICUBIC)
    img = ResizeOp(img).convert("RGB")
    return _transform(img)

class SampleBuilder:
    can_h = 512
    can_w = 768

    normalfunc = NormalSample(can_h, can_w, train=False)

    def __init__(self, device, llama_model_weights_path=os.path.join(os.path.dirname(__file__), "FixedPointPromptCounting/data_fsc147/llama_model_weights")):
        self.clipmodel, self.preprocess = clip.load("ViT-B/16", device=device)
        self.clip_inres = self.clipmodel.visual.input_resolution
        self.clip_ksize = self.clipmodel.visual.conv1.kernel_size
        self.llama_model, self.llama_preprocess = llama.load("BIAS-7B", llama_model_weights_path, "cuda" if torch.cuda.is_available() else "cpu")
        self.nlp = spacy.load("en_core_web_sm")
        self.device = device

    def llama_describe(self, pil_img, ques="Objects and corresponding counts in the picture"):
        
        prompt = llama.format_prompt(ques)

        img = self.llama_preprocess(pil_img).unsqueeze(0)
        # Convert the image tensor to half precision if the model is on GPU
        if torch.cuda.is_available():
            img = img.half().cuda()
        result = self.llama_model.generate(img, [prompt])[0]
        
        return result
    
    def spacy_process(self, description : str, gt_category : str):
        nlp_desc = self.nlp(description)
        noun_desc = {}
        for chunk in nlp_desc.noun_chunks:
            text, root, tag = chunk.text, chunk.root.lemma_, chunk.root.tag_
            if tag not in ["NN", "NNS"]:
                continue
            if root in ["image", "picture"]:
                continue
            if root in noun_desc:
                noun_desc[root].append(text)
            else:
                noun_desc[root] = [text]
        nlp_gt = self.nlp(f"some {gt_category}")
        gt_noun = {
            nlp_gt[-1].lemma_: gt_category
        }
        return dict(
            nouns = noun_desc,
            gt_noun = gt_noun,
            desc = description
        )

    def compute_clip_mask(self, img, prompt):
        #obj_info = obj_infos[f'{imid}.jpg']['nouns']
        #obj_info = {prompt: [prompt]}
        
        all_infos = self.spacy_process(self.llama_describe(img), gt_category=prompt)
        obj_info = all_infos['nouns']
        items, prompts = list(obj_info.keys()), list(obj_info.values())
        # print(prompts, weight)

        # prompts = [info['category'], "backgeround"]
        chunknum = [len(prompt) for prompt in prompts]
        prompts = sum(prompts, [])
        
        # add background category
        prompts += ['image']
        chunknum += [1]
        items += ['image']

        #gt_info = {prompt: [prompt]}
        gt_info = all_infos['gt_noun']
        gt_item, gt_prompts = list(gt_info.keys())[0], list(gt_info.values())[0]
        if gt_item not in items:
            prompts, chunknum = prompts + [gt_prompts], chunknum + [1]
            items = items + [gt_item]
        gtid = items.index(gt_item)

        imw, imh = img.size
        image = imgprocess(img, self.clip_ksize, scale_factor=1).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            text = clip.tokenize(items, truncate=True).to(self.device)
            text_features = self.clipmodel.encode_text(text)[None, ...]
            dense_features = self.clip_encode_dense(image)
            
            dense_features = F.normalize(dense_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # cosine similarity as logits
            logit_scale = self.clipmodel.logit_scale.exp()
            logits_per_position = logit_scale * dense_features @ text_features.permute(0, 2, 1) # (1, 256, 768) @ (1, 768, CLS_NUM) = (1, 256, CLS_NUM)
            logits_per_position = logits_per_position.softmax(dim=-1)[0]
            # # split into different iterms
            # logits_per_item = [torch.sum(logit, dim=-1, keepdim=True) for logit in torch.split(logits_per_position, chunknum, dim=-1)]
            # probs = torch.cat(logits_per_item, dim=-1)
            
            probs = logits_per_position
            prob = probs.reshape(imh // self.clip_ksize[0], imw // self.clip_ksize[1], -1)


            probt = prob[..., gtid]
            probt = F.interpolate(probt[None, None, ...], scale_factor=16, mode='bilinear', align_corners=False)[0, 0]
            
            np_probt = (probt * 255).long()
            np_probt = np_probt.cpu().numpy().astype('uint8')
            return np_probt
            #cv2.imwrite(os.path.join(llamaroot, f"{imid}.png"), np_probt)

    def getSample(self, image : Image, text : str = "", points = [[]], boxes = [[]]):
        """
        - image: PIL image
        - points: list of (x, y) coordinates
        - boxes: list of (x1, y1, x2, y2) coordinates
        - returns image, dotmap, boxmap, dotprompt, clip_mask
        """
            #print(imgid)
        #sample = self.samples[imgid]
        if image.size != (self.can_w, self.can_h):
            image, points, boxes = self.resize_image(image, points, boxes)
        
        points_was_empty = False
        if points is not None and len(points) > 0 and len(points[0]) > 0:
            points = torch.tensor(points).round().long() # N x (w, h)
        else:
            # since normalfunc expects points, we need to pass a dummy tensor
            # we cannot provide random points, as it will affect the dotmap
            # by the way, the normalfunc method tries to access dots[:, 0] and dots[:, 1]
            # so we need to provide a tensor with at least 2 elements 
            points = torch.zeros((1, 2), dtype=torch.long)
            points_was_empty = True

        # llama clip mask
        #mapath = os.path.join(self.root_path, sample['maskpath'])
        #mask = Image.open(mapath).convert('L')
        mask_np = self.compute_clip_mask(image, text)
        mask = Image.fromarray(mask_np).convert('L')
        
        # box
        boxes = torch.tensor(boxes).round().long().reshape(-1, 4) # 3 x ((left, top), (right, bottom))
        if boxes is not None and len(boxes) > 0 and len(boxes[0]) > 0:
            # h = boxes[:, 3] - boxes[:, 1]
            # w = boxes[:, 2] - boxes[:, 0]
            # boxes = torch.stack((h, w), dim=-1)
            # print(boxes)
            box_sel = torch.randn((boxes.size(0), 1))
            box_sel = (box_sel == box_sel.max(dim=0, keepdim=True).values)
            boxes = (boxes * box_sel).sum(dim=0, keepdim=True)
            # print(boxes, box_sel)
        else:
            boxes = None
        image, dotmap, boxmap, clip_mask, _ = self.normalfunc(image, dots=points, boxes=boxes, clip_mask=mask)
        
        # generate point prompt
        dotppt = (torch.rand_like(dotmap) * dotmap).flatten()
        dotsel = dotppt.argmax()
        dotppt = torch.zeros_like(dotppt)
        dotppt[dotsel] = 1
        dotppt = dotppt.view_as(dotmap)

        return image, dotmap, boxmap, dotppt, clip_mask
    
    def resize_image(self, image : Image, points = None, boxes = None):
        # convert PIL Image to cv2, equivalent to cv2.imread
        img = np.array(image)
        
        H, W, _ = img.shape
        nH, nW = min(int(round(H / 16) * 16), self.can_h), min(int(round(W / 16) * 16), self.can_w)
        if nH < self.can_h and nW < self.can_w:
            rh, rw = self.can_h / nH, self.can_w / nW
            if rh < rw:
                nH, nW = self.can_h, min(int((rh * nW) / 16 + 0.5) * 16, self.can_w)
            else:
                nH, nW = min(int((rw * nH) / 16 + 0.5) * 16, self.can_h), self.can_w
        rh, rw = nH / H, nW / W
        ph, pw = (self.can_h - nH) // 2, (self.can_w - nW) // 2
        

        # resize image
        img = cv2.resize(img, (nW, nH), interpolation = cv2.INTER_AREA)
        canvas = np.zeros((self.can_h, self.can_w, 3), dtype='uint8')
        canvas[ph:ph+nH, pw:pw+nW, :] = img
        
        resized_pil_image = Image.fromarray(canvas)
        #imgpath = os.path.join(nroot, imgdir, f'{imid}.jpg')
        #cv2.imwrite(imgpath, canvas)
        
        # resize box
        #boxes = label['box_examples_coordinates']
        if boxes is not None and len(boxes) > 0 and len(boxes[0]) > 0:
            nboxes = np.array(boxes, dtype='float32') # ((w1, h1), (w1, h2), (w2, h2), (w2, h1)) # N 4 2
            nboxes[:, :, 0] = nboxes[:, :, 0] * rw + pw
            nboxes[:, :, 1] = nboxes[:, :, 1] * rh + ph
            box_lt = nboxes[:, 0, :]
            box_rb = nboxes[:, 2, :]
            if box_lt.min() < 0 or box_rb.min() < 0:
                print(boxes, rw, pw, rh, ph, box_lt, box_rb)
                raise
            nboxes = np.stack((box_lt, box_rb), axis=1)
            resize_boxes = nboxes.tolist()
        else:
            resize_boxes = [[]]
        # relocate point
        #pots = label['points'] 
        if points is not None and len(points) > 0 and len(points[0]) > 0:
            npots = np.array(points)
            npots[:, 0] = npots[:, 0] * rw + pw
            npots[:, 1] = npots[:, 1] * rh + ph
            resize_pots = npots.tolist()
        else:
            resize_pots = [[]]
        # write info into label
        #info[imid] = dict(
        #    imagepath = os.path.join(imgdir, f'{imid}.jpg'),
        #    points = resize_pots,
        #    boxes = resize_boxes,
        #    category = cates[imid]
        #)
        return resized_pil_image, resize_pots, resize_boxes
        #with open(os.path.join(nroot, f'fsc147_{can_h}x{can_w}.json'), 'w+') as f:
        #    json.dump(info, f)
    @torch.inference_mode()
    def clip_encode_dense(self, x):
        # modified from CLIP
        x = x.half()
        x = self.clipmodel.visual.conv1(x)  
        feah, feaw = x.shape[-2:]

        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1) 
        class_embedding = self.clipmodel.visual.class_embedding.to(x.dtype)

        x = torch.cat([class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1]).to(x), x], dim=1)


        pos_embedding = self.clipmodel.visual.positional_embedding.to(x.dtype)
        tok_pos, img_pos = pos_embedding[:1, :], pos_embedding[1:, :]
        pos_h = self.clip_inres // self.clip_ksize[0]
        pos_w = self.clip_inres // self.clip_ksize[1]

        assert img_pos.size(0) == (pos_h * pos_w), f"the size of pos_embedding ({img_pos.size(0)}) does not match resolution shape pos_h ({pos_h}) * pos_w ({pos_w})"

        img_pos = img_pos.reshape(1, pos_h, pos_w, img_pos.shape[1]).permute(0, 3, 1, 2)
        # print("[POS shape]:", img_pos.shape, (feah, feaw))
        img_pos = torch.nn.functional.interpolate(img_pos, size=(feah, feaw), mode='bicubic', align_corners=False)
        img_pos = img_pos.reshape(1, img_pos.shape[1], -1).permute(0, 2, 1)

        pos_embedding = torch.cat((tok_pos[None, ...], img_pos), dim=1)

        x = x + pos_embedding
        
        x = self.clipmodel.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = torch.nn.Sequential(*self.clipmodel.visual.transformer.resblocks[:-1])(x)
        
        # LastTR.attention
        LastTR = self.clipmodel.visual.transformer.resblocks[-1]
        x1 = LastTR.ln_1(x)

        linear = torch._C._nn.linear

        # # ------ [maskclip with refine key] ----------
        q, k, v = linear(x1, LastTR.attn.in_proj_weight, LastTR.attn.in_proj_bias).chunk(3, dim=-1)
        qkv = torch.stack((q, k, v), dim=0)
        qkv = linear(qkv, LastTR.attn.out_proj.weight, LastTR.attn.out_proj.bias)
        q, k, attn_output = qkv[0], qkv[1], qkv[2]

        x = attn_output + x
        x = x + LastTR.mlp(LastTR.ln_2(x))

        # print("[x]:", x.shape)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # preserve all spatial tokens
        x = self.clipmodel.visual.ln_post(x[:, :, :])

        if self.clipmodel.visual.proj is not None:
            x = x @ self.clipmodel.visual.proj

        return x[:, 1:]


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

import argparse
from .FixedPointPromptCounting.config import get_config

from .FixedPointPromptCounting.models import build_model
from .FixedPointPromptCounting.optimizer import build_optimizer

from .FixedPointPromptCounting.lr_scheduler import build_scheduler
from .FixedPointPromptCounting.utils import load_checkpoint

def get_args_parser(device = None, checkpoint_path = None):
    parser = argparse.ArgumentParser('Counting Everything training and evaluation script', add_help=False)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint', default=checkpoint_path)
    parser.add_argument('--use-checkpoint', action='store_true', default=checkpoint_path is not None,
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    args, unparsed = parser.parse_known_args()

    #args.MODEL.DEVICE = device

    config = get_config(args)

    return args, config

import logging
logger = logging.getLogger()

class FixedPointPromptCountingModel(BaseModel):
    def __init__(self, img_directory, split_images, split_classes, device=None, checkpoint_path = os.path.join(os.path.dirname(__file__), "FixedPointPromptCounting/fxp.pth")
        ):
        super().__init__(img_directory, split_images, split_classes)

        if device is None:
            device = get_device()
        
        self.model_name = "FixedPointPromptCounting"
        self.sample_builder = SampleBuilder(device=device)

        args, self.config = get_args_parser(device, checkpoint_path)

        self.model, criterion = build_model(self.config.MODEL)
        self.model.cuda()
        criterion.cuda()
        optimizer = build_optimizer(self.config, self.model)
        self.model_without_ddp = self.model

        len_data_loader_train = 6000#len(data_loader_train) # a caso, tanto non si fa training davvero

        lr_scheduler = build_scheduler(self.config, optimizer, len_data_loader_train)

        #self.config.MODEL.RESUME = checkpoint_path
        #self.config.MODEL.DEVICE = device

        max_accuracy = load_checkpoint(self.config, self.model_without_ddp, optimizer, lr_scheduler, logger)
        self.model.to(device)
        self.model.eval()

        
    def get_text_prompt(self, text):
        """
        Implement the specific prompt retrieval logic for the FixedPointPromptCounting model.
        """
        return f"{text}"

    def infer(self, img, text, keep_initial_dims=False):
        """
        Implement the specific inference logic for the FixedPointPromptCounting model.
        """
        initial_w, initial_h = img.size
        image, dotmap, boxmap, dotprompt, clip_mask = self.sample_builder.getSample(img, text)

        boxmap = boxmap.cuda(non_blocking=True)
        dotmap = dotmap.cuda(non_blocking=True)
        clip_mask = clip_mask.cuda(non_blocking=True)

        # we need to unsqueeze the image
        image = image.unsqueeze(0).cuda()

        with torch.no_grad():
            for i, masks in enumerate([clip_mask]): #boxmap, dotmap, clip_mask
                if masks is None:
                    continue
                nh, nw = image.shape[-2:]
                boxmaps = F.adaptive_max_pool2d(masks, (nh // 16, nw // 16))
                denmap = self.model(image, boxmaps=boxmaps).relu()
                density_map_tensor = denmap
                #loss = criterion(denmap, tardot=target)
                pred_cnt = (denmap / self.config.MODEL.FACTOR).sum(dim=(1,2,3)).item()
                #tarnum = target.sum(dim=(1,2,3))
                #print(f"predicted num: ", prednum)
                #diff = torch.abs(prednum - tarnum)
                #mae, mse = diff.mean(), (diff ** 2).mean()

        if keep_initial_dims:
            density_map_tensor = F.interpolate(density_map_tensor, size=(initial_h, initial_w), mode='bilinear', align_corners=False)

        return pred_cnt, density_map_tensor.squeeze().cpu()
