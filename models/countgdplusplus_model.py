import torch
import random
import os
from PIL import Image, ImageDraw, ImageFont
from itertools import cycle
import json
cwd = os.path.dirname(__file__)

import sys

if os.path.join(cwd, "CountGDPlusPlus") not in sys.path:
    sys.path.append(os.path.join(cwd, "CountGDPlusPlus"))

from PIL import Image, ImageDraw, ImageFont

import torchvision.transforms.functional as F
import numpy as np
import argparse

from .CountGDPlusPlus.util.slconfig import SLConfig, DictAction
from .CountGDPlusPlus.util.misc import nested_tensor_from_tensor_list
from .CountGDPlusPlus.datasets import transforms as T

#import scipy.ndimage as ndimage
#import matplotlib.pyplot as plt

from base_model import BaseModel



#CONF_THRESH = 0.23

# MODEL:
def get_args_parser():
    parser = argparse.ArgumentParser("CountGD++ Arg Parser", add_help=False)
    parser.add_argument(
        "--device", default="cuda", help="device to use for inference"
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )

    # dataset parameters
    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")
    parser.add_argument("--synth_exemplar_folder", type=str, default="CountGDPlusPlus/synthetic_exemplars", help="name of folder containing synthetic exemplars and JSON file with corresponding box coordinates")

    # training parameters
    parser.add_argument("--note", default="", help="add some notes to the experiment")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--pretrain_model_path",
        help="load from other checkpoint",
        default="checkpoint_best_regular.pth",
    )
    parser.add_argument("--finetune_ignore", type=str, nargs="+")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_false")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--find_unused_params", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_log", action="store_true")

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--local_rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument(
        "--local-rank", type=int, help="local rank for DistributedDataParallel"
    )
    parser.add_argument("--amp", action="store_true", help="Train with mixed precision")
    return parser

def draw_boxes(
    image,
    boxes,
    labels=None,
    colors=None,
    width=3,
    fmt="xyxy",             # one of: "xyxy", "xywh", "cxcywh"
    normalized=False,       # True if coords are in [0,1]
    font=None,              # a PIL ImageFont; if None, will use default
    label_bg_alpha=160,     # 0..255 (used if image is RGBA)
):
    """
    Draw rectangles on a PIL.Image with optional labels.

    Args:
        image (PIL.Image.Image): input image (RGB or RGBA).
        boxes (list): list of boxes. Each box is [x1,y1,x2,y2] for 'xyxy',
                      [x,y,w,h] for 'xywh', or [cx,cy,w,h] for 'cxcywh'.
        labels (list|None): list of text labels matching boxes (optional).
        colors (list|tuple|None): either a single (R,G,B) or list per box.
        width (int): stroke width for rectangle outlines.
        fmt (str): 'xyxy' | 'xywh' | 'cxcywh'.
        normalized (bool): interpret coords in [0,1] if True.
        font (ImageFont|None): optional font for labels.
        label_bg_alpha (int): background alpha for label box (RGBA images).
    Returns:
        PIL.Image.Image: image with boxes drawn (same mode as input).
    """
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")

    W, H = image.size

    def to_xyxy(b):
        if fmt == "xyxy":
            x1, y1, x2, y2 = b
        elif fmt == "xywh":
            x, y, w, h = b
            x1, y1, x2, y2 = x, y, x + w, y + h
        elif fmt == "cxcywh":
            cx, cy, w, h = b
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            raise ValueError("fmt must be 'xyxy', 'xywh', or 'cxcywh'")

        if normalized:
            x1, x2 = x1 * W, x2 * W
            y1, y2 = y1 * H, y2 * H

        # normalize, clip, and cast to int
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = max(0, min(int(round(x1)), W - 1))
        y1 = max(0, min(int(round(y1)), H - 1))
        x2 = max(0, min(int(round(x2)), W - 1))
        y2 = max(0, min(int(round(y2)), H - 1))
        return x1, y1, x2, y2

    # Prepare colors
    if colors is None:
        colors_iter = cycle([(255, 0, 0), (0, 255, 0), (0, 180, 255), (255, 165, 0), (255, 0, 255)])
    elif isinstance(colors, tuple) and len(colors) == 3:
        colors_iter = cycle([colors])
    else:
        colors_iter = cycle(colors)

    # Font
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

    # We’ll draw on an overlay (RGBA) to support semi-transparent label backgrounds cleanly.
    base_mode = image.mode
    if base_mode != "RGBA":
        work = image.convert("RGBA")
    else:
        work = image.copy()

    overlay = Image.new("RGBA", work.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = to_xyxy(box)
        color = next(colors_iter)

        # Rectangle outline
        # Pillow supports 'width' in rectangle; fallback to multiple rectangles if needed.
        try:
            draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=width)
        except TypeError:
            # Fallback for very old Pillow versions
            for k in range(width):
                draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color + (255,))

        # Label (if provided)
        if labels and i < len(labels) and labels[i]:
            text = str(labels[i])

            # Measure text
            try:
                tb = draw.textbbox((0, 0), text, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
            except Exception:
                tw, th = draw.textsize(text, font=font)

            # Place label at top-left of box
            pad = 2
            tx1, ty1 = x1, max(0, y1 - th - 2 * pad)
            tx2, ty2 = x1 + tw + 2 * pad, ty1 + th + 2 * pad

            # Background (semi-transparent if RGBA)
            bg = color + (label_bg_alpha,)
            draw.rectangle([tx1, ty1, tx2, ty2], fill=bg)

            # Text (white or black depending on bg luminance)
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_fill = (0, 0, 0, 255) if lum > 186 else (255, 255, 255, 255)
            draw.text((tx1 + pad, ty1 + pad), text, font=font, fill=text_fill)

    # Composite overlay
    out = Image.alpha_composite(work, overlay)
    if base_mode != "RGBA":
        out = out.convert(base_mode)
    return out

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# Get counting model.
def build_model_and_transforms(args):
    cfg = SLConfig.fromfile(os.path.join(cwd, "CountGDPlusPlus/cfg_app.py"))
    cfg.merge_from_dict({"text_encoder_type": os.path.join(cwd, "CountGDPlusPlus/checkpoints/bert-base-uncased")})
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    #scales = getattr(args, "data_aug_scales", scales)
    #max_size = getattr(args, "data_aug_max_size", max_size)
    
    scales = [800]
    max_size = 1333

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    data_transform = T.Compose(
        [
            T.RandomResize(scales, max_size=max_size),
            normalize,
        ]
    )

    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # we use register to maintain models from catdet6 on.
    from .CountGDPlusPlus.models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, _, _ = build_func(args)

    checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")["model"]
    model.load_state_dict(checkpoint, strict=False)

    model.eval()

    return model, data_transform

def get_box_inputs(prompts):
    box_inputs = []
    for prompt in prompts:
        if prompt[2] == 2.0 and prompt[5] == 3.0:
            box_inputs.append([prompt[0], prompt[1], prompt[3], prompt[4]])

    return box_inputs

def get_ind_to_filter(text, word_ids, keywords):
    if len(keywords) <= 0:
        return list(range(len(word_ids)))
    input_words = text.split()
    keywords = keywords.split(",")
    keywords = [keyword.strip() for keyword in keywords]

    word_inds = []
    for keyword in keywords:
        if keyword in input_words:
            if len(word_inds) <= 0:
                ind = input_words.index(keyword)
                word_inds.append(ind)
            else:
                ind = input_words.index(keyword, word_inds[-1])
                word_inds.append(ind)
        else:
            raise Exception("Only specify keywords in the input text!")

    inds_to_filter = []
    for ind in range(len(word_ids)):
        word_id = word_ids[ind]
        if word_id in word_inds:
            inds_to_filter.append(ind)

    return inds_to_filter

def get_pseudo_exemplars(model_output, image_size, box_threshold, num_exemplars=3):
    input_ids = model_output["input_ids"][0]
    logits = model_output["pred_logits"].sigmoid()[0][:, :]
    boxes = model_output["pred_boxes"][0]

    # [pos_neg_split_idx] is the index of the first occurence of the "." separating token.
    for idx in range(len(input_ids)):
        token = input_ids[idx]
        if token == 1012:
            pos_neg_split_idx = idx
            break

    pos_logits = logits[:, :(pos_neg_split_idx + 1)]
    neg_logits = logits[:, (pos_neg_split_idx + 1):]

    # Get positive pseudo-exemplars
    # Stage 1 filtering
    pos_box_mask = pos_logits.max(dim=-1).values > box_threshold
    pos_boxes = boxes[pos_box_mask, :]
    pos_logits = pos_logits[pos_box_mask, :]

    # Stage 2 filtering
    neg_logits_pos_filter = neg_logits[pos_box_mask, :]
    pos_box_mask = pos_logits.max(dim=-1).values > neg_logits_pos_filter.max(dim=-1).values
    pos_boxes = pos_boxes[pos_box_mask, :]
    pos_logits = pos_logits[pos_box_mask, :]
    pos_scores = pos_logits.max(dim=-1).values
    num_exemplars_pos = min(num_exemplars, pos_boxes.shape[0])
    
    # Out of all the boxes, select at most [num_exemplars] of the highest scoring boxes.
    pos_scores, indices = torch.sort(pos_scores, dim=0, descending=True)
    pos_boxes = pos_boxes[indices, :]
    pos_pseudo_exemplars = pos_boxes[:num_exemplars_pos, :]
    # Convert the normalized boxes to the exemplars format.
    (img_h, img_w) = (image_size[0], image_size[1])
    cx = img_w * pos_pseudo_exemplars[:, 0]
    cy = img_h * pos_pseudo_exemplars[:, 1]
    w = img_w * pos_pseudo_exemplars[:, 2]
    h = img_h * pos_pseudo_exemplars[:, 3]
    x0 = torch.clamp(cx - w/2, min=0, max=img_w)
    x1 = torch.clamp(cx + w/2, min=0, max=img_w)
    y0 = torch.clamp(cy - h/2, min=0, max=img_h)
    y1 = torch.clamp(cy + h/2, min=0, max=img_h)
    pos_pseudo_exemplars = torch.stack([x0, y0, x1, y1], dim=-1)
    
    # Get negative pseudo-exemplars
    # Stage 1 filtering
    neg_box_mask = neg_logits.max(dim=-1).values > box_threshold
    neg_boxes = boxes[neg_box_mask, :]
    neg_logits = neg_logits[neg_box_mask, :]

    # Stage 2 filtering
    pos_logits_neg_filter = logits[neg_box_mask, :(pos_neg_split_idx + 1)]
    neg_box_mask = neg_logits.max(dim=-1).values > pos_logits_neg_filter.max(dim=-1).values
    neg_boxes = neg_boxes[neg_box_mask, :]
    neg_logits = neg_logits[neg_box_mask, :]
    neg_scores = neg_logits.max(dim=-1).values
    num_exemplars_neg = min(num_exemplars, neg_boxes.shape[0])
    
    # Out of all the boxes, select at most [num exemplars] of the highest scoring boxes.
    neg_scores, indices = torch.sort(neg_scores, dim=0, descending=True)
    neg_boxes = neg_boxes[indices, :]
    neg_pseudo_exemplars = neg_boxes[:num_exemplars_neg, :]
    # Convert the normalized boxes to the exemplars format.
    (img_h, img_w) = (image_size[0], image_size[1])
    cx = img_w * neg_pseudo_exemplars[:, 0]
    cy = img_h * neg_pseudo_exemplars[:, 1]
    w = img_w * neg_pseudo_exemplars[:, 2]
    h = img_h * neg_pseudo_exemplars[:, 3]
    x0 = torch.clamp(cx - w/2, min=0, max=img_w)
    x1 = torch.clamp(cx + w/2, min=0, max=img_w)
    y0 = torch.clamp(cy - h/2, min=0, max=img_h)
    y1 = torch.clamp(cy + h/2, min=0, max=img_h)
    neg_pseudo_exemplars = torch.stack([x0, y0, x1, y1], dim=-1)

    return pos_pseudo_exemplars, neg_pseudo_exemplars
    
import inflect

class CountGDPlusPlusModel(BaseModel):
    
    #model_ckpt=os.path.join(cwd, "CountGD/checkpoint_best_regular.pth")
    def __init__(self, img_directory, split_images, split_classes, model_ckpt=os.path.join(cwd, "CountGDPlusPlus/checkpoints/countgd_plusplus.pth"), device=None, singularize:bool=True):
        super().__init__(img_directory, split_images, split_classes)
        
        parser = argparse.ArgumentParser("CountGD++", parents=[get_args_parser()])
        
        #parser.add_argument("--f", type=str, help="Workaround for ipynb")

        args = parser.parse_args([])
        args.pretrain_model_path = model_ckpt
        if device is None:
            device = get_device()
        self.model, self.transform = build_model_and_transforms(args)
        self.model = self.model.to(device)
        self.TEXT_THRESH = getattr(args, "text_threshold", 0.0)#args.get("text_threshold", 0.0)
        self.CONF_THRESH = getattr(args, "box_threshold", 0.23)#args.get("box_threshold", 0.23)
        self.model_name = "CountGDPlusPlus"#"CountGD_checkpoint_fsc147_best"

        # Load synthetic exemplars (generated using only text).
        self.synthetic_exemplars_folder = os.path.join(cwd, args.synth_exemplar_folder)
        with open(os.path.join(self.synthetic_exemplars_folder, "generated_exemplars.json"), 'r') as f:
            self.synthetic_exemplars = json.load(f)
        
        self.singularize = singularize
        if self.singularize:
            self.p = inflect.engine()
        
    def get_text_prompt(self, text, filename):
        """
        Implement the specific prompt retrieval logic for the CountGD model.
        """
        if text == "legos" and filename == "7611.jpg":
            return 'yellow lego stud' # prompt correction
        if self.singularize:
            singularized = self.p.singular_noun(text)
            ret = singularized if singularized and text not in ["sunglasses"] else text
            return ret
        else:
            return text
        
    def infer(self, img, pos_text, neg_text, pos_exemplar_img, neg_exemplar_img, pos_exemplars, neg_exemplars):
        """
        Implement the specific inference logic for the CountGD++ model.
        """

        # Start with synthetic positive/negative exemplars
        input_image, _ = self.transform(img, {"exemplars": torch.tensor([[0,0,0,0]])}) # dummy exemplar to be discarded later
        pos_exemplar_image, pos_target = self.transform(pos_exemplar_img, {"exemplars": torch.tensor(pos_exemplars)})
        pos_exemplars = pos_target["exemplars"]
        if neg_text is not None:
            neg_exemplar_image, neg_target = self.transform(neg_exemplar_img, {"exemplars": torch.tensor(neg_exemplars)})
            neg_exemplars = neg_target["exemplars"]
        
        device = next(self.model.parameters()).device
        input_image = input_image.unsqueeze(0).to(device)
        pos_exemplar_image = pos_exemplar_image.unsqueeze(0).to(device)
        if neg_text is not None:
            neg_exemplar_image = neg_exemplar_image.unsqueeze(0).to(device)

        if neg_text is not None:
            with torch.no_grad():
                
                # First forward pass with synthetic exemplars:
                model_output = self.model(
                    nested_tensor_from_tensor_list(input_image),
                    nested_tensor_from_tensor_list(pos_exemplar_image),
                    [pos_exemplars.to(device)],
                    [nested_tensor_from_tensor_list(neg_exemplar_image)],
                    [[neg_exemplars.to(device)]], # One level of list for number of negative images and another level for batch size
                    captions=[pos_text + " . " + neg_text + " ."],
                )
                
                # Second forward pass with self-generated exemplars:
                pos_pseudo_exemplars, neg_pseudo_exemplars = get_pseudo_exemplars(model_output, (input_image.size()[-2], input_image.size()[-1]), self.CONF_THRESH)
    
                model_output = self.model(
                    nested_tensor_from_tensor_list(input_image),
                    nested_tensor_from_tensor_list(input_image),
                    [pos_pseudo_exemplars],
                    [nested_tensor_from_tensor_list(input_image)],
                    [[neg_pseudo_exemplars]], # One level of list for number of negative images and another level for batch size
                    captions=[pos_text + " . " + neg_text + " ."],
                )
    
            input_ids = model_output["input_ids"][0]
            logits = model_output["pred_logits"].sigmoid()[0][:, :]
            boxes = model_output["pred_boxes"][0]
    
            # [pos_neg_split_idx] is the index of the first occurence of the "." separating token.
            for idx in range(len(input_ids)):
                token = input_ids[idx]
                if token == 1012:
                    pos_neg_split_idx = idx
                    break
    
            pos_logits = logits[:, :(pos_neg_split_idx + 1)]
            neg_logits = logits[:, (pos_neg_split_idx + 1):]
        
            # Stage 1 filtering:
            box_mask = pos_logits.max(dim=-1).values > self.CONF_THRESH
            boxes = boxes[box_mask, :]
            logits = logits[box_mask, :]
        
            # Stage 2 filtering:
            pos_logits = pos_logits[box_mask, :]
            neg_logits = neg_logits[box_mask, :]
            box_mask = pos_logits.max(dim=-1).values > neg_logits.max(dim=-1).values
            boxes = boxes[box_mask, :].cpu().numpy()
            logits = logits[box_mask, :].cpu().numpy().max(axis=-1)
        else:
            with torch.no_grad():
                
                # First forward pass with synthetic exemplars:
                model_output = self.model(
                    nested_tensor_from_tensor_list(input_image),
                    nested_tensor_from_tensor_list(pos_exemplar_image),
                    [pos_exemplars.to(device)],
                    [],
                    [], 
                    captions=[pos_text + " ."],
                )

                
                # Second forward pass with self-generated exemplars:
                pos_pseudo_exemplars, neg_pseudo_exemplars = get_pseudo_exemplars(model_output, (input_image.size()[-2], input_image.size()[-1]), self.CONF_THRESH)
    
                model_output = self.model(
                    nested_tensor_from_tensor_list(input_image),
                    nested_tensor_from_tensor_list(input_image),
                    [pos_pseudo_exemplars],
                    [],
                    [], 
                    captions=[pos_text + " ."],
                )
    
            input_ids = model_output["input_ids"][0]
            logits = model_output["pred_logits"].sigmoid()[0][:, :]
            boxes = model_output["pred_boxes"][0]
    
            # Stage 1 filtering:
            box_mask = logits.max(dim=-1).values > self.CONF_THRESH
            boxes = boxes[box_mask, :].cpu().numpy()
            logits = logits[box_mask, :].cpu().numpy()
        
        # Plot results.
        (w, h) = img.size
        det_map = np.zeros((h, w))
        det_map[(h * boxes[:, 1]).astype(int), (w * boxes[:, 0]).astype(int)] = 1

        pred_cnt = logits.shape[0]
        density_map_tensor = torch.from_numpy(det_map)
        return pred_cnt, density_map_tensor
