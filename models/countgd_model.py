import torch
import random
import os
cwd = os.path.dirname(__file__)

import sys

if os.path.join(cwd, "CountGD") not in sys.path:
    sys.path.append(os.path.join(cwd, "CountGD"))

from PIL import Image, ImageDraw, ImageFont

import torchvision.transforms.functional as F
import numpy as np
import argparse

from .CountGD.util.slconfig import SLConfig, DictAction
from .CountGD.util.misc import nested_tensor_from_tensor_list
from .CountGD.datasets import transforms as T

#import scipy.ndimage as ndimage
#import matplotlib.pyplot as plt

from base_model import BaseModel



#CONF_THRESH = 0.23

# MODEL:
def get_args_parser():
    """
    Example eval command:
    >> python main.py --output_dir ./gdino_test -c config/cfg_fsc147_vit_b_test.py --eval --datasets config/datasets_fsc147.json --pretrain_model_path ../checkpoints_and_logs/gdino_train/checkpoint_best_regular.pth --options text_encoder_type=checkpoints/bert-base-uncased --sam_tt_norm --crop
    """
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
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

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# Get counting model.
def build_model_and_transforms(args):
    
    #scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    #max_size=1333

    cwd = os.path.dirname(__file__)
    cfg = SLConfig.fromfile(os.path.join(cwd, "CountGD/cfg_app.py"))
    text_encoder_type = os.path.abspath(os.path.join(cwd, "CountGD/checkpoints/bert-base-uncased"))
    cfg.merge_from_dict({"text_encoder_type": text_encoder_type})
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
    from .CountGD.models.registry import MODULE_BUILD_FUNCS

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

import inflect

class CountGDModel(BaseModel):
    
    #model_ckpt=os.path.join(cwd, "CountGD/checkpoint_best_regular.pth")
    def __init__(self, img_directory, split_images, split_classes, model_ckpt=os.path.join(cwd, "CountGD/checkpoints/checkpoint_fsc147_best.pth"), device=None, singularize:bool=True):
        super().__init__(img_directory, split_images, split_classes)
        
        parser = argparse.ArgumentParser("CountGD", parents=[get_args_parser()])
        
        #parser.add_argument("--f", type=str, help="Workaround for ipynb")

        args = parser.parse_args([])
        args.pretrain_model_path = model_ckpt
        if device is None:
            device = get_device()
        self.model, self.transform = build_model_and_transforms(args)
        self.model = self.model.to(device)
        self.TEXT_THRESH = getattr(args, "text_threshold", 0.0)#args.get("text_threshold", 0.0)
        self.CONF_THRESH = getattr(args, "box_threshold", 0.23)#args.get("box_threshold", 0.23)
        self.model_name = "CountGD"#"CountGD_checkpoint_fsc147_best"
        
        self.singularize = singularize
        if self.singularize:
            self.p = inflect.engine()
        
    def get_text_prompt(self, text):
        """
        Implement the specific prompt retrieval logic for the CountGD model.
        """
        if self.singularize:
            singularized = self.p.singular_noun(text)
            ret = singularized if singularized and text not in ["sunglasses"] else text
            return ret
        else:
            return text
        
    def infer(self, img, text):
        """
        Implement the specific inference logic for the CountGD model.
        """
        
        keywords = "" # do not handle this for now
        
        # Handle no prompt case.
        #if prompts is None:
        #    prompts = {"image": img, "points": [[0,0,0,0]]}

        input_image, _ = self.transform(img, {"exemplars": torch.tensor([[0,0,0,0]])})
        device = next(self.model.parameters()).device
        input_image = input_image.unsqueeze(0).to(device)

        #exemplars = get_box_inputs(prompts["points"])
        
        #if len(exemplars) == 0:
        #    exemplars.append([[0,0,0,0]])

        #input_image_exemplars, exemplars = transform(prompts["image"], {"exemplars": torch.tensor(exemplars)})
        #input_image_exemplars = input_image_exemplars.unsqueeze(0).to(device)
        #exemplars = [exemplars["exemplars"].to(device)]
        
        with torch.no_grad():
            model_output = self.model(
                    nested_tensor_from_tensor_list(input_image),
                    torch.Tensor([[]]), #nested_tensor_from_tensor_list(input_image_exemplars),
                    torch.Tensor([[]]), #exemplars,
                    None, #[torch.tensor([0]).to(device) for _ in range(len(input_image))],
                    captions=[text + " ."] * len(input_image),
                )
        
        ind_to_filter = get_ind_to_filter(text, model_output["token"][0].word_ids, keywords)
        logits = model_output["pred_logits"].sigmoid()[0][:, ind_to_filter]
        boxes = model_output["pred_boxes"][0]
        
        #if len(keywords.strip()) > 0:
        #    box_mask = (logits > self.CONF_THRESH).sum(dim=-1) == len(ind_to_filter)
        #else:
        #    box_mask = logits.max(dim=-1).values > self.CONF_THRESH
        #logits = logits[box_mask, :].cpu().numpy()
        #boxes = boxes[box_mask, :].cpu().numpy()

        # BEGIN NEW VERSION
        # Step 1: Filter based on box confidence threshold.
        box_mask = logits.max(dim=-1).values > self.CONF_THRESH
        logits = logits[box_mask, :]
        boxes = boxes[box_mask, :]

        # Step 2: Filter based on text alignment threshold.
        text_mask = (logits > self.TEXT_THRESH).sum(dim=-1) == len(ind_to_filter)
        logits = logits[text_mask, :].cpu().numpy()
        boxes = boxes[text_mask, :].cpu().numpy()
        # END NEW VERSION
        

        
        # Plot results.
        (w, h) = img.size
        det_map = np.zeros((h, w))
        det_map[(h * boxes[:, 1]).astype(int), (w * boxes[:, 0]).astype(int)] = 1
        
        #det_map = ndimage.gaussian_filter(
        #    det_map, sigma=(w // 200, w // 200), order=0
        #)
        #
        #plt.imshow(img)
        #plt.imshow(det_map[None, :].transpose(1, 2, 0), 'jet', interpolation='none', alpha=0.7)
        #plt.axis('off')
        #img_buf = io.BytesIO()
        #plt.savefig(img_buf, format='png', bbox_inches='tight')
        #plt.close()

        #output_img = Image.open(img_buf)
        pred_cnt = logits.shape[0]
        density_map_tensor = torch.from_numpy(det_map)
        return pred_cnt, density_map_tensor