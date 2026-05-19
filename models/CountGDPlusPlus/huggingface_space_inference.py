#import copy
import random
import torch
#import PIL
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as F
import numpy as np
import argparse
#import json
#import plotly.express as px
#import pandas as pd
from util.slconfig import SLConfig, DictAction
from util.misc import nested_tensor_from_tensor_list
import datasets.transforms as T
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
# https://github.com/PhyscalX/gradio-image-prompter/tree/main/backend/gradio_image_prompter/templates/component
import io
from enum import Enum
import os
#import subprocess
#from subprocess import call
#import shlex
#import shutil
#os.environ["GRADIO_TEMP_DIR"] = os.path.join(os.getcwd(), "tmp")
cwd = os.getcwd()


class AppSteps(Enum):
    JUST_TEXT = 1
    TEXT_AND_EXEMPLARS = 2
    JUST_EXEMPLARS = 3
    FULL_APP = 4

CONF_THRESH = 0.23

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
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    data_transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            normalize,
        ]
    )
    cfg = SLConfig.fromfile("cfg_app.py")
    cfg.merge_from_dict({"text_encoder_type": "checkpoints/bert-base-uncased"})
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, _, _ = build_func(args)

    checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")["model"]
    model.load_state_dict(checkpoint, strict=False)

    model.eval()

    return model, data_transform


parser = argparse.ArgumentParser("Counting Application", parents=[get_args_parser()])
parser.add_argument("--f", type=str, help="L'ho messo per fare andare ipynb")
args = parser.parse_args()
device = get_device()
model, transform = build_model_and_transforms(args)
model = model.to(device)

# examples = [
#     ["strawberry.jpg", "strawberry", {"image": "strawberry.jpg"}],
#     ["strawberry.jpg", "blueberry", {"image": "strawberry.jpg"}],
#     ["bird-1.JPG", "bird", {"image": "bird-2.JPG"}],
#     ["fish.jpg", "fish", {"image": "fish.jpg"}],
#     ["women.jpg", "girl", {"image": "women.jpg"}],
#     ["women.jpg", "boy", {"image": "women.jpg"}],
#     ["balloon.jpg", "hot air balloon", {"image": "balloon.jpg"}],
#     ["deer.jpg", "deer", {"image": "deer.jpg"}],
#     ["apple.jpg", "apple", {"image": "apple.jpg"}],
#     ["egg.jpg", "egg", {"image": "egg.jpg"}],
#     ["stamp.jpg", "stamp", {"image": "stamp.jpg"}],
#     ["green-pea.jpg", "green pea", {"image": "green-pea.jpg"}],
#     ["lego.jpg", "lego", {"image": "lego.jpg"}]
# ]

# APP:
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

def count(image, text, prompts, state, device):

    keywords = "" # do not handle this for now
    
    # Handle no prompt case.
    if prompts is None:
        prompts = {"image": image, "points": []}
    input_image, _ = transform(image, {"exemplars": torch.tensor([])})
    input_image = input_image.unsqueeze(0).to(device)
    exemplars = get_box_inputs(prompts["points"])

    if len(exemplars) == 0:
        exemplars.append([0,0,0,0])
    
    input_image_exemplars, exemplars = transform(prompts["image"], {"exemplars": torch.tensor(exemplars)})
    input_image_exemplars = input_image_exemplars.unsqueeze(0).to(device)
    exemplars = [exemplars["exemplars"].to(device)]

    with torch.no_grad():
        model_output = model(
                nested_tensor_from_tensor_list(input_image),
                nested_tensor_from_tensor_list(input_image_exemplars),
                exemplars,
                [torch.tensor([0]).to(device) for _ in range(len(input_image))],
                captions=[text + " ."] * len(input_image),
            )
    
    ind_to_filter = get_ind_to_filter(text, model_output["token"][0].word_ids, keywords)
    logits = model_output["pred_logits"].sigmoid()[0][:, ind_to_filter]
    boxes = model_output["pred_boxes"][0]
    if len(keywords.strip()) > 0:
        box_mask = (logits > CONF_THRESH).sum(dim=-1) == len(ind_to_filter)
    else:
        box_mask = logits.max(dim=-1).values > CONF_THRESH
    logits = logits[box_mask, :].cpu().numpy()
    boxes = boxes[box_mask, :].cpu().numpy()
    
    # Plot results.
    (w, h) = image.size
    det_map = np.zeros((h, w))
    det_map[(h * boxes[:, 1]).astype(int), (w * boxes[:, 0]).astype(int)] = 1
    det_map = ndimage.gaussian_filter(
        det_map, sigma=(w // 200, w // 200), order=0
    )
    plt.imshow(image)
    plt.imshow(det_map[None, :].transpose(1, 2, 0), 'jet', interpolation='none', alpha=0.7)
    plt.axis('off')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    plt.close()

    output_img = Image.open(img_buf)

    out_label = "Detected instances predicted with"
    if len(text.strip()) > 0:
        out_label += " text"
        if exemplars[0].size()[0] == 1:
            out_label += " and " + str(exemplars[0].size()[0]) + " visual exemplar."        
        elif exemplars[0].size()[0] > 1:
            out_label += " and " + str(exemplars[0].size()[0]) + " visual exemplars."
        else:
            out_label += "."
    elif exemplars[0].size()[0] > 0:
        if exemplars[0].size()[0] == 1:
            out_label += " " + str(exemplars[0].size()[0]) + " visual exemplar."
        else:
            out_label += " " + str(exemplars[0].size()[0]) + " visual exemplars."
    else:
        out_label = "Nothing specified to detect."
    
    return output_img, boxes.shape[0]
    #return (gr.Image(output_img, visible=True, label=out_label, show_label=True), gr.Number(label="Predicted Count", visible=True, value=boxes.shape[0]), new_submit_btn, gr.Tab(visible=True), step_3, state)

def count_main(image, text, prompts, device):
    keywords = "" # do not handle this for now
    # Handle no prompt case.
    if prompts is None:
        prompts = {"image": image, "points": [[0,0,0,0]]}
    input_image, _ = transform(image, {"exemplars": torch.tensor([[0,0,0,0]])})
    input_image = input_image.unsqueeze(0).to(device)
    #exemplars = get_box_inputs(prompts["points"])
    
    #if len(exemplars) == 0:
    #    exemplars.append([[0,0,0,0]])

    #input_image_exemplars, exemplars = transform(prompts["image"], {"exemplars": torch.tensor(exemplars)})
    #input_image_exemplars = input_image_exemplars.unsqueeze(0).to(device)
    #exemplars = [exemplars["exemplars"].to(device)]
    
    with torch.no_grad():
        model_output = model(
                nested_tensor_from_tensor_list(input_image),
                torch.Tensor([[]]), #nested_tensor_from_tensor_list(input_image_exemplars),
                torch.Tensor([[]]), #exemplars,
                None, #[torch.tensor([0]).to(device) for _ in range(len(input_image))],
                captions=[text + " ."] * len(input_image),
            )
    
    ind_to_filter = get_ind_to_filter(text, model_output["token"][0].word_ids, keywords)
    logits = model_output["pred_logits"].sigmoid()[0][:, ind_to_filter]
    boxes = model_output["pred_boxes"][0]
    if len(keywords.strip()) > 0:
        box_mask = (logits > CONF_THRESH).sum(dim=-1) == len(ind_to_filter)
    else:
        box_mask = logits.max(dim=-1).values > CONF_THRESH
    logits = logits[box_mask, :].cpu().numpy()
    boxes = boxes[box_mask, :].cpu().numpy()
    
    # Plot results.
    (w, h) = image.size
    det_map = np.zeros((h, w))
    det_map[(h * boxes[:, 1]).astype(int), (w * boxes[:, 0]).astype(int)] = 1
    det_map = ndimage.gaussian_filter(
        det_map, sigma=(w // 200, w // 200), order=0
    )
    plt.imshow(image)
    plt.imshow(det_map[None, :].transpose(1, 2, 0), 'jet', interpolation='none', alpha=0.7)
    plt.axis('off')
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    plt.close()

    output_img = Image.open(img_buf)

    return output_img, boxes.shape[0]
