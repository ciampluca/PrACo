import torch
from torchvision import transforms
from .base_model import BaseModel
from torch.nn import DataParallel
from dotmap import DotMap

from .DAVE.models.dave import build_model
from .DAVE.utils.data import pad_image


SCALE_FACTOR = 60


class DAVEModel(BaseModel):
    def __init__(self, img_directory, split_images, split_classes, model_ckpt='models/DAVE/pretrained/DAVE_0_shot.pth', feat_comp_ckpt="models/DAVE/pretrained/verification.pth", device = "cuda:0"):
        super().__init__(img_directory, split_images, split_classes)

        torch.cuda.set_device(device)
        gpu = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{gpu}")

        args = DotMap()
        args.image_size = 512
        args.num_enc_layers = 3
        args.num_dec_layers = 3
        args.num_objects = 3
        args.zero_shot = True
        args.emb_dim = 256
        args.num_heads = 8
        args.kernel_dim = 3
        args.backbone = 'resnet50'
        args.swav_backbone = True
        args.train_backbone = False
        args.reduction = 8
        args.dropout = 0.1
        args.pre_norm = True
        args.use_query_pos_emb = True
        args.use_objectness = True
        args.use_appearance = True
        args.d_s = 1.0
        args.m_s = 0.0
        args.i_thr = 0.55
        args.d_t = 3
        args.s_t = 0.002
        args.norm_s = False
        args.egv = 0.1  # 0.17
        args.prompt_shot = True
        args.det_train = False   
        args.layer_norm_eps = 0.00001
        args.mlp_factor = 8
        args.norm = True
        args.backbone_lr = 0


        self.model_name = "DAVE"

        self.model = DataParallel(
            build_model(args).to(self.device),
            device_ids=[gpu],
            output_device=gpu
        )


        self.model.load_state_dict(
            torch.load(model_ckpt)['model'], strict=False
        )

        pretrained_dict_box = {k.split("box_predictor.")[1]: v for k, v in
                            torch.load(model_ckpt)[
                                'model'].items() if 'box_predictor' in k}
        self.model.module.box_predictor.load_state_dict(pretrained_dict_box)

        pretrained_dict_feat = {k.split("feat_comp.")[1]: v for k, v in
                                torch.load(feat_comp_ckpt)[
                                    'model'].items() if 'feat_comp' in k}
        self.model.module.feat_comp.load_state_dict(pretrained_dict_feat)

        backbone_params = dict()
        non_backbone_params = dict()
        fcos_params = dict()
        feat_comp = dict()
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if 'backbone' in n:
                backbone_params[n] = p
                backbone_params[n].requires_grad = False
            elif 'box_predictor' in n:
                fcos_params[n] = p
            elif 'feat_comp' in n:
                feat_comp[n] = p
                feat_comp[n].requires_grad = False
            else:
                non_backbone_params[n] = p
                non_backbone_params[n].requires_grad = False

        self.model.eval()

        self.resized_img_size = 512
        self.img_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.resized_img_size, self.resized_img_size), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def get_text_prompt(self, text):
        """
        Implement the specific prompt retrieval logic for the DAVE model.
        """
        return f"{text}"


    def infer(self, img, text, text_positive=None, resized_img_size=512, save_gpu_memory=False, clip_threshold=0.85, take_min_score=False, per_cluster_thresh=False, return_bboxes=False):
        """
        Implement the specific inference logic for the DAVE model.
        - img: PIL Image
        - text: string (class prompt)
        - text_positive: string or list of strings (positive class prompts)
        - resized_img_size: int (size to which the image is resized)
        - save_gpu_memory: bool (whether to use memory-saving techniques)
        - clip_threshold: float (threshold for CLIP filtering)
        - take_min_score: bool (whether to use the minimum score among positive classes)
        - per_cluster_thresh: bool (whether to use per-cluster thresholds)

        CLIP threshold is used to filter out low-confidence detections.
        If you provide a higher threshold, only high-confidence detections will be considered.
        In the case of multiple positive classes, the highest confidence among them is used.
        """
        w, h = img.size
        img = self.img_trans(img)
        #print(f"DEBUG - Original size: ({w}, {h}), Transformed size: {img.shape[1:]}")
        img = img.unsqueeze(0)
        img = img.to(self.device)
        bboxes = None
        img_name = None

        # print('-----------WARNING PARTS-----------')
        # for name, param in self.model.named_parameters():
        #     if param.device != torch.device('cuda:0'):
        #         print(f'Parameter {name} is on {param.device}')

        # for name, buffer in self.model.named_buffers():
        #     if buffer.device != torch.device('cuda:0'):
        #         print(f'Buffer {name} is on {buffer.device}')
        if any([p.device != self.device for p in self.model.parameters()]):
            print(f'WARNING: Model parameters are not on the correct device. Moving to {self.device}')
            # print which parameters are on the wrong device
            for name, param in self.model.named_parameters():
                if param.device != self.device:
                    print(f'Parameter {name} is on {param.device}')
            self.model.to(self.device)

        text_positive_list = ([text_positive] if isinstance(text_positive, str) else text_positive) if text_positive is not None else None

        try:
            with torch.no_grad():
                out, aux, tblr, boxes_pred = self.model(img, bboxes, img_name, classes=[text], positive_classes=text_positive_list, save_gpu_memory=save_gpu_memory, clip_threshold=clip_threshold, take_min_score=take_min_score, per_cluster_thresh=per_cluster_thresh)
        except (IndexError, RuntimeError) as e:
            # Handle edge cases where DAVE fails (e.g., no detections, malformed boxes)
            print(f"DAVE inference error: {e}. Returning zero count.")
            # Return zero count and empty density map
            density_map_tensor = torch.zeros((self.resized_img_size, self.resized_img_size))
            return 0.0, density_map_tensor

        boxes_predicted = boxes_pred.box
        scale_y = min(1, 50 / (boxes_predicted[:, 2] - boxes_predicted[:, 0]).mean())
        scale_x = min(1, 50 / (boxes_predicted[:, 3] - boxes_predicted[:, 1]).mean())

        if scale_x < 1 or scale_y < 1:
            scale_x = (int(resized_img_size * scale_x) // 8 * 8) / resized_img_size
            scale_y = (int(resized_img_size * scale_y) // 8 * 8) / resized_img_size
            resize_ = transforms.Resize((int(resized_img_size * scale_x), int(resized_img_size * scale_y)), antialias=True)
            img_resized = resize_(img)

            shape = img_resized.shape[1:]
            img_resized = pad_image(img_resized[0]).unsqueeze(0)

        else:
            scale_y = max(1, 11 / (boxes_predicted[:, 2] - boxes_predicted[:, 0]).mean())
            scale_x = max(1, 11 / (boxes_predicted[:, 3] - boxes_predicted[:, 1]).mean())

            if scale_y > 1.9:
                scale_y = 1.9
            if scale_x > 1.9:
                scale_x = 1.9

            scale_x = (int(resized_img_size * scale_x) // 8 * 8) / resized_img_size
            scale_y = (int(resized_img_size * scale_y) // 8 * 8) / resized_img_size
            resize_ = transforms.Resize((int(resized_img_size * scale_x), int(resized_img_size * scale_y)), antialias=True)
            img_resized = resize_(img)
            shape = img_resized.shape[1:]
        if scale_x != 1.0 or scale_y != 1.0:
            try:
                with torch.no_grad():
                    out, aux, tblr, boxes_pred = self.model(img_resized, bboxes, img_name, classes=[text], positive_classes=text_positive_list, save_gpu_memory=save_gpu_memory, clip_threshold=clip_threshold, take_min_score=take_min_score, per_cluster_thresh=per_cluster_thresh)
                    boxes_predicted = boxes_pred.box
            except (IndexError, RuntimeError) as e:
                # Handle edge cases where DAVE fails on rescaled image
                print(f"DAVE inference error on rescaled image: {e}. Using original inference result.")
                # Continue with the original 'out' from the first inference

        ## DAVE can produce negative density values for negative detections
        ## Only count positive values (objects belonging to the query class)
        #out_clamped = torch.clamp(out, min=0.0)
        
        pred_cnt = torch.sum(out).item()

        out = out.cpu()
        density_map_tensor = out.squeeze()

        # torch.cuda.empty_cache()
        print(f"DEBUG count: {pred_cnt} len(boxes_pred): {len(boxes_predicted)} text: {text} -- text_positive: {text_positive}")

        if return_bboxes:
            return pred_cnt, density_map_tensor, boxes_predicted.cpu()
        return pred_cnt, density_map_tensor
