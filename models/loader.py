
IMPLEMENTED_MODELS = ['CounTX', 'CLIP-Count', 'TFPOC', 'VLCounter', 'DAVE', 'ZSC', 'PseCo', 'GroundingREC', 'CountGD', 'FixedPointPromptCounting']

MULTICLASS_IMPLEMENTED_MODELS = IMPLEMENTED_MODELS.copy()

def load_model(model_name, img_directory, split_images, split_classes, load_filtered_checkpoints=False, device=None, split='test'):
    """Load and return the specified model."""

    import os
    dirname_file = os.path.dirname(__file__)

    if model_name == 'CounTX':
        from models.countx_model import CounTXModel
        if load_filtered_checkpoints:
            filtered_checkpoint = os.path.join(dirname_file, "CounTX/checkpoints/countx_fsc147_filtered_checkpoint-1000.pth")
            return CounTXModel(img_directory, split_images, split_classes, model_ckpt=filtered_checkpoint, device=device)
        else:
            checkpoint = os.path.join(dirname_file, "pretrained_models/paper-model.pth")
            return CounTXModel(img_directory, split_images, split_classes, model_ckpt=checkpoint, device=device)
    elif model_name == 'CLIP-Count':
        from models.clipcount_model import CLIPCountModel
        if load_filtered_checkpoints:
            filtered_checkpoint = os.path.join(dirname_file, "CLIPCount/checkpoints/clipcount_fsc147_filtered.ckpt")
            return CLIPCountModel(img_directory, split_images, split_classes, model_ckpt=filtered_checkpoint, device=device)
        else:
            checkpoint = os.path.join(dirname_file, "CLIPCount/checkpoints/clipcount_pretrained.ckpt")
            return CLIPCountModel(img_directory, split_images, split_classes, model_ckpt=checkpoint, device=device)
    elif model_name == 'TFPOC':
        from models.TFPOC_model import ClipSAMModel
        if load_filtered_checkpoints:
            filtered_checkpoint = os.path.join(dirname_file, "TFPOC/pretrain/fsc_filtered_sam_vit_b_01ec64.pth")
            return ClipSAMModel(img_directory, split_images, split_classes, sam_checkpoint=filtered_checkpoint, device=device)
        else:
            checkpoint = os.path.join(dirname_file, "pretrained_models/sam_vit_b_01ec64.pth")
            return ClipSAMModel(img_directory, split_images, split_classes, sam_checkpoint=checkpoint, device=device)
    elif model_name == 'VLCounter':
        from models.vlcounter_model import VLCounterModel
        if load_filtered_checkpoints:
            filtered_checkpoint = os.path.join(dirname_file, "VLCounter/checkpoints/fsc_filtered_vlcounter_283_best.pth")
            return VLCounterModel(img_directory, split_images, split_classes, model_ckpt=filtered_checkpoint, device=device)
        else:
            checkpoint = os.path.join(dirname_file, "VLCounter/checkpoints/182_best.pth")
            return VLCounterModel(img_directory, split_images, split_classes, model_ckpt=checkpoint, device=device)
    elif model_name == 'DAVE':
        from models.dave_model import DAVEModel
        if load_filtered_checkpoints:
            filtered_checkpoint = os.path.join(dirname_file, "pretrained_models/DAVE_FSC_filtered_0_shot.pth")
            feat_comp_ckpt = os.path.join(dirname_file, "pretrained_models/DAVE_FSC_filtered_verification.pth")
            return DAVEModel(img_directory, split_images, split_classes, model_ckpt=filtered_checkpoint, feat_comp_ckpt=feat_comp_ckpt, device=device)
        else:
            return DAVEModel(img_directory, split_images, split_classes, device=device)
    elif model_name == 'ZSC':
        from models.ZSC_model import ZSCModel
        if load_filtered_checkpoints:
            filtered_checkpoint = os.path.join(dirname_file, "ZSC/checkpoints/ZSC_FSC_filtered_model_best.pth")
            #filtered_config = os.path.join(dirname_file, "ZSC/config/ZSC_FSC_filtered_config.yaml")
            filtered_config = os.path.join(dirname_file, "ZSC/config/test.yaml")
            print("Using public available regressor for ZSC-Count.")
            regressor_path = os.path.join(dirname_file, "ZSC/checkpoints/ZSC_public_checkpoint_regressor.pth")
            print("Using vae feats publicly available for ZSC-Count.")
            vae_feats_path = os.path.join(dirname_file, 'ZSC/checkpoints/bmnet+_ep3_epoch300_no_refiner/fsc_vae_feats.npy')
            import json
            classes_path = os.path.join(dirname_file, "../data/multiclass-dataset/multiclass_split_classes.json")
            classes_list = json.load(open(classes_path, 'r'))[split]
            print(f"Using classes list of length {len(classes_list)} for ZSC-Count.")

            return ZSCModel(img_directory, split_images, split_classes, model_ckpt=filtered_checkpoint, device=device, config=filtered_config, regressor_path=regressor_path, classes_list=classes_list, vae_feats_path=vae_feats_path)
        else:
            checkpoint = os.path.join(dirname_file, "ZSC/checkpoints/model_best_original_training.pth")
            config = os.path.join(dirname_file, "ZSC/config/test.yaml")
            regressor_path = os.path.join(dirname_file, "ZSC/checkpoints/ZSC_public_checkpoint_regressor.pth")
            vae_feats_path = os.path.join(dirname_file, 'ZSC/checkpoints/bmnet+_ep3_epoch300_no_refiner/fsc_vae_feats.npy')
            import json
            classes_path = os.path.join(dirname_file, "../data/Split_Classes_FSC147.json")
            classes_list = json.load(open(classes_path, 'r'))[split]
            return ZSCModel(img_directory, split_images, split_classes, model_ckpt=checkpoint, device=device, config=config, regressor_path=regressor_path, classes_list=classes_list, vae_feats_path=vae_feats_path)
    elif model_name == 'PseCo':
        from models.PseCo_model import PseCoModel
        if load_filtered_checkpoints:
            filtered_point_decoder_ckpt = os.path.join(dirname_file, "PseCo/checkpoints/PseCo_fsc_filtered_point_decoder_vith.pth")
            filtered_cls_head_ckpt = os.path.join(dirname_file, "PseCo/checkpoints/PseCo_fsc_filtered_cls_head-10000.tar")
            filtered_clip_text_prompt_ckpt = os.path.join(dirname_file, "PseCo/checkpoints/PseCo_fsc_filtered_clip_text_prompt_test_split.pth")
            return PseCoModel(img_directory, split_images, split_classes,
                              point_decoder_ckpt=filtered_point_decoder_ckpt,
                              cls_head_ckpt=filtered_cls_head_ckpt,
                              clip_text_prompt_ckpt=filtered_clip_text_prompt_ckpt, device=device)
        else:
            point_decoder_ckpt = os.path.join(dirname_file, "PseCo/checkpoints/point_decoder_vith.pth")
            cls_head_ckpt = os.path.join(dirname_file, "PseCo/checkpoints/MLP_small_box_w1_zeroshot.tar")
            clip_text_prompt_ckpt = os.path.join(dirname_file, "PseCo/checkpoints/clip_text_prompt.pth")
            return PseCoModel(img_directory, split_images, split_classes, point_decoder_ckpt=point_decoder_ckpt, cls_head_ckpt=cls_head_ckpt, clip_text_prompt_ckpt=clip_text_prompt_ckpt, device=device)
    elif model_name == 'GroundingREC':
        from models.GroundingREC_model import GroundingRECModel
        if load_filtered_checkpoints:
            filtered_checkpoint = os.path.join(dirname_file, "pretrained_models/GroundingREC_FSC_filtered_model.pth")
            return GroundingRECModel(img_directory, split_images, split_classes, model_ckpt=filtered_checkpoint, device=device)
        else:
            #checkpoint = os.path.join(dirname_file, "pretrained_models/groundingdino_swint_ogc.pth")
            checkpoint = os.path.join(dirname_file, "pretrained_models/GroundingREC_model_original_training_all_dataset.pth")
            return GroundingRECModel(img_directory, split_images, split_classes, model_ckpt=checkpoint, device=device)
    elif model_name == 'GroundingRECFSC':
        from models.GroundingREC_model import GroundingRECModel
        if load_filtered_checkpoints:
            raise NotImplementedError("Filtered checkpoints for GroundingRECFSC are not available.")
        checkpoint = os.path.join(dirname_file, "pretrained_models/GroundingREC_model_original_training_only_FSC.pth")
        print("Loading GroundingRECFSC with original training on FSC147 only.")
        return GroundingRECModel(img_directory, split_images, split_classes, model_ckpt=checkpoint, device=device)
    elif model_name == 'CountGD':
        from models.countgd_model import CountGDModel
        if load_filtered_checkpoints:
            
            filtered_checkpoint = os.path.join(dirname_file, "CountGD/checkpoints/checkpoint_best_regular_fsc147_filtered.pth")
            return CountGDModel(img_directory, split_images, split_classes, model_ckpt=filtered_checkpoint)
        else:
            return CountGDModel(img_directory, split_images, split_classes)
    elif model_name == 'CountGDPlusPlus':
        from models.countgdplusplus_model import CountGDPlusPlusModel
        if load_filtered_checkpoints:
            
            filtered_checkpoint = os.path.join(dirname_file, "CountGDPlusPlus/checkpoints/checkpoint_best_regular_fsc147_filtered.pth")
            return CountGDPlusPlusModel(img_directory, split_images, split_classes, model_ckpt=filtered_checkpoint)
        else:
            return CountGDPlusPlusModel(img_directory, split_images, split_classes)
    elif model_name == 'FixedPointPromptCounting':
        from models.fixedpointpromptcounting_model import FixedPointPromptCountingModel
        if load_filtered_checkpoints:
            filtered_checkpoint = os.path.join(dirname_file, "FixedPointPromptCounting/fxp_filtered.pth")
            return FixedPointPromptCountingModel(img_directory, split_images, split_classes, checkpoint_path=filtered_checkpoint, device=device)
        else:
            return FixedPointPromptCountingModel(img_directory, split_images, split_classes, device=device)
    else:
        raise ValueError(f"Model {model_name} is not implemented.")