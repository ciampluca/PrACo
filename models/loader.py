
IMPLEMENTED_MODELS = ['CounTX', 'CLIP-Count', 'TFPOC', 'VLCounter', 'DAVE', 'ZSC', 'PseCo', 'GroundingREC', 'CountGD', 'FixedPointPromptCounting']

MULTICLASS_IMPLEMENTED_MODELS = IMPLEMENTED_MODELS.copy()

def load_model(model_name, img_directory, split_images, split_classes, load_filtered_checkpoints=False, device=None):
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
            return CLIPCountModel(img_directory, split_images, split_classes, device=device)
    elif model_name == 'TFPOC':
        from models.TFPOC_model import ClipSAMModel
        if load_filtered_checkpoints:
            filtered_checkpoint = os.path.join(dirname_file, "TFPOC/pretrain/fsc_filtered_sam_vit_b_01ec64.pth")
            return ClipSAMModel(img_directory, split_images, split_classes, sam_checkpoint=filtered_checkpoint, device=device)
        else:
            return ClipSAMModel(img_directory, split_images, split_classes)
    elif model_name == 'VLCounter':
        from models.vlcounter_model import VLCounterModel
        if load_filtered_checkpoints:
            raise NotImplementedError("Filtered checkpoint loading not implemented for VLCounter.")
        else:
            return VLCounterModel(img_directory, split_images, split_classes)
    elif model_name == 'DAVE':
        from models.dave_model import DAVEModel
        return DAVEModel(img_directory, split_images, split_classes)
    elif model_name == 'ZSC':
        from models.ZSC_model import ZSCModel
        if load_filtered_checkpoints:
            raise NotImplementedError("Filtered checkpoint loading not implemented for ZSC.")
        else:
            return ZSCModel(img_directory, split_images, split_classes)
    elif model_name == 'PseCo':
        from models.PseCo_model import PseCoModel
        if load_filtered_checkpoints:
            raise NotImplementedError("Filtered checkpoint loading not implemented for PseCo.")
        else:
            return PseCoModel(img_directory, split_images, split_classes)
    elif model_name == 'GroundingREC':
        from models.GroundingREC_model import GroundingRECModel
        if load_filtered_checkpoints:
            raise NotImplementedError("Filtered checkpoint loading not implemented for GroundingREC.")
        else:
            return GroundingRECModel(img_directory, split_images, split_classes)
    elif model_name == 'CountGD':
        from models.countgd_model import CountGDModel
        if load_filtered_checkpoints:
            
            filtered_checkpoint = os.path.join(dirname_file, "CountGD/checkpoints/checkpoint_best_regular_fsc147_filtered.pth")
            return CountGDModel(img_directory, split_images, split_classes, model_ckpt=filtered_checkpoint)
        else:
            return CountGDModel(img_directory, split_images, split_classes)
    elif model_name == 'FixedPointPromptCounting':
        from models.fixedpointpromptcounting_model import FixedPointPromptCountingModel
        if load_filtered_checkpoints:
            filtered_checkpoint = os.path.join(dirname_file, "FixedPointPromptCounting/fxp_filtered.pth")
            return FixedPointPromptCountingModel(img_directory, split_images, split_classes, checkpoint_path=filtered_checkpoint, device=device)
        else:
            return FixedPointPromptCountingModel(img_directory, split_images, split_classes, device=device)
    else:
        raise ValueError(f"Model {model_name} is not implemented.")