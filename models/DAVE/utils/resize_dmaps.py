import os
import torch
import numpy as np
from torchvision import transforms as T
import cv2


SRC_DIR = "/mnt/Datasets/few-shot-counting/FSC147/dmaps"
DST_DIR = "/mnt/Datasets/few-shot-counting/FSC147/gt_density_map_adaptive_512_512_object_VarV2"


def main():
    for dmap in os.listdir(SRC_DIR):
        src_dmap_path = os.path.join(SRC_DIR, dmap)
        dst_dmap_path = os.path.join(DST_DIR, dmap)
        
        np_density_map = np.load(src_dmap_path)
        #density_map = torch.from_numpy(np_density_map).unsqueeze(0)
        
        orig_num_obj = np.sum(np_density_map)
        
        #resize_den = T.Resize((512, 512), interpolation=T.InterpolationMode.NEAREST)
        #den_map_resized = T.Compose([
        #    resize_den,
        #])(density_map)
        
        np_dmap_resized = cv2.resize(np_density_map, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
        
        if np.sum(np_dmap_resized) != 0:
            np_dmap_resized = orig_num_obj*np_dmap_resized/np_dmap_resized.sum()
            
        #np_dmap_resized = den_map_resized.detach().cpu().numpy()[:,:,-1]
        np.save(dst_dmap_path, np_dmap_resized) # save


if __name__ == '__main__':
    main()