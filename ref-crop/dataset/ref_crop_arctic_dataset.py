import cv2
import os
import torch
import json
import os.path as op
import math

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset

import src.datasets.dataset_utils as dataset_utils
from common.data_utils import read_img
import common.data_utils as data_utils
from .arctic_dataset import ArcticDataset
from src.datasets.dataset_utils import get_valid, pad_jts2d

MAX_ENCODE_POWER = 3
BOX_POSI_VEC_LEN = 4 * 2 * (2 * (MAX_ENCODE_POWER + 1)) # 4points, 2thta, 4*2sin/cos

def calc_point_posi_vec(center:tuple[float, float], point:tuple[float, float], focal_length:float):
    def calc_theta(c_x, p_x):
        return math.atan((p_x-c_x) / focal_length)
    
    def encode_theta(theta:float):
        # encoding(θx)=[sin(πθx),cos(πθx),sin(2πθx),cos(2πθx),sin(4πθx),cos(4πθx),sin(8πθx),cos(8πθx)]
        # theta = pi*theta
        theta *= math.pi
        scale = 1
        encoded:list[float] = []
        for i in range(MAX_ENCODE_POWER+1):
            encoded += [math.sin(scale*theta), math.cos(scale*theta)]
            scale *= 2
        return encoded
    
    posi_vec = encode_theta(calc_theta(center[0], point[0])) + encode_theta(calc_theta(center[1], point[1]))
    return posi_vec


def calc_box_posi_vec(center:tuple[float, float], focal_length, x, y, w, h):
    corners = [
        (x, y),
        (x+w, y),
        (x+w, y+h),
        (x, y+h),
    ]
    posi_vec = []
    for point in corners:
        posi_vec += calc_point_posi_vec(center, point, focal_length)
    return torch.tensor(posi_vec)


class RefCropArcticDataset(ArcticDataset):
    target_size = (224, 224)

    def plain_process_rgb(self, rgb_img):
        args = self.args
        # augment parameters
        augm_dict = data_utils.augm_params(
            self.aug_data,
            args.flip_prob,
            args.noise_factor,
            args.rot_factor,
            args.scale_factor,
        )
        pn = augm_dict["pn"]
        rgb_img[:, :, 0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 0] * pn[0]))
        rgb_img[:, :, 1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 1] * pn[1]))
        rgb_img[:, :, 2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:, :, 2] * pn[2]))
        rgb_img = np.transpose(rgb_img.astype("float32"), (2, 0, 1)) / 255.0
        return rgb_img

    def _load_ref_img_and_posi_vec(self, imgname, inputs):
        # WARN modify dict inputs directly
        if not self.ref_online: # offline
            img_basename = os.path.basename(imgname).split(".")[0]
            ref_img_name = f"{img_basename}.png"
            seq_name = "_".join(imgname.split("/")[-4:-1])
            modes = ["gt_mesh_l", "gt_mesh_r", "gt_mesh_obj"]
            # sub_types = ["crop_image"]#, "crop_mask"]

            for mode, mode_alias in zip(modes, ["l", "r", "o"]):
                ref_img_path = os.path.join(self.reference_exp_dir, seq_name, mode, "images", "crop_image", ref_img_name)
                ref_img, img_status = read_img(ref_img_path, self.target_size + (3,), nonexsit_ok=True)
                img_w, img_h = ref_img.shape[:2]
                default_principle_point = (img_w / 2, img_h / 2)
                box = self.position_info[seq_name].get(ref_img_name) # xywh
                if box is not None:
                    x, y, w, h = self.position_info[seq_name][ref_img_name] # xywh
                else:
                    x, y = default_principle_point
                    w, h = 0, 0
                    # posi_vec as [0,1,0,1,...]
                posi_vec = calc_box_posi_vec(default_principle_point, self.focal_length, x, y, w, h)
                ref_img = cv2.resize(ref_img, self.target_size)
                ref_img = self.plain_process_rgb(ref_img)
                ref_img = torch.from_numpy(ref_img).float()
                inputs[f"ref_img_{mode_alias}_rgb"] = ref_img # ref_img_l_rgb
                inputs[f"ref_img_{mode_alias}_posi_vec"] = posi_vec
                # for sub_type, sub_type_alias in zip(sub_types, ["rgb", "mask"]):
                #     ref_img_path = os.path.join(self.reference_exp_dir, seq_name, mode, "images", sub_type, ref_img_name)
                #     ref_img, img_status = read_img(ref_img_path, self.target_size + (3,), nonexsit_ok=True)
                #     ref_img = cv2.resize(ref_img, self.target_size)
                #     ref_img = self.plain_process_rgb(ref_img)
                #     ref_img = torch.from_numpy(ref_img).float()
                #     inputs[f"ref_img_{mode_alias}_{sub_type_alias}"] = ref_img # ref_img_l_rgb
        else:
            raise NotImplementedError()
        return inputs

    def getitem(self, imgname, load_rgb=True):
        inputs, targets, meta_info = super().getitem(imgname, load_rgb=True)
        inputs = self._load_ref_img_and_posi_vec(imgname, inputs)
        return inputs, targets, meta_info
    
    def _preload_position_info(self):
        #  TODO position info改为rgb和mask共享(if mask needed)
        self.position_info = {}
        for seq in os.listdir(self.reference_exp_dir):
            for mode in ["gt_mesh_l", "gt_mesh_r", "gt_mesh_obj"]:
                imgs_path = os.path.join(self.reference_exp_dir, seq, mode, "images", "crop_image")
                with open(os.path.join(imgs_path, "position_info.json")) as f:
                    self.position_info[seq] = json.load(f)


    def __init__(self, args, split, seq=None):
        super().__init__(args, split, seq=seq)
        self.reference_exp_dir = args.ref_crop_folder # logs/3558f1342/render
        self.focal_length = 1000 # TODO show as render info
        if args.ref_mode == "online":
            self.ref_online = True
            raise NotImplementedError()
        else:
            self.ref_online = False
            self._preload_position_info()