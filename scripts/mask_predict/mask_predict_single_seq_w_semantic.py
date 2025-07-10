# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import cv2
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import warnings
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import natsort as ns
import open_clip

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
import warnings

from helpers import get_new_pallete, get_cropped_image, pad_into_square, save_mask_images
warnings.filterwarnings("ignore", category=UserWarning)


"""To segment a given RGB-D sequence, getting mask_maps, mask_color_maps, and save the visual embedding of each detected mask"""


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def load_clip(pretrained_path=None):
    print(f'[INFO] loading CLIP model...')

    if pretrained_path is None or not os.path.exists(pretrained_path):
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    else:
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=pretrained_path)  # load from local

    model.cuda()
    model.eval()
    print(f'[INFO]', ' finish loading CLIP model...')
    return model, preprocess


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="third_party/detectron2/projects/CropFormer/configs/entityv2/entity_segmentation/mask2former_hornet_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--seq_name",
        type=str
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/media/javens/igrape_8T/scannet",
    )
    parser.add_argument(
        "--image_path_pattern",
        type=str,
        default="frames/color/*",
    )
    parser.add_argument(
        "--seg_interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/javens/git_repos/remote_repo/OnlineAnySeg_v1/testing/",  # output dir
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="/home/javens/git_repos/MaskClustering/models/CLIP-ViT-H-14-laion2b_s32b_b79k/open_clip_pytorch_model.bin",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="scannet",
    )
    parser.add_argument(
        "--dst_h",
        type=int,
        default=480,
    )
    parser.add_argument(
        "--dst_w",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.6,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--min_mask_pixel_size",
        type=int,
        default=500,
        help="Minimum pixel of a valid 2D mask",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class ImageDataset(Dataset):
    def __init__(self, cfg, seq_name, root_dir, output_root, image_path_pattern, seg_interval, dst_h, dst_w, confidence_threshold):
        self.seq_name = seq_name
        self.seq_dir = os.path.join(root_dir, seq_name)  # root dir of this sequence
        self.output_seq_dir = os.path.join(output_root, seq_name)
        self.dst_h = dst_h
        self.dst_w = dst_w
        self.confidence_threshold = confidence_threshold
        self.seg_interval = seg_interval

        color_path_list = glob.glob(os.path.join(self.seq_dir, image_path_pattern))
        self.image_list = ns.natsorted(color_path_list)  # path of each RGB image, sorted by frame_ID
        if seg_interval > 1:
            self.image_list = self.image_list[::seg_interval]

        self.output_mask_dir = os.path.join(self.output_seq_dir, 'mask')
        self.output_mask_color_dir = os.path.join(self.output_seq_dir, 'mask_color')
        self.output_mask_feature_dir = os.path.join(self.output_seq_dir, "mask_embeddings")

        os.makedirs(self.output_mask_dir, exist_ok=True)
        os.makedirs(self.output_mask_color_dir, exist_ok=True)
        os.makedirs(self.output_mask_feature_dir, exist_ok=True)

        self.demo = VisualizationDemo(cfg)  # load CropFormer

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # Step 1: read RGB
        path = self.image_list[idx]
        img = read_image(path, format="BGR")

        if img.shape[0] != self.dst_h or img.shape[1] != self.dst_w:
            img = cv2.resize(img, (self.dst_w, self.dst_h), interpolation=cv2.INTER_NEAREST)  # ndarray(H, W, 3)
        else:
            img = img.copy()

        # Step 2: infer on CropFormer
        predictions = self.demo.run_on_image(img)  # ***
        pred_masks = predictions["instances"].pred_masks
        pred_scores = predictions["instances"].scores

        # 2.1: select by confidence threshold
        selected_indexes = (pred_scores >= args.confidence_threshold)
        selected_scores = pred_scores[selected_indexes]  # Tensor(instance_num, ), dtype=float, device=cuda:0
        selected_masks = pred_masks[selected_indexes]  # Tensor(instance_num, H, W), dtype=float, device=cuda:0

        return path, img, selected_scores, selected_masks


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    cfg = setup_cfg(args)

    # Step 1: preparation
    my_dataset = ImageDataset(cfg, args.seq_name, args.root, args.output_root, args.image_path_pattern, args.seg_interval, args.dst_h, args.dst_w, args.confidence_threshold)
    my_dataloader = DataLoader(my_dataset)

    # load CLIP model
    model, preprocess = load_clip(args.pretrained_path)

    for i, (path, img, selected_scores, selected_masks) in tqdm(enumerate(my_dataloader)):
        path = path[0]
        img = img[0].cpu().numpy()  # ndarray(H, W, 3)
        original_selected_scores = selected_scores[0]  # Tensor(mask_num, )
        selected_masks = selected_masks[0]  # Tensor(mask_num, H, W)

        _, m_H, m_W = selected_masks.shape
        mask_image = np.zeros((m_H, m_W), dtype=np.uint8)  # value: mask_ID (starting from 1)
        mask_color_image = np.zeros((m_H, m_W, 3), dtype=np.uint8)

        # Step 2: read RGB frame --> segment --> save segmentation result
        mask_id = 1
        selected_scores, ranks = torch.sort(original_selected_scores, descending=True)
        rgb_value_list = get_new_pallete(ranks.shape[0])  # list of int (3 * label_num)
        valid_instance_num = 0
        valid_mask_ids = []
        valid_mask_scores = []
        for index in ranks:
            num_pixels = torch.sum(selected_masks[index])
            mask_score = selected_scores[index].item()
            if num_pixels < args.min_mask_pixel_size:
                continue  # ignore small masks
            valid_instance_num += 1

            # draw mask_ID map
            mask_image[(selected_masks[index] == 1).cpu().numpy()] = mask_id
            valid_mask_ids.append(mask_id)
            valid_mask_scores.append(mask_score)
            mask_id += 1

            # assign different color to each mask_ID
            rgb_value = [rgb_value_list[index * 3], rgb_value_list[index * 3 + 1], rgb_value_list[index * 3 + 2]]
            mask_color_image[(selected_masks[index] == 1).cpu().numpy()] = rgb_value
        print("frame_%d: %d valid instances..." % (i, valid_instance_num))
        cv2.imwrite(os.path.join(my_dataset.output_mask_dir, os.path.basename(path).split('.')[0] + '.png'), mask_image)
        cv2.imwrite(os.path.join(my_dataset.output_mask_color_dir, os.path.basename(path).split('.')[0] + '.png'), mask_color_image)

        mask_visual_embedding_list = []
        # Step 3: get cropped image
        for valid_mask_id in valid_mask_ids:
            # 3.1: get image crop and do multi-scaling
            mask = (mask_image == valid_mask_id)
            cropped_images = get_cropped_image(mask, img)

            # 3.2: pre-process
            input_images = [preprocess(pad_into_square(Image.fromarray(cropped_image))) for cropped_image in cropped_images]
            images = torch.stack(input_images)  # Tensor(scale_num, 3, h_padded, w_padded)

            # 3.3: feed this multi-scale image crop to CLIP visual encoder
            images = images.reshape(-1, 3, 224, 224)  # Tensor(scale_num, 3, 224, 224)
            image_input = images.cuda()
            with torch.no_grad():
                image_features = model.encode_image(image_input).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)  # Tensor(scale_num, CLIP_dim)

            mean_image_feature = image_features.mean(dim=0)  # Tensor(CLIP_dim, )
            mask_visual_embedding_list.append(mean_image_feature)

        mask_visual_embeddings = torch.stack(mask_visual_embedding_list, dim=0)
        mask_visual_embeddings = mask_visual_embeddings / (mask_visual_embeddings.norm(dim=-1, keepdim=True) + 1e-7)  # normalization for visual embeddings

        output_embeddings_basename = os.path.basename(path).split('.')[0] + '.pt'
        embeddings_path = os.path.join(my_dataset.output_mask_feature_dir, output_embeddings_basename)
        torch.save(mask_visual_embeddings, embeddings_path)
