import sys
sys.path.append("third_party/FCGF")
import argparse
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from Dataset.dataset import get_dataset
import tool.config as config
from Scene_rep import Scene_rep

max_frame_num = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="./config/scannet_yuan.yaml")
    parser.add_argument("-d", "--dataset", type=str, default="~/t7/ScanNet/aligned_scans")
    parser.add_argument("-i", "--instance_dir", type=str, default="./data/grounding_sam_result/scannet/scene0653_00")
    parser.add_argument("--seq_name", default="scene0011_00")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("-o", "--output_dir", default="./output/scannet")
    args = parser.parse_args()

    cfg = config.load_config(args.config)

    dataset = get_dataset(args.dataset, args.instance_dir, cfg, args.device)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    last_seg_frame_id = dataset.last_seg_frame_id

    scene_rep = Scene_rep(cfg, args, dataset, args.device)
    last_valid_c2w = torch.eye(4)

    local_output_dir = str( os.path.join(args.output_dir, args.seq_name) )
    os.makedirs(local_output_dir, exist_ok=True)

    ########################################### main process ###########################################
    for frame_id, (color_img, depth_img, pose_c2w, seg_image, mask_features, seg_flag) in tqdm(enumerate(dataloader)):
        if max_frame_num is not None and frame_id >= max_frame_num:
            break

        # @color_img: Tensor(1, H , W, 3), dtype=float32, RGB
        # @depth_img: Tensor(1, H, W), dtype=float32
        # @pose_c2w: Tensor(1, 4, 4), dtype=float32
        # @seg_image: Tensor(1, H, W), dtype=uint8
        # @mask_features: Tensor(1, mask_num, 512), dtype=float32
        # @seg_flag: Tensor(1, ), dtype=bool, dtype=cpu

        # Step 1: preparation
        if torch.isnan(pose_c2w[0]).any().item() or torch.isinf(pose_c2w[0]).any().item():
            pose_c2w[0] = last_valid_c2w
        else:
            last_valid_c2w = pose_c2w[0]

        if seg_flag[0]:
            seg_image = seg_image[0]  # mask_id of each pixel, 0 means this pixel doesn't belongs to any mask, ndarray(H, W), dtype=uint8
            mask_features = mask_features[0]  # semantic feature extracted from 2D Foundation model

        # Step 2: do integration
        if frame_id % cfg["mapping"]["keyframe_freq"] == 0:
            with torch.no_grad():
                # 2.1: fusing current frame
                pose_w2c = torch.inverse(pose_c2w[0])
                frustum_block_coords, extrinsic = scene_rep.integrate_frame(frame_id, color_img[0], depth_img[0], pose_w2c)  # integrate this frame to global PC

                if seg_flag[0] and frame_id % cfg["seg"]["seg_add_interval"] == 0:
                    # 2.2: add detected masks in current frame into global mask bank
                    valid_mask_ids, valid_mask_voxels = scene_rep.insert_seg_frame(frame_id, color_img[0], depth_img[0], pose_c2w[0], frustum_block_coords, seg_image, mask_features)

                    # 2.3: periodically merge some rows and cols
                    if frame_id > 0 and frame_id % scene_rep.merge_frame_interval == 0:
                        scene_rep.update_masks(frame_id)

        # Step 2.4: update visualized segmentation result
        if scene_rep.vis_pc_flag:
            scene_rep.vis_pc.update()

        # Step 2.5: save latest ckpt
        if cfg["save"]["ckpt_interval"] > 0 and frame_id > 0 and frame_id % cfg["save"]["ckpt_interval"] == 0:
            ckpt_save_dir = os.path.join(local_output_dir, "ckpt_%d" % frame_id)
            os.makedirs(ckpt_save_dir, exist_ok=True)
            scene_rep.save_ckpt(frame_id, ckpt_save_dir)


    # Step 3: save final segmentation results
    print("################ Begin to save final ckpt and seg mesh/pointcloud...")
    frame_id_final = dataset.__len__() - 1 if max_frame_num is None else max_frame_num - 1  # final frame_ID of this sequence

    ckpt_save_dir = os.path.join(local_output_dir, "ckpt_%d" % frame_id_final)
    os.makedirs(ckpt_save_dir, exist_ok=True)

    # save final ckpt
    final_ckpt_path = os.path.join(local_output_dir, "ckpt_final.npz")
    pred_instance_mask_list = scene_rep.save_ckpt(frame_id_final, ckpt_save_dir, filter_flag=True, ckpt_path=final_ckpt_path)

    # save final seg pc
    scene_rep.save_merging_result(frame_id_final, reextract=False)

    # save finally reconstructed pointcloud
    final_pc_path = os.path.join(local_output_dir, "final.ply")
    scene_rep.save_pc(scene_rep.points, scene_rep.colors, final_pc_path)

    print("Input sequence finished !")

