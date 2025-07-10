import os
import time
import cv2
import numpy as np
import torch
import torch.utils.dlpack
import open3d as o3d
import open3d.core as o3c

from voxelized_points import VoxelBlockGrid
from PC_extractor import PointFeatureExtractor
from MaskGraph import MaskGraph
from Metrics import Metrics
from tool.geometric_helpers import compute_complementary, keep_min_rows
from tool.helper_functions import create_lower_triangular_matrix, set_diagonal_to_zero, set_row_and_column_zero, do_clustering, \
        extract_rows_and_cols, mask_matrix_rows_and_cols, retain_max_per_column, assign_elements_2d
from tool.post_process import filter_instances, export_instance_mask
from tool.visualization_helpers import vis_one_object, adjust_colors_to_pastel
from tool.vis_utils import Vis_color, Vis_pointcloud


class Scene_rep:
    def __init__(self, cfg, args, dataset, device="cuda:0"):
        self.cfg = cfg
        self.args = args
        self.dataset = dataset
        self.device = device
        self.device_o3c = o3c.Device(self.device)
        self.img_h = dataset.target_h
        self.img_w = dataset.target_w
        self.depth_scale = self.cfg["cam"]["depth_scale"]
        self.depth_far = self.cfg["cam"]["depth_far"]
        self.intrinsics = dataset.cam_intrinsic  # Tensor(3, 3), dtype=float32
        self.intrinsic_o3c = o3c.Tensor.from_numpy( self.intrinsics.cpu().numpy().astype("float64") )
        self.pinhole_cam_intrinsic = dataset.pinhole_cam_intrinsic  # o3d.camera.PinholeCameraIntrinsic obj

        # VoxelBlockGrid representation
        self.voxel_grids = VoxelBlockGrid(self.cfg, dataset, self.device)  # ***
        self.voxel_block_grids = self.voxel_grids.voxel_block_grids
        self.voxel_hash_table = self.voxel_grids.voxel_hash_table
        self.trunc_voxel_multiplier = self.cfg["scene"]["trunc_voxel_multiplier"] if "trunc_voxel_multiplier" in self.cfg["scene"] else 4.

        # vars for corresponding (active) voxels in the latest frame
        self.active_frame_id = -1
        self.active_voxel_coords = None
        self.active_voxel_indices = None

        # Pointcloud geometric feature extractor
        self.pc_extractor = PointFeatureExtractor(self.cfg, self.voxel_grids, self.device)
        self.pc_xyz = None  # latest extracted pointcloud, Tensor(pts_num, 3)
        self.pc_feature = None  # latest extracted feature pointcloud, Tensor(pts_num, feature_dim)
        self.pc_feature_frame_id = -1  # latest frame_id that we extracted feature pointcloud, int

        self.maskGraph = MaskGraph(self.cfg, self.dataset, self.voxel_grids, self.device)
        self.metrics = Metrics(self.cfg, self.device)

        # related matrix
        self.t_mask_num = self.cfg["scene"]["mask_num"]  # total mask num
        self.contain_matrix = self.maskGraph.contain_matrix  # [i, j]: the proportion that mask_i contains mask_j (overlap voxel_num / mask_j voxel_num)

        # *** for segmentation
        self.g_mask_num = 0  # number of global(original) masks
        self.c_mask_num = 0  # number of merged masks
        self.global_f_mask_ids = []  # index: each original 2D_mask's global_mask_id (1D); value: this original 2D_mask's f_mask_id, (frame_id, mask_id); (append-only)
        self.global2merged_mask_id = []  # index: each original 2D_mask's global_mask_id (1D), value: this original 2D_mask's corresponding merged_mask currently; (append-only)
        self.merged_mask_size = []  # each merged_mask's current size (number of voxel)
        self.mask_voxels_coords = self.maskGraph.mask_voxels_coords  # a dict, key is merged mask's merged_mask_id(1D); value merged mask's corresponding voxel_ids, Tensor(m_pts_num, ), dtype=int64;
        self.mask_features = torch.zeros((self.t_mask_num, self.cfg["mask"]["feature_dim"]), dtype=torch.float32, device=self.device)  # each merged_mask's semantic feature, Tensor(512, )
        self.merged_mask_weight = torch.zeros((self.t_mask_num, ), dtype=torch.float32, device=self.device)
        self.mask_geo_features = torch.zeros((self.t_mask_num, self.cfg["pc_extractor"]["feature_dim"]), dtype=torch.float32, device=self.device)  # each merged_mask's geometric feature, Tensor(16, )
        self.mask_geo_features_mask = torch.zeros((self.t_mask_num, ), dtype=torch.float32, device=self.device)  # whether each merged mask's geometric feature has been extracted successfully
        self.merged_mask_bbox = torch.zeros((self.t_mask_num, 6), dtype=torch.float32, device=self.device)  # each row: [x_min, y_min, z_min, x_max, y_max, z_max]
        self.merged_mask_last_frame = -1 * torch.ones((self.t_mask_num,), dtype=torch.int64, device=self.device)
        self.merge_time = 0
        self.pc_frame_id = -1
        self.points = None
        self.colors = None

        self.pred_inst_mask_frame_id = -1
        self.pred_inst_masks = None

        self.initialize_seg_vars()

        # for debug and visualization
        self.vis_color_flag = self.cfg["vis"]["vis_color"]
        self.vis_pc_flag = self.cfg["vis"]["vis_pc"]
        self.vis_c = Vis_color(self.vis_color_flag)
        self.vis_pc = Vis_pointcloud(self.vis_pc_flag, self.args, self.device)

        # for saving
        self.save_interval = self.cfg["save"]["save_interval"]
        self.seq_output_dir = str( os.path.join(self.args.output_dir, self.args.seq_name) )
        os.makedirs(self.seq_output_dir, exist_ok=True)
    # END __init__()

    def initialize_seg_vars(self):
        self.merge_frame_interval = self.cfg["seg"]["seg_add_interval"] * self.cfg["seg"]["merge_kf_interval"]  # frame interval num for updating masks
        self.merge_overlap_thresh = self.cfg["seg"]["merge_overlap_thresh"]  # default: 0.4
        self.with_feature = self.cfg["seg"]["with_feature"]
        self.sim_merge_thresh = self.cfg["seg"]["sim_merge_thresh"]
        self.merge_contain_iter_num = self.cfg["seg"]["merge_contain_iter_num"]
        self.merge_contain_ratio = self.cfg["seg"]["merge_contain_ratio"]
        self.merge_contain_ratio_feat_list = self.cfg["seg"]["merge_contain_ratio_feat_list"]
        self.contain_feature_sim_thresh_list = self.cfg["seg"]["contain_feature_sim_thresh_list"]
        self.containing_ratio = self.cfg["seg"]["containing_ratio"]  # default: 0.8
        self.contained_ratio = self.cfg["seg"]["contained_ratio"]  # default: 0.1
        self.merge_supporter_num = self.cfg["seg"]["merge_supporter_num"]  # supporter_num threshold, default: 5


    #################################################### properties ####################################################
    @property
    def get_pc_xyz(self):
        pcd = self.voxel_block_grids.extract_point_cloud()
        points = pcd.point.positions.cpu().numpy()
        return points

    @property
    def get_mask_features(self):
        return self.mask_features[:self.c_mask_num]

    @property
    def get_mask_geo_features(self):
        return self.mask_geo_features[:self.c_mask_num]

    @property
    def get_mask_geo_features_mask(self):
        return self.mask_geo_features_mask[:self.c_mask_num]

    @property
    def get_merged_mask_weight(self):
        return self.merged_mask_weight[:self.c_mask_num]

    @property
    def get_merged_mask_last_frame(self):
        return self.merged_mask_last_frame[:self.c_mask_num]

    @property
    def get_mask_w_geo_feature(self):
        return torch.where(self.mask_geo_features_mask[:self.c_mask_num] > 0)[0]

    @property
    def get_mask_wo_geo_feature(self):
        return torch.where(self.mask_geo_features_mask[:self.c_mask_num] <= 0)[0]

    @property
    def get_containing_mat(self):
        return self.contain_matrix[:self.c_mask_num, :self.c_mask_num]

    @property
    def get_containing_mat_w_geo_feature(self):
        c_mat = self.get_containing_mat
        invalid_m_mask_id = self.get_mask_wo_geo_feature
        c_mat_w_geo_feature = set_row_and_column_zero(c_mat, invalid_m_mask_id, invalid_m_mask_id)
        return c_mat_w_geo_feature


    #################################################### functions ####################################################

    # @brief: integrate a new frame into VoxelBlockGrids (only for mapping);
    # @param color_img: Tensor(h, w, 3), dtype=float32, RGB;
    # @param depth_img: depth image (metrics: m), Tensor(h, w), dtype=float32;
    # @param pose_w2c: Tensor(4, 4), dtype=float32;
    #-@return frustum_block_coords: Voxel coordinates of voxels in the frustum, o3d.Tensor(n, 3), dtype=int32;
    #-@return extrinsic: o3d.Tensor(4, 4), dtype=float32.
    def integrate_frame(self, frame_id, color_img, depth_img, pose_w2c):
        color_img = (color_img * 255.).cpu().numpy().astype("uint8")
        depth_img = (depth_img * self.depth_scale).cpu().numpy().astype("uint16")
        pose_w2c = pose_w2c.cpu().numpy().astype("float64")

        color_img = np.ascontiguousarray(color_img)
        depth_img = np.ascontiguousarray(depth_img)
        pose_w2c = np.ascontiguousarray(pose_w2c)
        color = o3d.t.geometry.Image(color_img).to(o3c.uint8).to(self.device_o3c)
        depth = o3d.t.geometry.Image(depth_img).to(o3c.uint16).to(self.device_o3c)
        extrinsic = o3c.Tensor.from_numpy(pose_w2c)

        # Get active frustum block coordinates from input frame, and them integrate input frame to TSDF Volume
        frustum_block_coords = self.voxel_block_grids.compute_unique_block_coordinates(depth, self.intrinsic_o3c, extrinsic, self.depth_scale, self.depth_far)  # active blocks of this frame
        self.voxel_block_grids.integrate(frustum_block_coords, depth, color, self.intrinsic_o3c, extrinsic, self.depth_scale, self.depth_far, trunc_voxel_multiplier=self.trunc_voxel_multiplier)

        return frustum_block_coords, extrinsic


    # @brief: update active frame information
    def set_active_frustum(self, frame_id, voxel_coords, voxel_indices):
        self.active_frame_id = frame_id
        self.active_voxel_coords = voxel_coords
        self.active_voxel_indices = voxel_indices


    # @brief: for a merged mask, compute its merged geometric feature by weighted averaging from all its original masks before merging;
    # @param mask_cluster: mask_IDs to merge, list of int;
    #-@return cluster_geo_feature_merged: aggregated geometric feature, Tensor(feat_dim, ), dtype=float32;
    #-@return cluster_geo_feature_mask_merged: flag of whether this merged mask has geometric feature, 0/1, dtype=float32.
    def aggregate_masks_geo_features(self, mask_cluster):
        cluster_weights = self.get_merged_mask_weight[mask_cluster]
        cluster_geo_features_mask = self.get_mask_geo_features_mask[mask_cluster]
        cluster_geo_features_weight = cluster_weights * cluster_geo_features_mask  # Tensor(mask_num_i, )

        cluster_geo_features = self.get_mask_geo_features[mask_cluster]  # Tensor(mask_num_i, geo_feat_dim)
        weighted_cluster_geo_features = cluster_geo_features_weight[..., None] * cluster_geo_features  # Tensor(mask_num_i, geo_feat_dim)
        cluster_geo_feature_merged = torch.sum(weighted_cluster_geo_features, dim=0) / torch.sum(cluster_geo_features_weight)
        cluster_geo_feature_mask_merged = torch.any(cluster_geo_features_mask).float()
        return cluster_geo_feature_merged, cluster_geo_feature_mask_merged


    # @brief: insert a new seg frame, then process masks in this frame and incorporate it in Contain_Matrix
    # @param color_img: Tensor(h, w, 3), dtype=float32, RGB;
    # @param depth_img: depth image (metrics: m), Tensor(h, w), dtype=float32;
    # @param pose_c2w: Tensor(4, 4), dtype=float32;
    # @param frustum_block_coords: Voxel coordinates of voxels in the frustum, o3d.Tensor(n, 3), dtype=int32;
    # @param extrinsic: o3d.Tensor(4, 4), dtype=float32;
    # @param seg_image: Tensor(H, W), dtype=uint8;
    # @param mask_sem_features: semantic feature extracted from 2D Foundation model, Tensor(mask_num, 512), dtype=float32;
    #-@return frame_valid_mask_ids: merged_mask_IDs of all new mask inserted in this frame, list of int;
    #-@return frame_valid_mask_voxels: voxel coordinates(in World CS) of all new mask inserted in this frame, list of Tensor(n_i, 3).
    def insert_seg_frame(self, frame_id, color_img, depth_img, pose_c2w, frustum_block_coords_o3c, seg_image, mask_sem_features, mask_min_size=50):
        # Step 1: get corresponding blocks and their containing voxels in VoxelBlockGrid of current frame
        cur_block_indices_o3c, _ = self.voxel_block_grids.hashmap().find(frustum_block_coords_o3c)  # Tensor(N, )
        o3c.cuda.synchronize()
        voxel_coords_o3c, voxel_indices_o3c = self.voxel_block_grids.voxel_coordinates_and_flattened_indices(cur_block_indices_o3c)  # voxel coordinates and 1D indices contained by given blocks

        frustum_block_coords = torch.utils.dlpack.from_dlpack(frustum_block_coords_o3c.to_dlpack())
        cur_block_indices = torch.utils.dlpack.from_dlpack(cur_block_indices_o3c.to_dlpack())  # *** key in hashmap of each selected VoxelBlock, Tensor(s_block_num, ), dtype=int32
        voxel_coords = torch.utils.dlpack.from_dlpack(voxel_coords_o3c.to_dlpack())  # *** voxel coordinates in selected blocks(in World CS), Tensor(c_voxel_num, 3), dtype=float32
        voxel_indices = torch.utils.dlpack.from_dlpack(voxel_indices_o3c.to_dlpack())  # 1D indices of each voxel coordinate (in World CS)

        self.set_active_frustum(frame_id, voxel_coords, voxel_indices)  # update active frame frustum

        # Step 2: find valid 2D masks in this frame, and compute their corresponding voxels in 3D (coordinates in World CS) by back-projecting
        # 2.1: back-project each 2D mask in this frame to 3D
        mask_dict, valid_mask_ids, _, valid_mask_sem_features = self.voxel_grids.turn_mask_to_voxel(frame_id, depth_img, pose_c2w, voxel_coords, seg_image, mask_sem_features)
        mask_dict, mask_indices_dict, mask_in_frame_list = self.maskGraph.add_frame_masks(frame_id, mask_dict)  # compute each voxel's corresponding mask_ID in this frame

        # 2.2: check whether each 3D mask is valid
        valid_mask_ids_new = []
        mask_dict_new = {}
        valid_mask_sem_features_new = []
        mask_indices_dict_new = {}
        mask_in_frame_list_new = []
        for mask_id, mask_sem_feature, mask_in_frame in zip(valid_mask_ids, valid_mask_sem_features, mask_in_frame_list):
            mask_coords = mask_dict[mask_id]
            mask_indices = mask_indices_dict[mask_id]
            mask_size = mask_coords.shape[0]
            if mask_size > mask_min_size:
                valid_mask_ids_new.append(mask_id)
                mask_dict_new[mask_id] = mask_coords
                valid_mask_sem_features_new.append(mask_sem_feature)
                mask_indices_dict_new[mask_id] = mask_indices
                mask_in_frame_list_new.append(mask_in_frame)

        valid_mask_ids = valid_mask_ids_new
        mask_dict = mask_dict_new
        valid_mask_sem_features = valid_mask_sem_features_new
        mask_indices_dict = mask_indices_dict_new
        mask_in_frame_list = mask_in_frame_list_new

        mask_num_this_frame = len(valid_mask_ids)
        if mask_num_this_frame == 0:
            return [], []

        # Step 3: for each newly detected 2D mask, fill the new rows and new cols in Containing_matrix
        for i in range(mask_num_this_frame):
            mask_id = valid_mask_ids[i]
            f_mask_id = (frame_id, mask_id)
            merged_mask_id = self.c_mask_num + i  # 1D index in all merged masks
            mask_sem_feature = valid_mask_sem_features[i]  # visual embedding of this mask, Tensor(feature_dim, )

            # 3.1: fill related vars and create a new Instance obj for this new mask
            mask_voxel_coords = mask_dict[mask_id]
            mask_voxel_indices = mask_indices_dict[mask_id]

            self.maskGraph.insert_instance(frame_id, mask_id, merged_mask_id, mask_voxel_coords, mask_voxel_indices, mask_sem_feature)
            self.maskGraph.f_mask_voxels_clouds[f_mask_id] = mask_voxel_coords
            self.maskGraph.merged_mask_in_frame_mat[merged_mask_id] = mask_in_frame_list[i]
            self.mask_voxels_coords[merged_mask_id] = mask_voxel_coords
            self.mask_features[merged_mask_id] = mask_sem_feature

            # 3.2: compute contain_ratio between this new mask and existing masks
            if len(self.global_f_mask_ids) == 0:
                # for 1st seg image
                self.contain_matrix[i, i] = 1.
            else:
                # for >=2nd seg image,
                # 3.2.1: for this mask, compute other overlapping masks and their overlap voxel num
                overlap_mask_ids, contain_ratio_q2o, contain_ratio_o2q = self.maskGraph.query_mask_under_visible_part(mask_in_frame_list[i], mask_voxel_coords, self.global2merged_mask_id)

                # 3.2.2: for the row and column of this new mask in Containing_matrix, fill the values corresponding to overlapping masks
                if overlap_mask_ids.shape[0] > 0:
                    # 3.2.2.1: fill its corresponding row (containing ratio of this mask to existing masks)
                    mask_row_this = torch.zeros((self.c_mask_num, )).to(self.contain_matrix)
                    mask_row_this[overlap_mask_ids] = contain_ratio_q2o
                    self.contain_matrix[merged_mask_id, :self.c_mask_num] = mask_row_this

                    # 3.2.2.2: fill its corresponding col (containing ratio of existing masks to this mask)
                    mask_col_this = torch.zeros((self.c_mask_num,)).to(self.contain_matrix)
                    mask_col_this[overlap_mask_ids] = contain_ratio_o2q
                    self.contain_matrix[:self.c_mask_num, merged_mask_id] = mask_col_this

                self.contain_matrix[merged_mask_id, merged_mask_id] = 1.
        # END for

        # Step 4: for each valid 2D mask in current frame, update corresponding vars
        frame_valid_mask_ids = []  # merged_mask_id of each newly added mask in this frame
        frame_valid_mask_voxels = []
        frame_global_mask_id = []
        for i in range(mask_num_this_frame):
            mask_id = valid_mask_ids[i]  # mask_ID of this 2D mask in current frame, int
            f_mask_id = (frame_id, mask_id)
            mask_voxels = mask_dict[mask_id]  # Voxel coords of this 2D mask, Tensor(n_i, 3)
            merged_mask_id = self.c_mask_num  # merged (1D) mask_ID of this original 2D mask in merged mask list
            global_mask_id = len(self.global_f_mask_ids)
            frame_global_mask_id.append(global_mask_id)

            frame_valid_mask_ids.append(merged_mask_id)
            frame_valid_mask_voxels.append(mask_voxels)

            # 4.1: update its global_mask_id and merged_mask_id
            self.global_f_mask_ids.append(f_mask_id)
            self.global2merged_mask_id.append(merged_mask_id)  # for new 2D masks in this frame, its merged_mask_ID is the current largest merged_mask_ID + 1
            self.merged_mask_size.append(mask_voxels.shape[0])
            self.merged_mask_weight[merged_mask_id] = 1.
            self.merged_mask_last_frame[merged_mask_id] = frame_id

            # 4.2: update voxel hashing table
            self.voxel_hash_table.insert_mask_voxels(mask_voxels, global_mask_id)  # ***
            self.c_mask_num += 1
            self.g_mask_num += 1

        # Step 5: for each newly added mask, extract its geometric feature by querying
        if self.get_mask_w_geo_feature.shape[0] == 0:
            mask_voxels_list = list(self.mask_voxels_coords.values())
            mask_id_list = [i for i in range(len(mask_voxels_list))]
        else:
            mask_voxels_list = frame_valid_mask_voxels
            mask_id_list = frame_valid_mask_ids

        extract_flag, mask_geo_feature_list, pts_feature, pts_xyz = self.pc_extractor.get_masks_geometric_features(mask_voxels_list, scene_points=self.get_pc_xyz)
        if extract_flag:
            for i, mask_geo_feature in enumerate(mask_geo_feature_list):
                if mask_geo_feature is None:
                    continue
                merged_mask_id = mask_id_list[i]
                self.mask_geo_features[merged_mask_id] = mask_geo_feature
                self.mask_geo_features_mask[merged_mask_id] = 1.

        # save extracted feature pointcloud
        if extract_flag and pts_feature is not None and pts_xyz is not None:
            self.pc_xyz = pts_xyz
            self.pc_feature = pts_feature
            self.pc_feature_frame_id = frame_id

        # Step 6: visualize current color image
        if self.vis_color_flag:
            self.vis_c.update(color_img)

        return frame_valid_mask_ids, frame_valid_mask_voxels
    # END insert_frame()


    # @brief: re-compute geometric feature for each given mask using latest reconstructed pointcloud;
    def update_mask_geo_features(self, frame_id, merged_mask_ids=None):
        if self.pc_xyz is None or self.pc_feature is None:
            return False

        ########################### TEST ###########################
        start_time = time.time()
        mask_w_geo_num_bf = torch.sum(self.get_mask_geo_features_mask).item()
        ########################### END TEST ###########################

        # Step 1: preparation
        if merged_mask_ids is None:
            merged_mask_voxel_coords_list = list(self.mask_voxels_coords.values())
            merged_mask_num_valid = self.c_mask_num
            merged_mask_ids = [ i for i in range(self.c_mask_num) ]
        else:
            merged_mask_voxel_coords_list = [self.mask_voxels_coords[mask_id] for mask_id in merged_mask_ids]
            merged_mask_num_valid = merged_mask_ids.shape[0]
        if len(merged_mask_voxel_coords_list) == 0:
            return False

        # Step 2: re-extract per-point geometric feature for latest reconstructed pointcloud
        _, merged_mask_geo_feature_list, _, _ = self.pc_extractor.get_masks_geometric_features(merged_mask_voxel_coords_list, scene_points=self.pc_xyz, pts_feature=self.pc_feature)

        # Step 3: for each selected merged_mask, re-compute geometric feature for this merged mask, and update
        for i in range(merged_mask_num_valid):
            merged_mask_id = merged_mask_ids[i]
            merged_mask_geo_feature = merged_mask_geo_feature_list[i]
            if merged_mask_geo_feature is not None:
                merged_mask_geo_feature_mask = 1.
                self.mask_geo_features[merged_mask_id] = merged_mask_geo_feature
                self.mask_geo_features_mask[merged_mask_id] = merged_mask_geo_feature_mask

        ########################### TEST ###########################
        time_interval = time.time() - start_time
        mask_w_geo_num_aft = torch.sum(self.get_mask_geo_features_mask).item()
        print("\nAt frame_%d, number of mask with geometric feature: %d --> %d, takes %.4f s" % (frame_id, mask_w_geo_num_bf, mask_w_geo_num_aft, time_interval))
        ########################### END TEST ###########################
        return True


    # @brief: for existing 2D masks, detect the clusters of 2D masks needed to be merged, and then merge them
    def update_masks(self, frame_id):
        # Step 0: preparation: update each merged_mask's geometric feature
        self.update_mask_geo_features(frame_id)

        c_mat, sem_feature_mat, geo_feature_mat = self.get_containing_mat, self.get_mask_features, self.get_mask_geo_features
        ut_mask = create_lower_triangular_matrix(self.c_mask_num)

        # Step 1: Mask Merging Strategy 1: Overall Similarity
        # 1.1: compute similarity matrix based on: IoU + semantic feature similarity + geometric feature similarity
        final_sim_mat, iou_mat, sem_feature_sim_mat, geo_feature_sim_mat = self.metrics.compute_final_sim_mat(c_mat, sem_feature_mat, geo_feature_mat, use_IoU=True)
        sim_merge_mat = ut_mask & (final_sim_mat > self.sim_merge_thresh)  # using fixed classifying plane (TODO: to be improved )

        iou_merge_mat = ut_mask & (c_mat > self.merge_overlap_thresh) & (c_mat.T > self.merge_overlap_thresh)  # Double check: only mask pairs with mutual OR > threshold can be merged
        sim_merge_mat = sim_merge_mat & iou_merge_mat

        # Step 2: Mask Merging Strategy 2: supporter_num
        # 2.1: preparation
        c_mat_nd = c_mat
        mask_weight = self.merged_mask_weight[:self.c_mask_num]
        w_mat = torch.diag(mask_weight)

        # 2.2： compute supporter_num for each mask pair (according to merged_mask_weight)
        A = (c_mat_nd.T > self.containing_ratio) & (c_mat_nd > self.contained_ratio)
        A = A.float()  # A[i, j]==1 <=> overlap_ratio[j, i]>threshold_big & overlap_ratio[i, j]>threshold_small
        A = set_diagonal_to_zero(A)  # set diagonals of containing matrix to zeros
        supporter_num_mat = A @ w_mat @ A.T
        supporter_num_mat = supporter_num_mat * ut_mask.float()  # Tensor(c_mask_num, c_mask_num), dtype=float
        supporter_num_merge_mat = (supporter_num_mat >= self.merge_supporter_num)

        # Step 3: find all masks that need to be merged, and then update c_mat
        merge_mat = torch.logical_or(sim_merge_mat, supporter_num_merge_mat)  # Tensor(c_mask_num, c_mask_num), dtype=bool
        merge_mat = set_row_and_column_zero(merge_mat, self.get_mask_wo_geo_feature, self.get_mask_wo_geo_feature)  # if a mask has no geometric feature, ignore it
        valid_merged_mask_ids, _ = self.merge_masks(frame_id, merge_mat, c_mat, count_merge=True)  # merged_mask_Ids whose weight exceeds threshold, Tensor(valid_m_mask_num, )

        if self.save_interval > 0 and frame_id % self.save_interval == 0:
            self.save_merging_result(frame_id, valid_merged_mask_ids)  # *** save result after merging

        # Step 4: for all valid merged_masks, if A is almost contained by B, merge them
        # *** An iterative process. Ensuring that in each iteration a small mask can only be contained by at most 1 big mask
        if valid_merged_mask_ids is not None and valid_merged_mask_ids.shape[0] > 0:
            for iter_merge in range(self.merge_contain_iter_num):
                c_mat = self.get_containing_mat
                c_mat = mask_matrix_rows_and_cols(c_mat, valid_merged_mask_ids)  # mask out rows & cols corresponding to invalid masks

                # 4.1: Merging rule 1: a merged mask contains another merged mask > threshold_1
                contain_merge_mat = (c_mat > self.merge_contain_ratio)  # Tensor(c_mask_num, c_mask_num), dtype=bool
                if torch.count_nonzero(contain_merge_mat) == 0:
                    break

                # make sure that a mask can be contained by at most 1 another mask
                c_mat_merge_valid = contain_merge_mat.to(c_mat) * c_mat
                c_mat_merge_valid = set_diagonal_to_zero(c_mat_merge_valid)  # *** before get max element for each row, set all diagonal elements to zero
                c_mat_r = retain_max_per_column(c_mat_merge_valid)
                contain_merge_mat = c_mat_r.to(contain_merge_mat)

                # 4.2: Merging rule 2 (iterative merge with semantic similarity):
                if self.with_feature:
                    sem_feature_mat = self.get_mask_features
                    geo_feature_mat = self.get_mask_geo_features
                    feature_sim_mat, _, sem_feature_sim_mat, geo_feature_sim_mat = self.metrics.compute_final_sim_mat(c_mat, sem_feature_mat, geo_feature_mat, use_IoU=False)
                    feature_sim_mat = mask_matrix_rows_and_cols(feature_sim_mat, valid_merged_mask_ids)

                    for i in range(len(self.merge_contain_ratio_feat_list)):
                        merge_contain_ratio_seman = self.merge_contain_ratio_feat_list[i]
                        feature_sim_thresh = self.contain_feature_sim_thresh_list[i]
                        contain_merge_mat2 = (c_mat_r > merge_contain_ratio_seman) & (feature_sim_mat > feature_sim_thresh)
                        contain_merge_mat = contain_merge_mat | contain_merge_mat2

                contain_merge_mat = set_row_and_column_zero(contain_merge_mat, self.get_mask_wo_geo_feature, self.get_mask_wo_geo_feature)  # if a mask has no geometric feature, ignore it
                valid_merged_mask_ids, merge_flag = self.merge_masks(frame_id, contain_merge_mat, self.get_containing_mat, count_merge=False)  # Tensor(valid_m_mask_num, )
                if not merge_flag:
                    break
            # END for

            if self.save_interval > 0 and frame_id % self.save_interval == 0:
                self.save_merging_result(frame_id, valid_merged_mask_ids=None)  # *** save result after merging

        # Step 5: remove mis-segmented raw masks periodically
        self.select_masks_to_remove(frame_id)

        # Step 6: update segmented PC for visualization
        if self.vis_pc_flag:
            # 6.1: get current reconstructed pc
            points, colors = self.get_pc()

            # 6.2: get current valid merged masks
            mask_weight_threshold = min(self.merge_time + 2, self.cfg["seg"]["mask_weight_threshold"])  # only merged_mask with weight>threshold will be visualized
            valid_merged_mask_ids = torch.where(self.merged_mask_weight[:self.c_mask_num] >= mask_weight_threshold)[0]

            # 6.3: for each valid merged mask, get its Instance obj and corresponding voxel coordinates
            valid_instance_list = [self.maskGraph.instance_dict[i] for i in valid_merged_mask_ids.tolist()]
            valid_instance_coords_list = [self.mask_voxels_coords[i] for i in valid_merged_mask_ids.tolist()]
            instance_pts_mask_list = self.get_instance_scene_pts_mask(valid_instance_coords_list, points)

            self.vis_pc.show_current_seg_pc(points, valid_instance_list, instance_pts_mask_list)

        torch.cuda.empty_cache()
        print("Finish merging at frame_%d ! Current merged mask number = %d" % (frame_id, self.c_mask_num))


    # @brief: invoked function for periodical mask removing;
    # @param r_mask_weight: weight threshold for mask removing.
    def select_masks_to_remove(self, frame_id, max_kf_interval=20, r_mask_weight=1):
        # Step 1: find masks whose weight==1 (candidates)
        candi_mask_ids = torch.where(self.merged_mask_weight[:self.c_mask_num] <= r_mask_weight)[0]
        if candi_mask_ids.shape[0] == 0:
            return None

        # Step 2: for each candidate mask, get their last-edit frame_ID, and whether it needs to be removed
        candi_masks = [self.maskGraph.instance_dict[i] for i in candi_mask_ids.tolist()]  # Instance obj list of candidates masks
        candi_masks_frame_intervals = [frame_id - candi_mask.frame_id for candi_mask in candi_masks]
        candi_masks_frame_intervals = torch.tensor(candi_masks_frame_intervals).to(self.device)  # Tensor(candi_mask_num, ), dtype=int64

        max_frame_interval = self.cfg["mapping"]["keyframe_freq"] * max_kf_interval
        candi_masks_r_flag = (candi_masks_frame_intervals >= max_frame_interval)
        if ~torch.any(candi_masks_r_flag):
            return None

        # Step 3: for each mask needs to be removed, delete it from merged mask list, and update corresponding info
        remove_mask_ids = candi_mask_ids[candi_masks_r_flag]
        self.remove_masks(frame_id, remove_mask_ids)


    # @brief: merge some existing masks, including merge corresponding rows and cols in containig matrix, and reassign merge_mask_Id for all merged masks;
    # @param merge_mat: bool matrix indicating which masks needed to be merged, Tensor(c_mask_num, c_mask_num), dtype=bool;
    # @param c_mat: Tensor(c_mask_num, c_mask_num);
    #-@return valid_merged_mask_ids:
    #-@return merge_flag: whether at least 2 masks are merged in this iteration, bool.
    def merge_masks(self, frame_id, merge_mat, c_mat, count_merge=False, full_geo_merge=True):
        if torch.count_nonzero(merge_mat) == 0:
            return None, False

        c_mask_num_before = self.c_mask_num
        merged_mask_id_clusters = do_clustering(merge_mat)  # list of ndarray(n_i)

        # (1) assign a new merged_mask_id for each cluster, and reassign other merged masks' merged_mask_id;
        # (2) in containing matrix, firstly remove rows (and cols) corresponding to masks that need to be merged, then append new rows (and cols) for merged masks to the end;
        # (3) update members: global2merged_mask_id, merged_mask_size, mask_voxels_indices, merged_mask_weight

        # Step 1: identify rows(and cols) to keep and to merge
        rows_to_keep = []  # list of int
        rows_to_merge = []  # list of list
        for merged_mask_id_cluster in merged_mask_id_clusters:
            if merged_mask_id_cluster.shape[0] == 1:
                rows_to_keep.append(merged_mask_id_cluster.item())
            else:
                rows_to_merge.append(merged_mask_id_cluster)

        kept_mask_num, merged_mask_num = len(rows_to_keep), len(rows_to_merge)
        merged_mask_num_new = kept_mask_num + merged_mask_num  # total mask number after this merging
        rc_to_keep = torch.tensor(rows_to_keep, dtype=torch.long)

        # case: No masks are merged in this iteration
        if merged_mask_num == 0:
            return None, False

        # Step 2: reassign merged_mask_id for each kept mask and merged mask
        self.maskGraph.keep_and_merge_masks(frame_id, c_mask_num_before, rows_to_keep, rows_to_merge)

        # 2.1: preparation, create new tensors
        global2merged_mask_id = torch.tensor(self.global2merged_mask_id, dtype=torch.long)  # Tensor(collected_global_mask_num, )
        global2merged_mask_id_new = -1 * torch.ones_like(global2merged_mask_id)
        c_mat_new = torch.eye(merged_mask_num_new).to(c_mat)  # Tensor(m_mask_num, m_mask_num)
        mask_voxels_coords_new = {}
        mask_features_new = torch.zeros((c_mask_num_before, self.cfg["mask"]["feature_dim"]), dtype=torch.float32, device=self.device)
        mask_geo_features_new = torch.zeros((c_mask_num_before, self.cfg["pc_extractor"]["feature_dim"]), dtype=torch.float32, device=self.device)
        mask_geo_features_mask_new = torch.zeros((c_mask_num_before, ), dtype=torch.float32, device=self.device)  # whether each merged mask's geometric feature has been extracted successfully
        merged_mask_size_new = []
        merged_mask_weight_new = torch.zeros((self.t_mask_num,), dtype=torch.float32, device=self.device)
        merged_mask_last_frame_new = -1 * torch.ones((self.t_mask_num,), dtype=torch.int64, device=self.device)

        # 2.2: re-assign merged_mask_id to each kept mask, and copy their corresponding rows & cols from current c_mat to new c_mat
        c_mat_to_keep = extract_rows_and_cols(c_mat, rc_to_keep)
        c_mat_new[:kept_mask_num, :kept_mask_num] = c_mat_to_keep

        new_mask_counter = 0
        for mask_id_kept in rows_to_keep:
            corr_global_mask_ids = torch.where(global2merged_mask_id == mask_id_kept)[0]
            global2merged_mask_id_new[corr_global_mask_ids] = new_mask_counter

            mask_voxels_coords_new[new_mask_counter] = self.mask_voxels_coords[mask_id_kept]
            mask_features_new[new_mask_counter] = self.get_mask_features[mask_id_kept]
            mask_geo_features_new[new_mask_counter] = self.get_mask_geo_features[mask_id_kept]
            mask_geo_features_mask_new[new_mask_counter] = self.get_mask_geo_features_mask[mask_id_kept]
            merged_mask_size_new.append(self.merged_mask_size[mask_id_kept])
            merged_mask_weight_new[new_mask_counter] = self.get_merged_mask_weight[mask_id_kept]
            merged_mask_last_frame_new[new_mask_counter] = self.get_merged_mask_last_frame[mask_id_kept]
            new_mask_counter += 1

        # 2.3: for each new merged mask, judge whether its is valid, and reassign a new merged_mask_id, and fill new c_mat in region (2) and (3)
        global2merged_mask_id_new_list = global2merged_mask_id_new.tolist()
        valid_merge_mask_ids = global2merged_mask_id_new[global2merged_mask_id_new != -1]
        valid_merge_mask_ids = torch.unique(valid_merge_mask_ids).tolist()  # mask_IDs of all kept merged masks

        merged_mask_voxel_coords_list = []  # voxel coordinates of all valid merged mask
        merged_mask_ids_new = []
        merged_mask_num_valid = 0
        rows_to_merge_valid = []
        mask_cluster_list = []
        for mask_cluster in rows_to_merge:
            merged_mask_id_new = new_mask_counter

            # 2.3.1: compute instance mask of this merged mask
            mask_cluster = mask_cluster.tolist()
            mask_cluster_list.append(mask_cluster)
            cluster_voxel_list = [self.mask_voxels_coords[mask_id] for mask_id in mask_cluster]
            cluster_voxel_coords = torch.concat(cluster_voxel_list, dim=0)
            cluster_voxel_coords = torch.unique(cluster_voxel_coords, dim=0)  # Voxel coordinates of this merged mask, Tensor(new_mask_size, 3)

            # 2.3.2: compute merged semantic feature of this merged mask
            cluster_features = self.get_mask_features[mask_cluster]
            cluster_feature_merged = cluster_features.sum(dim=0)

            # 2.3.3: update self.mask_voxels_indices, self.merged_mask_weight
            merged_mask_num_valid += 1
            merged_mask_ids_new.append(merged_mask_id_new)
            merged_mask_voxel_coords_list.append(cluster_voxel_coords)
            rows_to_merge_valid.append(mask_cluster)
            mask_voxels_coords_new[merged_mask_id_new] = cluster_voxel_coords
            mask_features_new[merged_mask_id_new] = cluster_feature_merged

            merged_mask_weight_new[merged_mask_id_new] = torch.sum(self.get_merged_mask_weight[mask_cluster])  # *** reassign weight of this merged mask
            merged_mask_last_frame_new[merged_mask_id_new] = frame_id

            # 2.3.4: add new rows and cols for this merged mask to all kept merged masks (region (2) and (3))
            # ***Attention: 这里先只考虑计算该merged mask与其他kept masks之间的containing ratio
            merged_mask_frame = self.maskGraph.merged_mask_in_frame_mat[merged_mask_id_new]
            overlap_mask_ids, containing_ratio, contained_ratio = self.maskGraph.query_mask_under_visible_part(merged_mask_frame, cluster_voxel_coords, global2merged_mask_id_new_list, valid_merge_mask_ids)

            if overlap_mask_ids.shape[0] > 0:
                # fill corresponding row of this merged mask (fill region (3))
                mask_row_this = torch.zeros((kept_mask_num, )).to(c_mat_new)
                mask_row_this[overlap_mask_ids] = containing_ratio
                c_mat_new[merged_mask_id_new, :kept_mask_num] = mask_row_this

                # fill its corresponding col of this merged mask (fill region (3))
                mask_col_this = torch.zeros((kept_mask_num, )).to(c_mat_new)
                mask_col_this[overlap_mask_ids] = contained_ratio
                c_mat_new[:kept_mask_num, merged_mask_id_new] = mask_col_this

            new_mask_counter += 1

        # 2.4: for each new merged mask, re-compute its geometric feature, fill new c_mat in region (4), and update self.global2merged_mask_id
        if full_geo_merge and frame_id == self.pc_feature_frame_id:
            _, merged_mask_geo_feature_list, _, _ = self.pc_extractor.get_masks_geometric_features(merged_mask_voxel_coords_list, scene_points=self.pc_xyz, pts_feature=self.pc_feature)
        else:
            merged_mask_geo_feature_list = [None for _ in range(merged_mask_num_valid)]

        for i in range(merged_mask_num_valid):
            merged_mask_id_i = merged_mask_ids_new[i]
            mask_voxel_coords_i = merged_mask_voxel_coords_list[i]  # Tensor(voxel_num_i, 3)

            # 2.4.1: self.merged_mask_size
            mask_cluster_tensor = torch.tensor(rows_to_merge_valid[i])
            corr_global_mask_ids = torch.where(torch.isin(global2merged_mask_id, mask_cluster_tensor))[0]
            global2merged_mask_id_new[corr_global_mask_ids] = merged_mask_id_i  # update global2merged_mask_id
            merged_mask_size_new.append(mask_voxel_coords_i.shape[0])

            # 2.4.2: re-compute geometric feature for this newly merged mask, and update
            merged_mask_geo_feature = merged_mask_geo_feature_list[i]
            merged_mask_geo_feature_mask = 1.
            if merged_mask_geo_feature is None:
                merged_mask_geo_feature, merged_mask_geo_feature_mask = self.aggregate_masks_geo_features(mask_cluster_list[i])
            mask_geo_features_new[merged_mask_id_i] = merged_mask_geo_feature
            mask_geo_features_mask_new[merged_mask_id_i] = merged_mask_geo_feature_mask

        # 2.4.3: re-compute containing ratio between merged masks, i.e. fill new c_mat in region (4)
        c_mat_new_instances = self.maskGraph.compute_masks_contain_ratio(merged_mask_ids_new, global2merged_mask_id_new.tolist())
        c_mat_new = assign_elements_2d(c_mat_new, c_mat_new_instances, merged_mask_ids_new, merged_mask_ids_new)

        # 2.5: set new tensors as members
        merged_mask_num_new_valid = kept_mask_num + merged_mask_num_valid
        self.global2merged_mask_id = global2merged_mask_id_new.tolist()
        self.contain_matrix[:self.c_mask_num, :self.c_mask_num] = torch.eye(self.c_mask_num).to(c_mat_new)
        self.contain_matrix[:merged_mask_num_new, :merged_mask_num_new] = c_mat_new
        self.c_mask_num = merged_mask_num_new_valid
        self.mask_voxels_coords = mask_voxels_coords_new
        self.mask_features[:c_mask_num_before] = mask_features_new
        self.mask_geo_features[:c_mask_num_before] = mask_geo_features_new
        self.mask_geo_features_mask[:c_mask_num_before] = mask_geo_features_mask_new
        self.merged_mask_size = merged_mask_size_new
        self.merged_mask_weight = merged_mask_weight_new
        self.merged_mask_last_frame = merged_mask_last_frame_new

        # Step 3: select valid masks by weight
        mask_weight_threshold = min(self.merge_time + 2, self.cfg["seg"]["mask_weight_threshold"])
        valid_merged_mask_ids = torch.where(self.merged_mask_weight[:self.c_mask_num] >= mask_weight_threshold)[0]

        if count_merge:
            self.merge_time += 1

        return valid_merged_mask_ids, True
    # END self.merge_masks()


    # @brief: remove masks periodically;
    # @param remove_mask_ids: mask_IDs of merged masks that need to be removed, Tensor(r_mask_num, ), dtype=int64, device=self.device;
    def remove_masks(self, frame_id, remove_mask_ids):
        # Step 1: find mask_IDs of kept masks
        all_mask_ids = torch.arange(0, self.c_mask_num, dtype=remove_mask_ids.dtype).to(self.device)
        kept_mask_ids = compute_complementary(all_mask_ids, remove_mask_ids)
        kept_mask_num = kept_mask_ids.shape[0]

        # Step 2: update mask info in self.maskGraph obj
        kept_mask_ids_list = [kept_mask_id.item() for kept_mask_id in kept_mask_ids]
        mask_num_new = self.maskGraph.keep_and_merge_masks(frame_id, self.c_mask_num, kept_mask_ids_list)

        # Step 3: update corresponding members
        # 3.1: preparation, create new tensors
        c_mat = self.get_containing_mat
        c_mask_num_before = self.c_mask_num
        global2merged_mask_id = torch.tensor(self.global2merged_mask_id, dtype=torch.long)  # Tensor(collected_global_mask_num, )
        global2merged_mask_id_new = -1 * torch.ones_like(global2merged_mask_id)
        mask_voxels_coords_new = {}
        mask_features_new = torch.zeros((c_mask_num_before, self.cfg["mask"]["feature_dim"]), dtype=torch.float32, device=self.device)
        mask_geo_features_new = torch.zeros((c_mask_num_before, self.cfg["pc_extractor"]["feature_dim"]), dtype=torch.float32, device=self.device)
        mask_geo_features_mask_new = torch.zeros((c_mask_num_before,), dtype=torch.float32, device=self.device)  # whether each merged mask's geometric feature has been extracted successfully
        merged_mask_size_new = []
        merged_mask_weight_new = torch.zeros((self.t_mask_num,), dtype=torch.float32, device=self.device)
        merged_mask_last_frame_new = -1 * torch.ones((self.t_mask_num,), dtype=torch.int64, device=self.device)

        # 3.2: for each mask needs to be removed, find their raw (/global) masks, and set their mapping mask_IDs to -1
        r_global_mask_ids = torch.where( torch.isin(global2merged_mask_id, remove_mask_ids.to(global2merged_mask_id)) )[0]
        global2merged_mask_id_new[r_global_mask_ids] = -1

        # 3.3: re-assign merged_mask_id to each kept mask, and copy their corresponding rows & cols from current c_mat to new c_mat
        c_mat_new = extract_rows_and_cols(c_mat, kept_mask_ids)

        new_mask_counter = 0
        kept_mask_ids = kept_mask_ids.tolist()
        for i, mask_id_kept in enumerate(kept_mask_ids):
            corr_global_mask_ids = torch.where(global2merged_mask_id == mask_id_kept)[0]
            global2merged_mask_id_new[corr_global_mask_ids] = new_mask_counter

            mask_voxels_coords_new[new_mask_counter] = self.mask_voxels_coords[mask_id_kept]
            mask_features_new[new_mask_counter] = self.get_mask_features[mask_id_kept]
            mask_geo_features_new[new_mask_counter] = self.get_mask_geo_features[mask_id_kept]
            mask_geo_features_mask_new[new_mask_counter] = self.get_mask_geo_features_mask[mask_id_kept]
            merged_mask_size_new.append(self.merged_mask_size[mask_id_kept])
            merged_mask_weight_new[new_mask_counter] = self.get_merged_mask_weight[mask_id_kept]  # set as former weight value rather than 1
            merged_mask_last_frame_new[new_mask_counter] = self.get_merged_mask_last_frame[mask_id_kept]
            new_mask_counter += 1

        # 3.4: set new tensors as members
        self.global2merged_mask_id = global2merged_mask_id_new.tolist()
        self.contain_matrix[:self.c_mask_num, :self.c_mask_num] = torch.eye(self.c_mask_num).to(c_mat_new)
        self.contain_matrix[:kept_mask_num, :kept_mask_num] = c_mat_new
        self.c_mask_num = kept_mask_num
        self.mask_voxels_coords = mask_voxels_coords_new
        self.mask_features[:c_mask_num_before] = mask_features_new
        self.mask_geo_features[:c_mask_num_before] = mask_geo_features_new
        self.mask_geo_features_mask[:c_mask_num_before] = mask_geo_features_mask_new
        self.merged_mask_size = merged_mask_size_new
        self.merged_mask_weight = merged_mask_weight_new
        self.merged_mask_last_frame = merged_mask_last_frame_new


    # @brief: save latest segmentation results
    def save_merging_result(self, frame_id, valid_merged_mask_ids=None, reextract=True):
        # Step 1: get so-far reconstructed pointcloud and mesh
        if reextract and self.pc_frame_id != frame_id:
            self.points, self.colors = self.get_pc()
            self.pc_frame_id = frame_id

        # Step 2: save segmented pointcloud
        if valid_merged_mask_ids is None:
            mask_weight_threshold = min(self.merge_time + 2, self.cfg["seg"]["mask_weight_threshold"])
            valid_merged_mask_ids = torch.where(self.merged_mask_weight[:self.c_mask_num] >= mask_weight_threshold)[0]

        mask_coords_list = [self.mask_voxels_coords[i] for i in valid_merged_mask_ids.tolist()]
        inst_rgb_list = self.maskGraph.get_instance_rgb(valid_merged_mask_ids.tolist())
        seg_output_path = os.path.join(self.seq_output_dir, "seg_%d.ply" % frame_id)

        # Step 3: save reconstructed and segmented mesh or pointcloud
        if self.cfg["save"]["save_mesh"]:  # for mesh
            ckpt_save_dir = os.path.join(self.seq_output_dir, "ckpt_%d" % frame_id)
            os.makedirs(ckpt_save_dir, exist_ok=True)

            recon_mesh_save_path = os.path.join(ckpt_save_dir, "recon_mesh_%d.ply" % frame_id)
            vertices, recon_mesh = self.get_mesh_vertices(save_path=recon_mesh_save_path)

            instance_mask_save_path = os.path.join(ckpt_save_dir, "pred_instance_mask_%d.pt" % frame_id)

            self.get_seg_mesh(mask_coords_list, recon_mesh, seg_output_path, instance_mask_save_path, inst_rgb_list)
        else:  # for pointcloud
            self.get_seg_pc_w_overlap(frame_id, mask_coords_list, self.points, seg_output_path, inst_rgb_list)


    ############################################## Helper functions for visualization ##############################################

    # @brief: process boundary points for all predicted instance masks (each boundary point will be only assigned to the instance with minimal mask size);
    # @param pred_inst_masks: Tensor(pred_inst_num, pts_num), dtype=bool;
    #-@return: Tensor(pred_inst_num, pts_num), dtype=bool.
    def process_mask_boundary_pts(self, pred_inst_masks, return_list=False):
        if isinstance(pred_inst_masks, list):
            pred_inst_masks = torch.stack(pred_inst_masks, dim=0)

        # Step 1: sort each predicted instance by mask size, with ascending order
        pred_inst_size = torch.sum(pred_inst_masks, dim=-1)
        pred_inst_idx_asc = torch.argsort(pred_inst_size)
        pred_inst_masks = pred_inst_masks[pred_inst_idx_asc]  # predicted instances' masks (sort by mask size, ascending order), Tensor(pred_inst_num, n), dtype=bool

        # Step 2: for each boundary points, it will be only assigned to 1 pred instance (with minimal mask size)
        pred_inst_masks_new = keep_min_rows(pred_inst_masks)
        if return_list:
            pred_inst_masks_new = [inst_mask for inst_mask in pred_inst_masks]

        return pred_inst_masks_new

    # @brief: do Marching Cubes to get reconstruction result (extract vertices of reconstructed mesh as pointcloud)
    #-@return points: ndarray(vert_num, 3);
    #-@return colors: ndarray(vert_num, 3).
    def get_pc(self):
        pcd = self.voxel_block_grids.extract_point_cloud()
        points = pcd.point.positions.cpu().numpy()
        colors = pcd.point.colors.cpu().numpy()
        return points, colors


    # @brief: for given scene pointcloud and each instance, compute each instance's point mask over all scene points;
    # @param instance_voxels_coords: list Tensor(n_i, 3);
    # @param scene_points: ndarray(scene_pts_num, 3);
    #-@return: list of Tensor(scene_pts_num, ), dtype=bool.
    def get_instance_scene_pts_mask(self, instance_voxels_coords, scene_points):
        if isinstance(scene_points, np.ndarray):
            scene_points = torch.from_numpy(scene_points).to(self.device)

        # Step 1: for each extracted scene point, compute its corresponding voxel
        voxelized_scene_points = self.voxel_grids.world_coords2voxel_coords(scene_points).to(self.device)
        scene_point_voxel_indices = self.voxel_grids.voxel_coords2voxel_indices(voxelized_scene_points)

        num_instances = len(instance_voxels_coords)
        instance_pts_mask_list = []
        for idx in range(num_instances):
            mask_voxel_coords = instance_voxels_coords[idx]  # Voxel coordinates of this mask, Tensor(m_i, 3)
            mask_voxel_indices = self.voxel_grids.voxel_coords2voxel_indices(mask_voxel_coords)
            instance_pts_mask = torch.isin(scene_point_voxel_indices, mask_voxel_indices)
            instance_pts_mask_list.append(instance_pts_mask)
        return instance_pts_mask_list


    def get_seg_mesh(self, mask_voxels_coords, input_mesh, save_path=None, instance_mask_save_path=None, rgb_list=None):
        scene_points = np.asarray(input_mesh.vertices).astype("float32")
        scene_points = torch.from_numpy(scene_points).to(self.device)

        # Step 1: for each extracted scene point, compute its corresponding voxel
        voxelized_scene_points = self.voxel_grids.world_coords2voxel_coords(scene_points).to(self.device)
        point_voxel_indices = self.voxel_grids.voxel_coords2voxel_indices(voxelized_scene_points)

        # Step 2: for each detected valid mask, compute its corresponding points in scene points
        instance_colors = 200. * torch.ones_like(scene_points)

        num_instances = len(mask_voxels_coords)
        scene_point_mask_list = []
        for idx in range(num_instances):
            mask_voxel_coords = mask_voxels_coords[idx]  # Voxel coordinates of this mask, Tensor(m_i, 3)
            mask_voxel_indices = self.voxel_grids.voxel_coords2voxel_indices(mask_voxel_coords)
            scene_point_mask = torch.isin(point_voxel_indices, mask_voxel_indices)
            corr_point_ids = torch.where(scene_point_mask)[0]
            scene_point_mask_list.append(scene_point_mask)

            point_ids, points, colors, label_color, center = vis_one_object(corr_point_ids, scene_points)
            if rgb_list is None:
                instance_colors[point_ids] = label_color.to(self.device)
            else:
                instance_colors[point_ids] = rgb_list[idx].to(self.device)

        instance_colors = instance_colors.cpu().numpy().astype("float64") / 255.
        instance_colors = adjust_colors_to_pastel(instance_colors)  # adjust color brightness
        input_mesh.vertex_colors = o3d.utility.Vector3dVector(instance_colors)

        if save_path is not None:
            o3d.io.write_triangle_mesh(save_path, input_mesh)
        if instance_mask_save_path is not None:
            torch.save(scene_point_mask_list, instance_mask_save_path)
        return input_mesh


    # @brief: for each point in currently reconstructed pc, paint it with different color according to its instance ID;
    #   *** boundary points only assigned to one instance (with minimal size);
    # @param mask_voxels_coords: voxel list of all valid instances, list of Tensor(v_i, 3);
    # @param scene_points: latest reconstructed scene points, Tensor(n, 3);
    #-@return: segmentation pointcloud(colored by instance_ID), PointCloud obj;
    def get_seg_pc_w_overlap(self, frame_id, mask_voxels_coords, scene_points=None, save_path=None, rgb_list=None):
        if scene_points is None:
            scene_points, _ = self.get_pc()
        if isinstance(scene_points, np.ndarray):
            scene_points = torch.from_numpy(scene_points).to(self.device)

        # Step 1: for each extracted scene point, compute its corresponding voxel
        voxelized_scene_points = self.voxel_grids.world_coords2voxel_coords(scene_points).to(self.device)
        point_voxel_indices = self.voxel_grids.voxel_coords2voxel_indices(voxelized_scene_points)

        # Step 2: get each predicted instance's point mask
        num_instances = len(mask_voxels_coords)
        scene_pts_mask_list = []  # list of Tensor(n, ), dtype=bool;
        for idx in range(num_instances):
            mask_voxel_coords = mask_voxels_coords[idx]  # Voxel coordinates of this mask, Tensor(m_i, 3)
            mask_voxel_indices = self.voxel_grids.voxel_coords2voxel_indices(mask_voxel_coords)  # corresponding voxels of this 3D mask

            scene_point_mask = torch.isin(point_voxel_indices, mask_voxel_indices)
            scene_pts_mask_list.append(scene_point_mask)

        # *** Step 3: process boundary points
        pred_inst_masks = self.process_mask_boundary_pts(scene_pts_mask_list)
        self.pred_inst_mask_frame_id = frame_id
        self.pred_inst_mask = pred_inst_masks

        # Step 4: paint each instance's corresponding scene points
        scene_colors = torch.zeros_like(scene_points)
        scene_colors = torch.pow(scene_colors, 1 / 2.2)
        scene_colors = scene_colors * 255
        instance_colors = 200. * torch.ones_like(scene_colors)  # set background to gray

        for idx in range(num_instances):
            scene_point_mask = pred_inst_masks[idx]
            corr_point_ids = torch.where(scene_point_mask)[0]

            point_ids, points, colors, label_color, center = vis_one_object(corr_point_ids, scene_points)
            if rgb_list is None:
                instance_colors[point_ids] = label_color.to(self.device)
            else:
                instance_colors[point_ids] = rgb_list[idx].to(self.device)
        # END for

        scene_points = scene_points.cpu().numpy()
        instance_colors = instance_colors.cpu().numpy() / 255.
        instance_colors = adjust_colors_to_pastel(instance_colors)  # adjust color brightness

        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(scene_points)
        pc.colors = o3d.utility.Vector3dVector(instance_colors)

        if save_path is not None:
            o3d.io.write_point_cloud(save_path, pc)
        return pc


    # @brief: save reconstructed pointcloud(with raw colors)
    def save_pc(self, points, colors, save_path=None):
        if isinstance(points, torch.Tensor):
            points = points.cpu().numpy()
        if isinstance(colors, torch.Tensor):
            colors = colors.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            o3d.io.write_point_cloud(save_path, pcd)
        return pcd


    # @brief: get current reconstructed mesh
    def get_mesh_vertices(self, legacy=True, save_path=None):
        try:
            mesh = self.voxel_block_grids.extract_triangle_mesh()
        except RuntimeError as re:
            print(3)

        final_mesh = mesh.to_legacy() if legacy else mesh
        if save_path is not None:
            o3d.io.write_triangle_mesh(save_path, final_mesh)
        vertices = np.asarray(final_mesh.vertices)
        return vertices, final_mesh


    # @brief: Get all current valid instances;
    #-@return instance_mask_list: each instance's point mask in given pc, list of Tensor(pc_num, ), dtype=bool;
    #-@return valid_merged_mask_ids:
    #-@return mask_voxel_coords_list: each instance's voxel coordinate set, list of Tensor(mask_voxel_size, 3), dtype=float32;
    #-@return sem_featurelist: each instance's fused semantic feature (normalized), list of Tensor(feature_dim, ), dtype=float32;
    #-@return mask_scene_pts_coords_list: each instance's corresponding scene points, list of Tensor(mask_voxel_size, 3), dtype=float32;
    def get_valid_instances(self, scene_points=None, max_threshold=5, min_size=25):
        # Step 1: extract Voxel coordinates for each valid instance
        mask_weight_threshold = min(self.merge_time + 2, self.cfg["seg"]["mask_weight_threshold"])
        valid_merged_mask_ids = torch.where(self.merged_mask_weight[:self.c_mask_num] >= mask_weight_threshold)[0]
        mask_voxel_coords_list = [self.mask_voxels_coords[i] for i in valid_merged_mask_ids.tolist()]
        valid_instances = [instance for inst_id, instance in self.maskGraph.instance_dict.items() if inst_id in valid_merged_mask_ids.tolist()]

        # Step 2: for each extracted scene point, compute its corresponding voxel
        if scene_points is None:
            scene_points, _ = self.get_pc()  # Tensor(pts_num, 3)
        if isinstance(scene_points, np.ndarray):
            scene_points = torch.from_numpy(scene_points).to(self.device)

        inst_scene_pts_mask_list = self.get_instance_scene_pts_mask(mask_voxel_coords_list, scene_points)

        # Step 3: for each valid mask, compute its corresponding scene points
        instance_mask_list = []
        num_instances = len(mask_voxel_coords_list)
        mask_scene_pts_coords_list = []
        final_valid_idx = []
        for idx in range(num_instances):
            inst_scene_pts_mask = inst_scene_pts_mask_list[idx]
            corr_point_ids = torch.where(inst_scene_pts_mask)[0]  # corresponding point_ids of this valid instance (on given pc)
            if corr_point_ids.shape[0] < min_size:
                continue

            instance_mask_list.append(inst_scene_pts_mask)
            corr_scene_pts_coords = scene_points[corr_point_ids]
            mask_scene_pts_coords_list.append(corr_scene_pts_coords)
            final_valid_idx.append(idx)

        # Step 4: save semantic feature of each valid instance
        sem_feature_list = [instance.get_semantic_feature for instance in valid_instances]

        # Step 5: keep only info of final valid instances
        valid_merged_mask_ids = valid_merged_mask_ids[final_valid_idx]
        mask_voxel_coords_list = [mask_voxel_coords_list[idx] for idx in range(num_instances) if idx in final_valid_idx]
        sem_feature_list = [sem_feature_list[idx] for idx in range(num_instances) if idx in final_valid_idx]

        return instance_mask_list, valid_merged_mask_ids, mask_voxel_coords_list, sem_feature_list, mask_scene_pts_coords_list


    # @brief: save latest reconstructed results and per-instance segmentation results
    def save_ckpt(self, frame_id, ckpt_save_dir, filter_flag=True, ckpt_path=None):
        # Step 0: preparation
        if not os.path.isdir(ckpt_save_dir):
            return
        os.makedirs(ckpt_save_dir, exist_ok=True)

        # Step 1: get current reconstructed pointcloud
        if frame_id != self.pc_frame_id:
            self.points, self.colors = self.get_pc()
            self.pc_frame_id = frame_id

        # Step 2: get current detected 3D instances info
        # 2.1: get each pred instance's semantic feature, mask
        instance_mask_list, valid_merged_mask_ids, mask_voxel_coords_list, sem_feature_list, mask_scene_pts_coords_list = self.get_valid_instances(scene_points=self.points)
        instance_feature_list = [instance_sem_feature.cpu().numpy() for instance_sem_feature in sem_feature_list]

        # 2.2: for each pred instance mask, process boundary points
        instance_mask_list = self.process_mask_boundary_pts(instance_mask_list, return_list=True)
        instance_mask_list = [instance_mask.cpu().numpy() for instance_mask in instance_mask_list]


        # # Step 3: for each pred 3D instance, apply DBSCAN for filtering
        instance_mask_list_filtered, instance_feature_list_filtered, valid_instance_indices = filter_instances(self.points, instance_mask_list, instance_feature_list)

        # Step 4: save current reconstruction result
        recon_pc_path = os.path.join(ckpt_save_dir, "recon_%d.ply" % frame_id)
        self.save_pc(self.points, self.colors, recon_pc_path)

        if ckpt_path is None:
            ckpt_path = os.path.join(ckpt_save_dir, "ckpt_%d.npz" % frame_id)

        if filter_flag:
            export_instance_mask(ckpt_path, instance_mask_list_filtered, instance_feature_list_filtered)  # with DBSCAN
            return instance_mask_list_filtered
        else:
            export_instance_mask(ckpt_path, instance_mask_list, instance_feature_list)  # without DBSCAN
            return instance_mask_list
