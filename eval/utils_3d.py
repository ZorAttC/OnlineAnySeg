import os
import json
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import open3d as o3d
from scipy.spatial import cKDTree as KDTree

from tool.geometric_helpers import crop_scene_points, query_neighbors_mt
from eval.semantic_helpers import compute_label_id_by_sim, compute_label_id_by_sim2
from eval.constants import MATTERPORT_LABELS, MATTERPORT_IDS, SCANNET_LABELS, SCANNET_IDS, SCANNETPP_LABELS, SCANNETPP_IDS


def load_ids(filename):
    ids = open(filename).read().splitlines()
    ids = np.array(ids, dtype=np.int64)
    return ids

# ------------ Instance Utils ------------ #

class Instance(object):
    instance_id = 0
    label_id = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, mesh_vert_instances, instance_id):
        if (instance_id == -1):
            return
        self.instance_id = int(instance_id)
        self.label_id = int(self.get_label_id(instance_id))
        self.vert_count = int(self.get_instance_verts(mesh_vert_instances, instance_id))

    def get_label_id(self, instance_id):
        return int(instance_id // 1000)

    def get_instance_verts(self, mesh_vert_instances, instance_id):
        return (mesh_vert_instances == instance_id).sum()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_dict(self):
        dict = {}
        dict["instance_id"] = self.instance_id
        dict["label_id"]    = self.label_id
        dict["vert_count"]  = self.vert_count
        dict["med_dist"]    = self.med_dist
        dict["dist_conf"]   = self.dist_conf
        return dict

    def from_json(self, data):
        self.instance_id     = int(data["instance_id"])
        self.label_id        = int(data["label_id"])
        self.vert_count      = int(data["vert_count"])
        if ("med_dist" in data):
            self.med_dist    = float(data["med_dist"])
            self.dist_conf   = float(data["dist_conf"])

    def __str__(self):
        return "("+str(self.instance_id)+")"
# END class Instance


# @brief: get GT instances on GT pc;
# @param ids: GT result --- label_instance_ID for each GT point, ndarray(pts_num, );
# @param class_ids: list of int;
# @param class_labels: list of str;
# @param id2label: dict, key is class_ID, value is class_label;
def get_instances(ids, class_ids, class_labels, id2label, min_size=-1):
    instances = {}
    for label in class_labels:
        instances[label] = []

    instance_ids = np.unique(ids)

    gt_inst_counter = 0
    for id in instance_ids:
        if id // 1000 == 0:
            continue
        inst = Instance(ids, id)

        # ignore those small GT instances (if necessary)
        if min_size > 0 and inst.vert_count < min_size:
            continue

        gt_inst_counter += 1
        if inst.label_id in class_ids:
            inst_label = id2label[inst.label_id]
            instances[inst_label].append(inst.to_dict())

    # print("GT instance num: %d" % gt_inst_counter)
    return instances


def pc_align(pc1, pc2, threshold=0.05):
    trans_init = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pc1, pc2, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint() )
    return reg_p2p.transformation


# @brief: compute point correspondences from reconstructed pointcloud to GT pointcloud;
# @param recon_pc: PointCloud obj (r_num, );
# @param gt_pc: PointCloud obj (gt_num, );
#-@return corre_pt: In recon_pc, each point's correspnding_point_index in GT_pc (-1 means having no corresponding points), ndarray(r_num, ), dtype=int64.
#-@return valid_pt_indices: point indices of valid points in recon PC, ndarray(v_pts_num, )
def align_recon_pc_to_gt(recon_pc, gt_pc, fine_trans=True):
    # Step 1: align recon_pointcloud to GT_pointcloud
    if fine_trans:
        rel_trans = pc_align(recon_pc, gt_pc)
        recon_pc.transform(rel_trans)

    # Step 2: for each point in recon_pointcloud, find its corresponding point in GT_pointcloud by distance
    rec_points = np.asarray(recon_pc.points).astype("float32")
    gt_points = np.asarray(gt_pc.points).astype("float32")

    # 2.1: find valid nearest neighbor for each point in recon_pc
    gt_points_kd_tree = KDTree(gt_points)
    distances, corre_pt = gt_points_kd_tree.query(rec_points, distance_upper_bound=0.1)

    # 2.2: filter out points in recon_pc that have no corresponding points in GT_pc
    invalid_pt_indices = np.where(np.isinf(distances))[0]  # ndarray(I, )
    valid_pt_indices = np.where(~np.isinf(distances))[0]  # ndarray(V, )
    corre_pt[invalid_pt_indices] = -1
    return corre_pt, valid_pt_indices


# @brief: compute point correspondences from reconstructed pointcloud to GT pointcloud;
# @param recon_pc: PointCloud obj (r_num, );
# @param gt_pc: PointCloud obj (gt_num, );
#-@return corre_pt: In recon_pc, each point's correspnding_point_index in GT_pc (-1 means having no corresponding points), ndarray(r_num, ), dtype=int64.
#-@return valid_pt_indices: point indices of valid points in recon PC, ndarray(v_pts_num, )
def crop_gt_by_recon(recon_pc, gt_pc, fine_trans=True):
    # Step 1: align recon_pointcloud to GT_pointcloud
    if fine_trans:
        rel_trans = pc_align(recon_pc, gt_pc)
        recon_pc.transform(rel_trans)

    # Step 2: for each point in GT pointcloud, find its corresponding point in recon pointcloud by distance
    rec_points = np.asarray(recon_pc.points).astype("float32")
    gt_points = np.asarray(gt_pc.points).astype("float32")

    # 2.1: for each point in GT pc, find valid nearest neighbor for each point in recon pc
    recon_points_kd_tree = KDTree(rec_points)
    distances, corre_pt_in_recon = recon_points_kd_tree.query(gt_points, distance_upper_bound=0.1)

    # 2.2: filter out points in GT pc that have no corresponding points in recon pc
    invalid_pt_indices = np.where(np.isinf(distances))[0]  # ndarray(I, )
    valid_gt_pt_indices = np.where(~np.isinf(distances))[0]  # ndarray(V, )
    corre_pt_in_recon[invalid_pt_indices] = -1
    corpped_gt_pc = gt_pc.select_by_index(valid_gt_pt_indices)
    return corpped_gt_pc, valid_gt_pt_indices


# @brief: compute point correspondences from GT pc to reconstructed pc;
# @param gt_pc: PointCloud obj (gt_num, );
# @param recon_pc: PointCloud obj (r_num, );
#-@return corre_pt: In GT_pc, each point's correspnding_point_index in recon_pc (-1 means having no corresponding points), ndarray(r_num, ), dtype=int64.
#-@return valid_pt_indices: point indices of valid points in GT_pc, ndarray(v_pts_num, )
def align_gt_to_recon(gt_pc, recon_pc, fine_trans=True, valid_recon_pts_mask=None, distance_upper_bound=0.25):
    # Step 1: align recon_pointcloud to GT_pointcloud
    if fine_trans:
        rel_trans = pc_align(gt_pc, recon_pc)
        gt_pc.transform(rel_trans)

    # Step 2: for each point in GT pointcloud, find its corresponding point in recon_pointcloud by distance
    gt_points = np.asarray(gt_pc.points).astype("float32")
    recon_points = np.asarray(recon_pc.points).astype("float32")
    if valid_recon_pts_mask is not None:
        recon_points = recon_points[valid_recon_pts_mask]
        recon_pts_r2o = np.where(valid_recon_pts_mask)[0]

    # 2.1: find valid nearest neighbor for each point in GT_pc
    recon_points_kd_tree = KDTree(recon_points)
    distances, corr_pt = recon_points_kd_tree.query(gt_points, distance_upper_bound=distance_upper_bound)

    # 2.2: filter out points in recon_pc that have no corresponding points in GT_pc
    invalid_pt_indices = np.where(np.isinf(distances))[0]  # ndarray(I, )
    valid_pt_indices = np.where(~np.isinf(distances))[0]  # ndarray(V, )
    corr_pt[invalid_pt_indices] = -1

    if valid_recon_pts_mask is not None:
        corr_pt_2o = np.where(corr_pt > -1, recon_pts_r2o[corr_pt], -1)
        corr_pt = corr_pt_2o
    return corr_pt, valid_pt_indices


# @brief: map GT instances from GT pointcloud to reconstructed pointcloud
# @param ids: GT instance_IDs for each point in GT pc, ndarray(GT_pts_num, );
# @param corr_pt_in_recon: corresponding point_id in GT pc for each point in recon pc, ndarray(recon_pts_num, )
# @param valid_recon_pt_indices:
def get_instances_in_recon_pc(ids, corr_pt_in_recon, valid_recon_pt_indices, class_ids, class_labels, id2label):
    # assign ID for each point in recon_pc by their correspondences to GT pc
    recon_ids = np.zeros((corr_pt_in_recon.shape[0], ), dtype=ids.dtype)  # GT label of each point on recon pc
    valid_recon_pts_indices = (corr_pt_in_recon > -1)
    valid_recon_pts_corr = corr_pt_in_recon[valid_recon_pts_indices]
    recon_ids[valid_recon_pts_indices] = ids[valid_recon_pts_corr]

    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(recon_ids)  # ndarray(instance_num, )

    for id in instance_ids:
        if id == 0:
            continue
        inst = Instance(recon_ids, id)
        if inst.label_id in class_ids:
            instances[id2label[inst.label_id]].append(inst.to_dict())
    return instances, recon_ids


# @brief: for each pred instance, map it to GT pointcloud, and get the corresponding mask on GT pc;
# @param pred_info: pred segmentation result on recon pc, dict;
# @param corr_pt_in_recon: each GT point's corresponding point_ID on recon pc, ndarray(gt_pts_num, );
# @param valid_gt_pt_indices: indices of valid points on GT pc(e.g. having correspondence on recon pc), ndarray(v_gt_pts_num, );
# @param label_features: text embeddings of each label, ndarray(label_num, feat_dim).
def get_instances_in_GT_pc(pred_info, corr_pt_in_recon, valid_gt_pt_indices=None, valid_ratio_thresh=0.75, min_ovlp_thresh=100):
    pred_info_new = {}
    gt_pts_num = corr_pt_in_recon.shape[0]
    valid_pred_inst_indices = []
    for i, (key, pred_mask_file) in enumerate(pred_info.items()):
        pred_inst_pts_mask = pred_mask_file["mask"]  # mask of this pred instance on reconstructed pc, ndarray(recon_pts_num, ), dtype=bool
        pred_inst_size = np.sum(pred_inst_pts_mask)

        pred_inst_gt_pts_mask = np.zeros((gt_pts_num, ), dtype="bool")
        if valid_gt_pt_indices is not None:
            pred_inst_gt_pts_mask[valid_gt_pt_indices] = pred_inst_pts_mask[corr_pt_in_recon][valid_gt_pt_indices]
        else:
            pred_inst_gt_pts_mask = pred_inst_pts_mask[corr_pt_in_recon]

        pred_inst_gt_size = np.sum(pred_inst_gt_pts_mask)
        valid_ratio = pred_inst_gt_size / pred_inst_size

        if valid_ratio >= valid_ratio_thresh or pred_inst_gt_size > min_ovlp_thresh:
            pred_mask_file['mask'] = pred_inst_gt_pts_mask
            pred_info_new[key] = pred_mask_file
            valid_pred_inst_indices.append(i)

    pred_inst_num_bf = len(pred_info)
    pred_inst_num_aft = len(pred_info_new)
    print("Filter predicted instances: %d --> %d predicted instances" % (pred_inst_num_bf, pred_inst_num_aft))
    return pred_info_new


# @brief: for each pred instance, map it to GT pointcloud, and get the corresponding mask on GT pc;
# @param pred_info: pred segmentation result on recon pc, dict;
# @param corr_pt_in_recon: each GT point's corresponding point_ID on recon pc, ndarray(gt_pts_num, );
# @param valid_gt_pt_indices: indices of valid points on GT pc(e.g. having correspondence on recon pc), ndarray(v_gt_pts_num, );
# @param label_features: text embeddings of each label, ndarray(label_num, feat_dim).
def get_instances_in_GT_pc_seman(pred_info, corr_pt_in_recon, valid_gt_pt_indices=None, valid_ratio_thresh=0.75, min_ovlp_thresh=100,
                                 no_class_flag=True, label_features=None, valid_class_indices=None, class_labels=None, class_ids=None):
    # Step 1: compute similarity matrix between features of each pred instances and text embeddings of each label
    if label_features is not None:
        inst_features = [pred_mask_file["sem_feature"] for key, pred_mask_file in pred_info.items()]
        inst_features = np.stack(inst_features, axis=0)  # semantic feature of each predicted instance
        kept_inst_indices, inst_label_ids, inst_label_names = compute_label_id_by_sim2(inst_features, label_features, class_labels, class_ids, valid_class_indices=valid_class_indices)
    else:
        inst_label_ids, inst_label_names, kept_inst_indices, inst_label_ids, inst_label_names = [], [], [], [], []

    # Step 2: for each pred instance, map it to GT pointcloud, and get the corresponding mask on GT pc;
    pred_info_new = {}
    gt_pts_num = corr_pt_in_recon.shape[0]
    valid_pred_inst_indices = []
    for i, (key, pred_mask_file) in enumerate(pred_info.items()):
        if label_features is not None and i not in kept_inst_indices:
            continue

        pred_inst_pts_mask = pred_mask_file["mask"]  # ndarray(recon_pts_num, ), dtype=bool
        pred_inst_size = np.sum(pred_inst_pts_mask)
        pred_inst_gt_pts_mask = np.zeros((gt_pts_num, ), dtype="bool")

        if valid_gt_pt_indices is not None:
            pred_inst_gt_pts_mask[valid_gt_pt_indices] = pred_inst_pts_mask[corr_pt_in_recon][valid_gt_pt_indices]
        else:
            pred_inst_gt_pts_mask = pred_inst_pts_mask[corr_pt_in_recon]

        pred_inst_gt_size = np.sum(pred_inst_gt_pts_mask)
        valid_ratio = pred_inst_gt_size / pred_inst_size

        if valid_ratio >= valid_ratio_thresh or pred_inst_gt_size > min_ovlp_thresh:
            if no_class_flag == False and len(inst_label_ids) > 0:
                pred_mask_file['label_id'] = inst_label_ids[i]
                pred_mask_file['label_name'] = inst_label_names[i]
            pred_mask_file['mask'] = pred_inst_gt_pts_mask
            pred_info_new[key] = pred_mask_file
            valid_pred_inst_indices.append(i)

    valid_inst_label_ids = [inst_label_id for i, inst_label_id in enumerate(inst_label_ids) if i in valid_pred_inst_indices]
    valid_inst_label_names = [inst_label_name for i, inst_label_name in enumerate(inst_label_names) if i in valid_pred_inst_indices]

    pred_inst_num_bf = len(pred_info)
    pred_inst_num_aft = len(pred_info_new)
    print("Filter predicted instances: %d --> %d predicted instances" % (pred_inst_num_bf, pred_inst_num_aft))
    return pred_info_new


# @brief: for each predicted instance, if the majority of this instance falls beyond the valid range of reconstructed pc, it will be removed.
def filter_out_pred_instances(pred_info, valid_recon_pt_mask, valid_ratio_thresh=0.75, min_ovlp_thresh=100):
    pred_info_new = {}
    for key, pred_mask_file in pred_info.items():
        pred_inst_pts_mask = pred_mask_file["mask"]  # ndarray(recon_pts_num, ), dtype=bool
        pred_inst_size = np.sum(pred_inst_pts_mask)

        ovlp_pts_mask = (valid_recon_pt_mask & pred_inst_pts_mask)
        ovlp_size = np.sum(ovlp_pts_mask)

        valid_ratio = ovlp_size / pred_inst_size

        if valid_ratio >= valid_ratio_thresh:
            pred_inst_pts_mask_valid = pred_inst_pts_mask[valid_recon_pt_mask]
            pred_mask_file['mask'] = pred_inst_pts_mask_valid
            pred_info_new[key] = pred_mask_file

    pred_inst_num_bf = len(pred_info)
    pred_inst_num_aft = len(pred_info_new)
    print("Filter predicted instances: %d --> %d predicted instances" % (pred_inst_num_bf, pred_inst_num_aft))
    return pred_info_new


# @brief: map predicted instances from recon pc to GT pc
def visualize_pred_instances_in_gt_pc(gt_pc, pred_info_gt, save_path=None, bg_color=[211., 211., 211.]):
    gt_points = np.asarray(gt_pc.points).astype("float32")
    scene_colors = np.zeros_like(gt_points)
    scene_colors = np.power(scene_colors, 1.2 / 2.2)
    scene_colors = scene_colors * 255
    instance_colors = np.zeros_like(scene_colors)

    pred_inst_mask_list = []
    pred_inst_color_list = []
    for key, pred_mask_file in pred_info_gt.items():
        pred_inst_pts_mask = pred_mask_file[ "mask"]  # the corresponding mask on GT pc of this pred instance, ndarray(gt_pts_num, ), dtype=bool
        instance_point_ids = np.where(pred_inst_pts_mask)[0]
        pred_inst_mask_list.append(pred_inst_pts_mask)

        point_ids, points, colors, label_color, center = vis_one_object(instance_point_ids, gt_points)
        instance_colors[point_ids] = label_color
        pred_inst_color_list.append(label_color)

    bg_color = np.array(bg_color)
    pred_inst_masks = np.stack(pred_inst_mask_list, axis=0)
    front_mask = np.any(pred_inst_masks, axis=0)
    bg_mask = ~front_mask
    instance_colors[bg_mask] = bg_color

    instance_colors_01 = adjust_colors_to_pastel(instance_colors / 255.)  # make all colors lighter
    points_colors = o3d.utility.Vector3dVector(instance_colors_01)
    gt_pc.colors = points_colors

    if save_path is not None:
        o3d.io.write_point_cloud(save_path, gt_pc)

        pred_inst_color_info = {
            "front_mask": front_mask,
            "pred_instance_masks": pred_inst_mask_list,
            "pred_instance_colors": pred_inst_color_list
        }
        pred_info_save_path = os.path.join( os.path.dirname(save_path), "pred_instance_info.npz" )
        np.savez(pred_info_save_path, **pred_inst_color_info)
    return gt_pc


# @brief: map predicted instances from recon pc to GT mesh
def visualize_pred_instances_in_gt_mesh(gt_mesh, pred_info_gt, save_path=None, bg_color=[211., 211., 211.], target_vertex_count=-500000):
    gt_points = np.asarray(gt_mesh.vertices).astype("float32")
    scene_colors = np.zeros_like(gt_points)
    scene_colors = np.power(scene_colors, 1.2 / 2.2)
    scene_colors = scene_colors * 255
    instance_colors = np.zeros_like(scene_colors)

    pred_inst_mask_list = []
    pred_inst_color_list = []
    for key, pred_mask_file in pred_info_gt.items():
        pred_inst_pts_mask = pred_mask_file["mask"]  # the corresponding mask on GT pc of this pred instance, ndarray(gt_pts_num, ), dtype=bool
        instance_point_ids = np.where(pred_inst_pts_mask)[0]
        pred_inst_mask_list.append(pred_inst_pts_mask)

        point_ids, points, colors, label_color, center = vis_one_object(instance_point_ids, gt_points)
        instance_colors[point_ids] = label_color
        pred_inst_color_list.append(label_color)

    bg_color = np.array(bg_color)
    pred_inst_masks = np.stack(pred_inst_mask_list, axis=0)
    front_mask = np.any(pred_inst_masks, axis=0)
    bg_mask = ~front_mask
    instance_colors[bg_mask] = bg_color

    instance_colors_01 = adjust_colors_to_pastel(instance_colors / 255.)  # make all colors lighter
    vert_colors = o3d.utility.Vector3dVector(instance_colors_01)
    gt_mesh.vertex_colors = vert_colors

    if target_vertex_count > 0:
        gt_mesh = gt_mesh.simplify_quadric_decimation(target_number_of_triangles=target_vertex_count)

    if save_path is not None:
        o3d.io.write_triangle_mesh(save_path, gt_mesh)

        pred_inst_color_info = {
            "front_mask": front_mask,
            "pred_instance_masks": pred_inst_mask_list,
            "pred_instance_colors": pred_inst_color_list
        }
        pred_info_save_path = os.path.join( os.path.dirname(save_path), "pred_instance_info.npz" )
        np.savez(pred_info_save_path, **pred_inst_color_info)

    return gt_mesh


# @brief: according to point correspondences of GT pc and recon pc, visualize all GT instances on recon pc.
# @param ids: GT instance_IDs for each point in GT pc, ndarray(GT_pts_num, );
# @param corr_pt_in_recon: corresponding point_id in GT pc for each point in recon pc, ndarray(recon_pts_num, )
# @param valid_recon_pt_indices:
def visualize_gt_instances_in_recon_pc(recon_points, ids, corr_pt_in_recon, class_labels, save_path=None):
    # Step 1: assign ID for each point in recon_pc by their correspondences to GT pc
    recon_ids = np.zeros((corr_pt_in_recon.shape[0], ), dtype=ids.dtype)
    valid_recon_pts_indices = (corr_pt_in_recon > -1)
    valid_recon_pts_corr = corr_pt_in_recon[valid_recon_pts_indices]
    recon_ids[valid_recon_pts_indices] = ids[valid_recon_pts_corr]

    # Step 2: preparation for visualize instances
    scene_colors = np.zeros_like(recon_points)
    scene_colors = np.power(scene_colors, 1 / 2.2)
    scene_colors = scene_colors * 255
    instance_colors = np.zeros_like(scene_colors)

    # Step 3: draw each detected instance
    instances = {}
    for label in class_labels:
        instances[label] = []
    instance_ids = np.unique(recon_ids)

    for id in instance_ids:
        if id == 0 or id % 1000 == 0:
            continue

        instance_pts_ids_recon = np.where(recon_ids == id)[0]
        point_ids, points, colors, label_color, center = vis_one_object(instance_pts_ids_recon, recon_points)
        instance_colors[point_ids] = label_color

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(recon_points)
    pc.colors = o3d.utility.Vector3dVector(instance_colors / 255.)

    if save_path is not None:
        o3d.io.write_point_cloud(save_path, pc)
    return pc


# @brief: adjust given RGB colors lighter
def adjust_colors_to_pastel(rgb_array, factor=0.75):
    pastel_rgb_array = rgb_array * factor + (1 - factor)
    return pastel_rgb_array


def vis_one_object(point_ids, scene_points):
    points = scene_points[point_ids]
    color = (np.random.rand(3) * 0.7 + 0.3) * 255
    colors = np.tile(color, (points.shape[0], 1))
    return point_ids, points, colors, color, np.mean(points, axis=0)


####################################### helper functions #######################################
def save_pc_uniform_color(pts, save_path=None, pc_color=[1., 0., 0.]):
    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.estimate_normals()
    pc.paint_uniform_color(np.array(pc_color).astype("float64"))

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        o3d.io.write_point_cloud(save_path, pc)
    return pc


# @brief:
# @param gt_pc: Pointcloud obj
# @param gt_instances_mask: GT instance masks, Tensor(point_num, GT_instance_num), dtype=bool;
def save_each_gt_instance(gt_pc, gt_instances_mask, save_dir):
    gt_pts = np.asarray(gt_pc.points)
    gt_pts = torch.from_numpy(gt_pts).float().to(gt_instances_mask.device)  # Tensor(point_num, 3)

    os.makedirs(save_dir, exist_ok=True)

    instance_num = gt_instances_mask.shape[-1]
    for i in range(instance_num):
        instance_mask = gt_instances_mask[:, i]  # Tensor(point_num, )
        instance_pts = gt_pts[instance_mask]

        instance_save_path = os.path.join( save_dir, "gt_%d.ply" % (i+1) )  # *** GT instance的ID从1开始计数
        save_pc_uniform_color(instance_pts, instance_save_path)


def save_each_pred_instance(gt_pc, pred_info, save_dir):
    gt_pts = np.asarray(gt_pc.points).astype("float32")  # ndarray(point_num, 3)
    os.makedirs(save_dir, exist_ok=True)

    for i, pred_mask_file in enumerate(pred_info):
        pred_instance_mask = pred_info[pred_mask_file]['mask']  # predicted mask of this instance(on GT pc), ndarray(point_num, ), dtype=bool
        instance_pts = gt_pts[pred_instance_mask]

        instance_save_path = os.path.join(save_dir, "pred_%d.ply" % i)
        save_pc_uniform_color(instance_pts, instance_save_path, [0., 0., 1.])


def draw_pred_gt_instance_intersect(gt_pc, gt_instances_mask, pred_info, gt_instance_id, pred_instance_id, save_dir):
    gt_pts = np.asarray(gt_pc.points).astype("float32")  # ndarray(point_num, 3)
    gt_pts = torch.from_numpy(gt_pts).to(gt_instances_mask.device)
    os.makedirs(save_dir, exist_ok=True)

    gt_instance_mask = gt_instances_mask[:, gt_instance_id-1]  # Tensor(point_num, ), dtype=bool
    gt_instance_size = torch.count_nonzero(gt_instance_mask).item()
    gt_instance_pc = gt_pts[gt_instance_mask]

    pred_mask_file = list(pred_info.keys())[pred_instance_id]
    pred_instance_mask = pred_info[pred_mask_file]['mask']  # predicted mask of this instance(on GT pc), ndarray(point_num, ), dtype=bool
    pred_instance_mask = torch.from_numpy(pred_instance_mask).to(gt_instance_mask)
    pred_instance_size = torch.count_nonzero(pred_instance_mask).item()
    pred_instance_pc = gt_pts[pred_instance_mask]

    intersect_mask = torch.logical_and(gt_instance_mask, pred_instance_mask)
    intersect_size = torch.count_nonzero(intersect_mask).item()
    intersect_pc = gt_pts[intersect_mask]

    save_pc_uniform_color(gt_instance_pc, os.path.join(save_dir, "gt_%d.ply" % gt_instance_id), [1., 0., 0.])
    save_pc_uniform_color(pred_instance_pc, os.path.join(save_dir, "pred_%d.ply" % pred_instance_id), [0., 0., 1.])
    save_pc_uniform_color(intersect_pc, os.path.join(save_dir, "intersect_%d_%d.ply" % (gt_instance_id, pred_instance_id)), [0., 1., 0.])

    IoU = float(intersect_size) / (gt_instance_size + pred_instance_size - intersect_size)
    print("GT instance %d and Pred instance %d: size_1=%d, size_2=%d, intersection size=%d, IoU=%.4f" % (gt_instance_id, pred_instance_id, gt_instance_size, pred_instance_size, intersect_size, IoU))

