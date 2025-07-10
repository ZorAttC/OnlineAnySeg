import os
import sys
import argparse
from copy import deepcopy
import numpy as np
import torch
import open3d as o3d
from natsort import natsorted

from eval.utils_3d import get_instances, align_gt_to_recon, get_instances_in_GT_pc, visualize_pred_instances_in_gt_pc, visualize_pred_instances_in_gt_mesh
from eval.constants import MATTERPORT_LABELS, MATTERPORT_IDS, SCANNET_LABELS, SCANNET_IDS, SCANNETPP_LABELS, SCANNETPP_IDS

np.random.seed(27)

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', required=True, help='directory to save the predicted segmentation results.')
parser.add_argument('--seq_name', default='', help='the seq_name of selected sequences, split by comma.')
parser.add_argument('--gt_dir', required=True, help='directory of GT data.')
parser.add_argument('--gt_pc_pattern', default="%s/%s_vh_clean_2.ply", help='path pattern of GT pointcloud.')
parser.add_argument('--gt_seg_dir', default="./eval/scannet200/validation", help='directory of GT segmentation.')
parser.add_argument('--gt_seg_pattern', default="%s.txt", help='path pattern of GT segmentation.')

parser.add_argument('--pred_path', default="./data/prediction/scannet_class_agnostic", help='path to directory of predicted .txt files')
parser.add_argument('--dataset', default="scannet", help='type of dataset, e.g. matterport3d, scannet, etc.')
parser.add_argument('--output_file', default='', help='path to output file')
parser.add_argument('--no_class', default=True, action='store_true', help='class agnostic evaluation')
opt = parser.parse_args()

# ---------- Label info ---------- #
if opt.dataset == 'matterport3d':
    CLASS_LABELS = MATTERPORT_LABELS
    VALID_CLASS_IDS = MATTERPORT_IDS
elif opt.dataset == 'scannet':
    CLASS_LABELS = SCANNET_LABELS
    VALID_CLASS_IDS = SCANNET_IDS
elif opt.dataset == 'scannetpp':
    CLASS_LABELS = SCANNETPP_LABELS
    VALID_CLASS_IDS = SCANNETPP_IDS

if opt.output_file == '':
    opt.output_file = os.path.join(f'data/evaluation/{opt.dataset}', opt.pred_path.split('/')[-1] + '.txt')
    os.makedirs(os.path.dirname(opt.output_file), exist_ok=True)
if opt.no_class:
    if 'class_agnostic' not in opt.output_file:
        opt.output_file = opt.output_file.replace('.txt', '_class_agnostic.txt')


ID_TO_LABEL = {}
LABEL_TO_ID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]


# ---------- Evaluation params ---------- #
# overlaps for evaluation
opt.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
# minimum region size for evaluation [verts]
opt.min_region_sizes = np.array([100])
# distance thresholds [m]
opt.distance_threshes = np.array([float('inf')])
# distance confidences
opt.distance_confs = np.array([-float('inf')])


def evaluate_matches(matches):
    overlaps = opt.overlaps  # different overlap thresholds to compute AP
    min_region_sizes = [opt.min_region_sizes[0]]
    dist_threshes = [opt.distance_threshes[0]]
    dist_confs = [opt.distance_confs[0]]

    # results: class x overlap
    ap = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float)
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):

            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]['pred'][label_name]:
                            if 'filename' in p:
                                pred_visited[p['filename']] = False

            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False

                # go through each seq
                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]  # *** predicted instances, dict
                    gt_instances = matches[m]['gt'][label_name]  # *** GT instances, dict

                    # filter groups in ground truth
                    gt_instances = [gt for gt in gt_instances if gt['instance_id'] >= 1000 and gt['vert_count'] >= min_region_size and gt['med_dist'] <= distance_thresh and gt['dist_conf'] >= distance_conf]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)  # whether each GT instance have matching predicted instance

                    # Step 1: collect matches (for each GT instance, find its corresponding predicted instance)
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False  # whether this GT instance has found corresponding predicted instance
                        num_pred = len(gt['matched_pred'])  # number of predicted instances that have overlap with it

                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['filename']]:
                                continue

                            # 1.1: compute IoU of this pred instance and current GT instance
                            pred_instance_id = pred['pred_id']
                            intersect_num = pred['intersection']
                            gt_instance_num = gt['vert_count']
                            pred_instance_num = pred['vert_count']
                            overlap = float(intersect_num) / (gt_instance_num + pred_instance_num - intersect_num)  # IoU

                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['filename']] = True
                        # END for each matched pred instance

                        if not found_match:
                            hard_false_negatives += 1
                    # END for each GT instance

                    # remove non-matched GT instances
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    # Step 2: collect non-matched predicted instances as False Positive
                    pred_instance_unmatched = []  # to record those pred instances having no corresponding GT instance
                    for (pred_idx, pred) in enumerate(pred_instances):
                        found_gt = False
                        pred_instance_id = pred['pred_id']

                        for gt in pred['matched_gt']:
                            gt_instance_id = gt["instance_id"] % 1000  # ID of this GT instance (starting from 1)
                            intersect_num = gt['intersection']
                            gt_instance_num = gt['vert_count']
                            pred_instance_num = pred['vert_count']
                            overlap = float(intersect_num) / (gt_instance_num + pred_instance_num - intersect_num)

                            if overlap > overlap_th:
                                found_gt = True
                                break
                        # END for each matched GT instance

                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt['med_dist'] > distance_thresh or gt['dist_conf'] < distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore) / pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)
                                pred_instance_unmatched.append(pred_idx)  # only for testing
                    # END for each pred instance

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)
                # END for seq

                # Step 3: compute average precision for given GT instances and predicted instances
                if has_gt and has_pred:
                    if len(y_score) == 0:
                        ap_current = 0.0
                    else:
                        # 3.1: compute precision recall curve first
                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)  # number of predicted instance
                        num_true_examples = y_true_sorted_cumsum[-1]  # number of predicted instance that has corresponding GT instance
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives  # number of GT instance that has no corresponding pred instance
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.
                        recall[-1] = 0.

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.)

                        stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')

                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)
                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')

                ap[di, li, oi] = ap_current
            # END for
        # END for
    return ap


def compute_averages(aps):
    d_inf = 0
    o50 = np.where(np.isclose(opt.overlaps, 0.5))
    o25 = np.where(np.isclose(opt.overlaps, 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt.overlaps, 0.25)))
    avg_dict = {}
    # avg_dict['all_ap']     = np.nanmean(aps[ d_inf,:,:  ])

    # compute AP over all classes (ignore classes that have never appeared)
    avg_dict['all_ap'] = np.nanmean(aps[d_inf, :, oAllBut25])  # AP
    avg_dict['all_ap_50%'] = np.nanmean(aps[d_inf, :, o50])  # AP_50
    avg_dict['all_ap_25%'] = np.nanmean(aps[d_inf, :, o25])  # AP_25
    avg_dict["classes"] = {}

    # compute AP over each class
    for (li, label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name] = {}
        # avg_dict["classes"][label_name]["ap"]       = np.average(aps[ d_inf,li,  :])
        avg_dict["classes"][label_name]["ap"] = np.average(aps[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["ap50%"] = np.average(aps[d_inf, li, o50])
        avg_dict["classes"][label_name]["ap25%"] = np.average(aps[d_inf, li, o25])
    return avg_dict


def read_prediction_npz(path):
    pred_info = {}
    pred = np.load(path)

    num_instance = len(pred['pred_score'])
    mask = torch.from_numpy(pred['pred_masks']).cuda()
    if 'pred_sem_features' not in pred:
        for i in range(num_instance):
            pred_info[path.split('/')[-1] + '_' + str(i)] = {  # unique id of instance in all scenes
                'mask': mask[:, i].cpu().numpy(),
                'label_id': pred['pred_classes'][i],
                'conf': pred['pred_score'][i]
            }
    else:
        sem_features = torch.from_numpy(pred['pred_sem_features']).cuda()
        for i in range(num_instance):
            pred_info[path.split('/')[-1] + '_' + str(i)] = {  # unique id of instance in all scenes
                'mask': mask[:, i].cpu().numpy(),
                'sem_feature': sem_features[:, i].cpu().numpy(),
                'label_id': pred['pred_classes'][i],
                'conf': pred['pred_score'][i]
            }
    return pred_info


# @brief: return a dict of gt_tensor
# @param gt_ids: GT label_instance_ID of each GT point, ndarray(pts_num, ), dtype=float64;
# @param gt_instances:
def get_gt_tensor(gt_ids, gt_instances):
    gt_tensor_dict = {}
    point_num = len(gt_ids)  # point_num in GT pc
    for label in gt_instances:
        gt_instance_num = len(gt_instances[label])
        gt_tensor = torch.zeros((point_num, gt_instance_num), dtype=torch.bool).cuda()  # Tensor(point_num, instance_num)
        for i, gt_instance_info in enumerate(gt_instances[label]):
            gt_tensor[:, i] = torch.from_numpy( gt_ids == gt_instance_info['instance_id'] )  # for this GT instance, record its mask over all points
        gt_tensor_dict[label] = gt_tensor
    return gt_tensor_dict


# @brief: for each GT instance, find all predicted instances that have overlap with it (match), and vice versa.
# @param pred_file: prediction file(end with .npz);
# @param gt_file: GT label file(end with .txt);
# @param recon_pc: PointCloud obj (gt_num, );
# @param gt_pc: PointCloud obj (gt_num, );
def assign_instances_for_scan(seq_name, pred_file, gt_file, recon_pc, gt_pc, gt_mesh=None, save_seg_gt=False, seg_gt_save_path=None):
    pred_info_raw = read_prediction_npz(pred_file)  # read raw predicted segentation result
    inst_pred_masks = [ value['mask'] for key, value in pred_info_raw.items() ]
    inst_pred_masks = np.stack(inst_pred_masks, axis=0)  # ndarray(pred_inst_num, recon_pts_num), dtype=bool
    valid_recon_pts_mask = np.any(inst_pred_masks, axis=0)

    # pre-process
    pred_info = {}
    for key, value in pred_info_raw.items():
        new_key = "%s.%s" % (seq_name, key.split(".")[-1])
        pred_info[new_key] = value

    gt_ids = np.loadtxt(gt_file)  # instance_ID of each point in GT pc, ndarray(point_num, )

    # Step 1: preparation --- for each predicted instance, mapping it from recon_pc to GT_pc
    # 1.1: for each point in GT PC, find its correspondence in recon PC
    corr_pt_in_recon, valid_gt_pt_indices = align_gt_to_recon(gt_pc, recon_pc, valid_recon_pts_mask=valid_recon_pts_mask, distance_upper_bound=0.15)

    # 1.2: for each pred instance, get its point masks in GT PC
    pred_info = get_instances_in_GT_pc(pred_info, corr_pt_in_recon, valid_gt_pt_indices=valid_gt_pt_indices)

    if save_seg_gt:
        if gt_mesh is not None:
            visualize_pred_instances_in_gt_mesh(gt_mesh, pred_info, save_path=seg_gt_save_path)  # visualize predicted segmentation result on GT mesh
        else:
            visualize_pred_instances_in_gt_pc(gt_pc, pred_info, save_path=seg_gt_save_path)  # visualize predicted segmentation result on GT pc

    if opt.no_class:
        class_ids = gt_ids // 1000
        instance_ids = gt_ids % 1000
        class_ids_valid = np.isin(class_ids, VALID_CLASS_IDS)

        gt_ids_new = np.where(class_ids_valid, instance_ids + VALID_CLASS_IDS[0] * 1000, gt_ids)
        gt_ids = gt_ids_new

    # Step 2: get GT instances on GT pc
    # @param VALID_CLASS_IDS: id(in scannetv2-labels.combined.tsv, col_0);
    # @param CLASS_LABELS: raw_category(in scannetv2-labels.combined.tsv, col_1)
    gt_instances = get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)

    # associate
    gt2pred = deepcopy(gt_instances)
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt['matched_pred'] = []

    pred2gt = {}  # to record matched (with overlap) GT instances for each pred instance
    for label in CLASS_LABELS:
        pred2gt[label] = []
    num_pred_instances = 0

    bool_void = np.logical_not(np.in1d(gt_ids // 1000, VALID_CLASS_IDS))  # mask of points with void label in the GT pc
    gt_tensor_dict = get_gt_tensor(gt_ids, gt_instances)  # Tensor(pts_num, instance_num), dtype=bool

    # Step 3: go through all predicted instances
    for i, pred_mask_file in enumerate(pred_info):
        if opt.no_class:
            label_id = VALID_CLASS_IDS[0]
        else:
            label_id = int(pred_info[pred_mask_file]['label_id'])
        conf = pred_info[pred_mask_file]['conf']

        if not label_id in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]

        # 3.1: preparation
        pred_mask = pred_info[pred_mask_file]['mask']  # predicted mask of this instance(on GT pc) ndarray(point_num, ), dtype=bool

        if len(pred_mask) != len(gt_ids):
            print('wrong number of lines in ' + pred_mask_file + '(%d) vs #mesh vertices (%d), please double check and/or re-download the mesh' % (len(pred_mask), len(gt_ids)))
            raise NotImplementedError

        # convert to binary
        num = np.count_nonzero(pred_mask)  # size of this predicted instance
        if num < opt.min_region_sizes[0]:
            continue

        pred_instance = {}
        pred_instance['filename'] = pred_mask_file
        pred_instance['pred_id'] = num_pred_instances  # pred instance ID(starting from 0)
        pred_instance['label_id'] = label_id
        pred_instance['vert_count'] = num  # corresponding point size of ths pred instance on GT pc
        pred_instance['confidence'] = conf
        pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))

        # 3.2: try to match this pred instance with each GT instance
        matched_gt = []
        gt_tensor = gt_tensor_dict[label_name]  # GT instance masks, Tensor(point_num, GT_instance_num)

        # compute overlap size between this pred instance and each GT instance
        pred_mask_tensor = torch.from_numpy(pred_mask).cuda().unsqueeze(-1)  # Tensor(point_num, 1)
        intersection = torch.sum(gt_tensor & pred_mask_tensor, dim=0)  # overlap size between this pred mask and each GT instance, Tensor(GT_instance_num, )

        intersect_ids = torch.nonzero(intersection).cpu().numpy().reshape(-1)  # instance_IDs of GT instance that has overlap with this predicted instance
        for gt_id in intersect_ids:
            gt_copy = gt_instances[label_name][gt_id].copy()  # info of this GT instance, dict
            pred_copy = pred_instance.copy()  # info of this predicted instance, dict
            intersection_num = intersection[gt_id].item()

            gt_copy['intersection'] = intersection_num
            pred_copy['intersection'] = intersection_num

            matched_gt.append(gt_copy)
            gt2pred[label_name][gt_id]['matched_pred'].append(pred_copy)

        pred_instance['matched_gt'] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    if opt.no_class:
        first_key = list(pred2gt.keys())[0]
        final_pred_inst_num = len(pred2gt[first_key])
        final_gt_inst_num = len(gt2pred[first_key])
    else:
        final_pred_inst_num = sum([len(v) for k, v in pred2gt.items()])
        final_gt_inst_num = sum([len(v) for k, v in gt2pred.items()])
        final_gt_inst_num = sum([len(v) for k, v in gt2pred.items()])

    print("Final pred instance num: %d; final GT instance num %d (with valid labels)" % (final_pred_inst_num, final_gt_inst_num))
    return gt2pred, pred2gt


def print_results(avgs):
    sep = ""
    col1 = ":"
    lineLen = 64

    print("")
    print("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    line += "{:>15}".format("AP") + sep
    line += "{:>15}".format("AP_50%") + sep
    line += "{:>15}".format("AP_25%") + sep
    print(line)
    print("#" * lineLen)

    for (li, label_name) in enumerate(CLASS_LABELS):
        ap_avg = avgs["classes"][label_name]["ap"]
        if np.isnan(ap_avg):
            continue
        ap_50o = avgs["classes"][label_name]["ap50%"]
        ap_25o = avgs["classes"][label_name]["ap25%"]
        line = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(ap_avg) + sep
        line += sep + "{:>15.3f}".format(ap_50o) + sep
        line += sep + "{:>15.3f}".format(ap_25o) + sep
        print(line)

    all_ap_avg = avgs["all_ap"]
    all_ap_50o = avgs["all_ap_50%"]
    all_ap_25o = avgs["all_ap_25%"]

    print("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(all_ap_avg) + sep
    line += "{:>15.3f}".format(all_ap_50o) + sep
    line += "{:>15.3f}".format(all_ap_25o) + sep
    print(line)
    print("")
    return all_ap_25o, all_ap_50o, all_ap_avg


def write_result_file(avgs, filename):
    _SPLITTER = ','
    with open(filename, 'w') as f:
        f.write(_SPLITTER.join(['class', 'class id', 'ap', 'ap50', 'ap25']) + '\n')
        for i in range(len(VALID_CLASS_IDS)):
            class_name = CLASS_LABELS[i]
            class_id = VALID_CLASS_IDS[i]
            ap = avgs["classes"][class_name]["ap"]
            ap50 = avgs["classes"][class_name]["ap50%"]
            ap25 = avgs["classes"][class_name]["ap25%"]
            f.write(_SPLITTER.join([str(x) for x in [class_name, class_id, ap, ap50, ap25]]) + '\n')
        f.write(_SPLITTER.join([str(x) for x in [avgs["all_ap"], avgs["all_ap_50%"], avgs["all_ap_25%"]]]) + '\n')


# @param pred_files: paths of prediction files, list of str;
# @param gt_files: paths of all GT label files, list of str;
# @param recon_pc_list: paths of all reconstructed pointclouds, list of str;
# @param gt_pc_list: paths of all GT pointclouds, list of str;
def evaluate(seq_name_list, pred_files, gt_files, recon_pc_list, gt_pc_list, output_file=None):
    seq_num = len(pred_files)
    print('evaluating', seq_num, 'scans...')
    matches = {}
    for i in range(len(pred_files)):
        seq_name = seq_name_list[i]
        print("\nBegin to process sequence %s (%d / %d)" % (seq_name, i+1, seq_num))

        matches_key = os.path.abspath(gt_files[i])

        # Step 1: read reconstructed pc and GT pc
        recon_pc = o3d.io.read_point_cloud(recon_pc_list[i])
        gt_pc = o3d.io.read_point_cloud(gt_pc_list[i])

        # Step 2: get correspondences between all GT instances and predicted instances
        save_seg_gt = os.path.join( os.path.dirname(recon_pc_list[i]), "final_seg_on_gt.ply" )  # saving path of mapped segmentation result on GT pc
        gt2pred, pred2gt = assign_instances_for_scan(seq_name, pred_files[i], gt_files[i], recon_pc, gt_pc, gt_mesh=None, save_seg_gt=True, seg_gt_save_path=save_seg_gt)

        matches[matches_key] = {}
        matches[matches_key]['gt'] = gt2pred
        matches[matches_key]['pred'] = pred2gt
        sys.stdout.write("scans processed: %d / %d \n" % (i+1, seq_num))
        sys.stdout.flush()

    ap_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)

    print_results(avgs)
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        write_result_file(avgs, output_file)


def main(result_dir, gt_dir, gt_pc_pattern, gt_seg_dir, gt_seg_pattern, selected_seq_names=None):
    gt_pc_pattern = os.path.join(gt_dir, gt_pc_pattern)  # absolute path of GT pc pattern
    gt_file_pattern = os.path.join(gt_seg_dir, gt_seg_pattern)

    # Step 1: choose seq_names
    if selected_seq_names is None:
        seq_names = os.listdir(result_dir)
        seq_names = natsorted(seq_names)
    else:
        seq_name_list = selected_seq_names.strip().split(",")
        seq_names = [seq_name.strip() for seq_name in seq_name_list]

    # Step 2: load corresponding file of each selected sequence
    final_seq_name_list = []
    recon_pc_list = []
    gt_pc_list = []
    pred_files = []
    gt_files = []
    valid_seq_num = 0

    for i, seq_name in enumerate(seq_names):
        recon_pc = os.path.join(result_dir, seq_name, "final.ply")  # final reconstructed pointcloud
        pred_file = os.path.join(result_dir, seq_name, "ckpt_final.npz")  # final predicted segmentation result
        gt_pc = gt_pc_pattern % (seq_name, seq_name)
        gt_file = gt_file_pattern % seq_name

        if not os.path.isfile(recon_pc) or not os.path.isfile(pred_file) or not os.path.isfile(gt_pc) or not os.path.isfile(gt_file):
            print('Results of sequence %s missing...' % seq_name)
            continue

        final_seq_name_list.append(seq_name)
        recon_pc_list.append(recon_pc)
        gt_pc_list.append(gt_pc)
        pred_files.append(pred_file)
        gt_files.append(gt_file)
        valid_seq_num += 1

    print( "\nThere are %d sequences needed to be evaluated totally, %d selected sequences are missing..." % ( valid_seq_num, len(seq_names) - valid_seq_num ) )
    print('start evaluating:')

    # Step 3: compute AP for all selected sequences
    evaluate(final_seq_name_list, pred_files, gt_files, recon_pc_list, gt_pc_list)
    print('save results to', opt.output_file)


if __name__ == '__main__':
    if opt.seq_name == '':
        selected_seq_names = None
    else:
        selected_seq_names = opt.seq_name

    main(opt.result_dir, opt.gt_dir, opt.gt_pc_pattern, opt.gt_seg_dir, opt.gt_seg_pattern, selected_seq_names)
