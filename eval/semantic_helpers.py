import os
import numpy as np
import torch


def compute_label_id_by_sim(inst_features, label_text_features, label_names, label_ids, valid_class_indices=None, top_k=5):
    inst_num_bf = inst_features.shape[0]

    # Step 1: for each given instance, get the most possible label index by computing similarity
    raw_similarity = np.dot(inst_features, label_text_features.T)
    exp_sim = np.exp(raw_similarity * 100)
    prob = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
    label_indices = np.argmax(prob, axis=-1)

    if valid_class_indices is not None:
        inst_kept_mask = np.isin(label_indices, valid_class_indices)
        label_indices = np.where(inst_kept_mask, label_indices, -1 * np.ones_like(label_indices))
        kept_inst_indices = np.where(inst_kept_mask)[0]
    else:
        kept_inst_indices = np.arange(inst_num_bf)

    # Step 2: get the most possible label_name and label_ID for each given instance
    label_indices_list = label_indices.tolist()
    kept_inst_indices = kept_inst_indices.tolist()

    inst_label_names_list = []
    inst_label_ids_list = []
    for i, label_index in enumerate(label_indices_list):
        if label_index != -1:
            inst_label_names_list.append( label_names[label_index] )
            inst_label_ids_list.append( label_ids[label_index] )
        else:
            inst_label_names_list.append("unlabeled")
            inst_label_ids_list.append(-1)

    return kept_inst_indices, inst_label_ids_list, inst_label_names_list


def compute_label_id_by_sim2(inst_features, label_text_features, label_names, label_ids, valid_class_indices=None, top_k=5):
    inst_num_bf = inst_features.shape[0]

    # Step 1: for each given instance, get the most possible label index by computing similarity
    raw_similarity = np.dot(inst_features, label_text_features.T)
    exp_sim = np.exp(raw_similarity * 100)
    prob = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)
    label_indices = np.argmax(prob, axis=-1)

    if valid_class_indices is not None:
        inst_kept_mask = np.isin(label_indices, valid_class_indices)
        label_indices = np.where(inst_kept_mask, label_indices, -1 * np.ones_like(label_indices))
        kept_inst_indices = np.where(inst_kept_mask)[0]
    else:
        kept_inst_indices = np.arange(inst_num_bf)

    # Step 2: get the most possible label_name and label_ID for each given instance
    label_indices_list = label_indices.tolist()
    kept_inst_indices = kept_inst_indices.tolist()

    inst_label_names_list = []
    inst_label_ids_list = []
    for i, label_index in enumerate(label_indices_list):
        if label_index != -1:
            inst_label_names_list.append( label_names[label_index] )
            inst_label_ids_list.append( label_ids[label_index] )
        else:
            inst_label_names_list.append("unlabeled")
            inst_label_ids_list.append(-1)

    return kept_inst_indices, inst_label_ids_list, inst_label_names_list


def remove_keys_with_value(d, values_to_remove):
    return {k: v for k, v in d.items() if v not in values_to_remove}


def change_to_mc_format(pred_info, seq_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "%s.npz" % seq_name)

    inst_mask_list = []
    for key, value in pred_info.items():
        inst_mask = value["mask"]
        inst_mask_list.append(inst_mask)
    pred_inst_masks = np.stack(inst_mask_list, axis=0)
    pred_inst_num = len(inst_mask_list)

    pred_dict = {
        "pred_masks": pred_inst_masks.transpose(),
        "pred_score": np.ones(pred_inst_num),
        "pred_classes": np.zeros(pred_inst_num, dtype=np.int32)
    }

    np.savez(output_path, **pred_dict)
    print("Prediction result (MC format) of %s is saved to: %s" % (seq_name, output_path))


def main(pred_inst_file_raw, output_path):
    pred_inst_masks = np.load(pred_inst_file_raw)  # ndarray(inst_num, gt_pts_num)
    pred_inst_num = pred_inst_masks.shape[0]

    pred_dict = {
        "pred_masks": pred_inst_masks.transpose(),
        "pred_score": np.ones(pred_inst_num),
        "pred_classes": np.zeros(pred_inst_num, dtype=np.int32)
    }

    save_dir = os.path.dirname(output_path)
    os.makedirs(save_dir, exist_ok=True)
    np.savez(output_path, **pred_dict)