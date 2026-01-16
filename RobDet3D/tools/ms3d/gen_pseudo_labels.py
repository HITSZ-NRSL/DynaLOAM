import pickle
import numpy as np
import torch
from contextlib import contextmanager
from tqdm import tqdm

from rd3d.utils import box_fusion_utils
from rd3d.utils.viz_utils import viz_scenes
from rd3d.utils.common_utils import limit_period
from configs.pseudo_labels import *


# TODO
# 3. 如何 缝合 SimpleTracker和ImmortalTracker。
# DONE
# 2. tracker结果的输出是连续的，无需插值。但输出内确实有很低置信度的包围框。

@contextmanager
def cache(func, file):
    def wrapper(*args, **kwargs):
        if not Path(file).exists():
            pickle.dump(func(*args, **kwargs), open(file, 'wb'))
        return pickle.load(open(file, 'rb'))

    yield wrapper


def get_dataset():
    from rd3d.datasets.datasets.custom.custom_dataset import CustomDataset
    from rd3d.core.config import Config
    from rd3d import build_dataloader
    cfg = Config.fromfile("configs/base/datasets/custom.py")
    dataset: CustomDataset = build_dataloader(cfg.DATASET, training=False)
    return dataset


def to_world_frame(p, b, t):
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_matrix(t[:3, :3]).as_euler('xyz', degrees=False)
    p[:, :3] = (np.concatenate((p[:, :3], np.ones_like(p[..., :1])), axis=-1) @ t.T)[:, :3]
    b[:, :3] = (np.concatenate((b[:, :3], np.ones_like(b[..., :1])), axis=-1) @ t.T)[:, :3]
    b[:, 6] = np.arctan2(np.sin(b[:, 6] + rot[-1]), np.cos(b[:, 6] + rot[-1]))
    return p, b


def preprocess(pred_dicts):
    """
    1. filter out invalid class and map valid class to unify class name across dataset.
    2. filter out the boxes with low scores.
    3. limit the heading to [-pi,pi).
    4. reassign the pred_labels according to unify class names to avoid inconsistency after class mapping.
    Returns:
        pred_dicts: [
            {
                'name': ndarray(n,),
                'score': ndarray(n,),
                'boxes_lidar': ndarray(n,7),
                'pred_labels': ndarray(n,),
                'frame_id': str,
            }
        ]
    """
    for pred_dict in tqdm(iterable=pred_dicts, desc='preprocess'):
        cls_mask = np.vectorize(lambda x: x in dataset.class_names or x in dataset.class_map)(pred_dict['name'])
        dis_mask = np.linalg.norm(pred_dict['boxes_lidar'][:, :2], axis=-1) > preprocess_config.roi_threshold
        scr_mask = pred_dict['score'] > preprocess_config.score_threshold
        mask = np.logical_and(cls_mask, np.logical_and(dis_mask, scr_mask))

        pred_dict['name'] = np.vectorize(lambda x: dataset.class_map.get(x, x))(pred_dict['name'][mask])
        pred_dict['score'] = pred_dict['score'][mask]
        pred_dict['boxes_lidar'] = pred_dict['boxes_lidar'][mask]
        pred_dict['boxes_lidar'][:, 6] = limit_period(pred_dict['boxes_lidar'][:, 6], offset=0.5, period=2 * np.pi)
        pred_dict['pred_labels'] = np.vectorize(lambda x: dataset.class_names.index(x) + 1)(pred_dict['name'])
        pred_dict['frame_id'] = pred_dict['frame_id'][0]
    return pred_dicts


def fuse_tta_results(pred_dicts):
    fused_boxes = {}
    for scene_idx, pred_dict in tqdm(iterable=enumerate(pred_dicts), desc='boxes fusion'):
        tta_pred_boxes = np.hstack((pred_dict['boxes_lidar'],
                                    pred_dict['score'][:, None],
                                    pred_dict['pred_labels'][:, None].astype(np.float32)))

        fused_boxes[pred_dict['frame_id']] = box_fusion_utils.kde_boxes_fusion(tta_pred_boxes, kde_fusion_config)
    return fused_boxes


def gen_tracklets_immortal_tracker(dataset, pred_boxes, tracker_cfg, frame_ids=None):
    """
    Uses SimpleTrack to generate tracklets for the dataset
    """
    from rd3d.tracker.mot import MOTModel
    from rd3d.tracker.data_protos import BBox
    from rd3d.tracker.frame_data import FrameData
    from collections import defaultdict

    frame_id_of_seqs = defaultdict(list)  # {seq_name: [frame_id]}
    for frame_id in pred_boxes:
        seq_name, sample_idx = dataset.get_sequence_name(frame_id)
        frame_id_of_seqs[seq_name].append(frame_id)

    tracklets_of_seqs = {seq_name: {} for seq_name in frame_id_of_seqs}
    for seq_name in tqdm(iterable=frame_id_of_seqs, desc='tracking'):
        tracker = MOTModel(tracker_cfg)
        frame_ids = sorted(frame_id_of_seqs[seq_name])[:MAX_SEQUENCE_LENGTH]
        for sample_idx, frame_id in tqdm(iterable=enumerate(frame_ids), desc=seq_name, leave=False):
            boxes = pred_boxes[frame_id].copy()
            timestamp = dataset.get_timestamp(seq_name, sample_idx)
            pose = dataset.get_pose(seq_name, sample_idx)
            points = np.zeros((0, 3))

            points_global, boxes_global = to_world_frame(points, boxes, pose)
            # [(trk.get_state(), trk.id, state_string, trk.det_type)]
            results = tracker.frame_mot(FrameData(dets=list(boxes_global[:, [0, 1, 2, 6, 3, 4, 5, 7]]),
                                                  det_types=list(boxes_global[:, 8]), pc=points_global,
                                                  ego=pose, time_stamp=timestamp,
                                                  aux_info={'is_key_frame': True, 'velos': None}))
            # results = [res for res in results if Validity.valid(res[2])]
            for box, tk_id, state, cls in results:
                if tk_id not in tracklets_of_seqs[seq_name]:
                    tracklets_of_seqs[seq_name][tk_id] = defaultdict(list)
                box = BBox.bbox2array(box)[[0, 1, 2, 4, 5, 6, 3, 7]]
                tracklets_of_seqs[seq_name][tk_id]['boxes'].append(np.hstack((box, cls)))
                tracklets_of_seqs[seq_name][tk_id]['frame_id'].append(frame_id)
                tracklets_of_seqs[seq_name][tk_id]['associated'].append(state)

    return tracklets_of_seqs


def gen_tracklets_simple_tracker(dataset, pred_boxes, tracker_cfg, frame_ids=None):
    """
    Uses SimpleTrack to generate tracklets for the dataset
    """
    from collections import defaultdict
    from mot_3d.mot import MOTModel
    from mot_3d.data_protos import BBox
    from mot_3d.frame_data import FrameData

    frame_id_of_seqs = defaultdict(list)  # {seq_name: [frame_id]}
    for frame_id in pred_boxes:
        seq_name, sample_idx = dataset.get_sequence_name(frame_id)
        frame_id_of_seqs[seq_name].append(frame_id)

    tracklets_of_seqs = {seq_name: {} for seq_name in frame_id_of_seqs}
    for seq_name in tqdm(iterable=frame_id_of_seqs, desc='tracking'):
        tracker = MOTModel(tracker_cfg)
        frame_ids = sorted(frame_id_of_seqs[seq_name])[:MAX_SEQUENCE_LENGTH]
        for sample_idx, frame_id in tqdm(iterable=enumerate(frame_ids), desc=seq_name, leave=False):
            boxes = pred_boxes[frame_id].copy()
            timestamp = dataset.get_timestamp(seq_name, sample_idx)
            pose = dataset.get_pose(seq_name, sample_idx)
            points = np.zeros((0, 3))

            points_global, boxes_global = to_world_frame(points, boxes, pose)
            # [(trk.get_state(), trk.id, state_string, trk.det_type)]
            results = tracker.frame_mot(FrameData(dets=list(boxes_global[:, [0, 1, 2, 6, 3, 4, 5, 7]]),
                                                  det_types=list(boxes_global[:, 8]), pc=points_global,
                                                  ego=pose, time_stamp=timestamp,
                                                  aux_info={'is_key_frame': True, 'velos': None}))

            for box, tk_id, state, cls in results:
                if tk_id not in tracklets_of_seqs[seq_name]:
                    tracklets_of_seqs[seq_name][tk_id] = defaultdict(list)
                box = BBox.bbox2array(box)[[0, 1, 2, 4, 5, 6, 3, 7]]
                tracklets_of_seqs[seq_name][tk_id]['boxes'].append(np.hstack((box, cls)))
                tracklets_of_seqs[seq_name][tk_id]['frame_id'].append(frame_id)
                tracklets_of_seqs[seq_name][tk_id]['associated'].append(state)

    return tracklets_of_seqs


def preprocess_tracklets(all_tracklets, refine_config):
    from mot_3d.data_protos.validity import Validity

    for tracklets_of_seq in tqdm(iterable=all_tracklets.values(), desc=f"refine"):
        for tk_id in tracklets_of_seq:
            validation = np.vectorize(Validity.valid)(tracklets_of_seq[tk_id]['associated'])
            last_valid_index = np.nonzero(validation.tolist())[0][-1] + 1
            tracklets_of_seq[tk_id] = {k: np.array(v[:last_valid_index]) for k, v in tracklets_of_seq[tk_id].items()}

        for tk_id in list(tracklets_of_seq.keys()):
            valid_mask = np.greater(tracklets_of_seq[tk_id]['boxes'][:, 7], refine_config.score_threshold)
            if np.sum(valid_mask) < refine_config.min_confident_frame:
                del tracklets_of_seq[tk_id]
    return all_tracklets


def motion_state_estimation(all_tracklets, config):
    for seq in tqdm(iterable=all_tracklets, desc="motion estimation"):
        for tk in tqdm(iterable=all_tracklets[seq].values(), desc=seq, leave=False):
            mask = tk['boxes'][:, 7] > config.score_threshold
            confident_boxes = tk['boxes'][mask]
            location = confident_boxes[:, :2]
            start_to_end_dist = np.linalg.norm(location[0] - location[-1])
            location_variance = np.var(location, axis=0).mean()
            if start_to_end_dist > config.distance_threshold or location_variance > config.variance_threshold:
                motion_state = MOTION_STATE_DYNAMIC
            else:
                motion_state = MOTION_STATE_STATIC
            tk['motion_state'] = motion_state
    return all_tracklets


def refine_tracklets(all_tracklets, refine_config):
    def rolling_kde_refine():
        if len(boxes) < refine_config.rolling_kde_windows:
            refined_box = box_fusion_utils.kde_box_fusion_one_cluster(
                boxes, weights, **kde_fusion_config.BANDWIDTH
            )
            if tk['motion_state'] == MOTION_STATE_DYNAMIC:
                refined_boxes = boxes.copy()
                refined_boxes[:, 3:6] = refined_box[3:6]
                tk['refined_boxes'] = {fid: box for fid, box in zip(tk['frame_id'], refined_boxes)}
            if tk['motion_state'] == MOTION_STATE_STATIC:
                refined_box[7] = max(refined_box[7], refine_config.min_static_refined_score)
                tk['refined_boxes'] = {fid: refined_box for fid in tk['frame_id']}
        else:
            tk['refined_boxes'] = {}
            for i, fid in enumerate(tk['frame_id']):
                select_idx = np.arange(i - refine_config.rolling_kde_windows, i) + 1
                select_idx = np.clip(select_idx % len(boxes), 0, len(boxes) - 1)
                refined_box = box_fusion_utils.kde_box_fusion_one_cluster(
                    boxes[select_idx], weights[select_idx], **kde_fusion_config.BANDWIDTH
                )
                if tk['motion_state'] == MOTION_STATE_DYNAMIC:
                    refined_box = boxes[i].copy()
                    refined_box[3:6] = refined_box[3:6]
                    tk['refined_boxes'][fid] = refined_box
                if tk['motion_state'] == MOTION_STATE_STATIC:
                    refined_box[7] = max(refined_box[7], refine_config.min_static_refined_score)
                    tk['refined_boxes'][fid] = refined_box

    def corner_align():
        for i, (fid, refined_box) in enumerate(tk['refined_boxes'].items()):
            inc_size = refined_box[3:6] - tk['boxes'][i, 3:6]
            refined_box[:3] -= np.sign(refined_box[:3]) * inc_size

    for seq, tracklets in tqdm(iterable=all_tracklets.items(), desc='refine tracklets'):
        for tk_id, tk in tqdm(iterable=tracklets.items(), desc=seq, leave=False):
            boxes = tk['boxes']
            # predicted boxes of tracker have small scores.
            weights = np.clip(boxes[:, 7], refine_config.min_predicted_tracklet_box_score, 1.0)
            rolling_kde_refine()
            corner_align()
    return all_tracklets


dataset = get_dataset()

with cache(box_fusion_utils.concatenate_preds_from_disk,
           output / f"all_{preds_1f_path.name}_concat.pkl") as handler:
    pred_dicts = handler(preds_1f_path)

with cache(preprocess,
           output / f"all_{preds_1f_path.name}_preprocess.pkl") as handler:
    pred_dicts = handler(pred_dicts)

with cache(fuse_tta_results,
           output / f"all_{preds_1f_path.name}_fusion.pkl") as handler:
    pred_boxes = handler(pred_dicts)

with cache(gen_tracklets_simple_tracker, output / f"tracklets_{preds_1f_path.name}_world.pkl") as handler:
    tracklets_1f = handler(dataset, pred_boxes, tracker_1f_config)
# with cache(gen_tracklets_immortal_tracker, output / f"tracklets_{preds_1f_path.name}_world.pkl") as handler:
#     tracklets_1f = handler(dataset, pred_boxes, immortal_tracker_1f_config)

with cache(preprocess_tracklets, output / f"tracklets_{preds_1f_path.name}_preprocess.pkl") as handler:
    tracklets_1f = handler(tracklets_1f, tracklets_preprocess_config)

with cache(motion_state_estimation, output / f"tracklets_{preds_1f_path.name}_motion.pkl") as handler:
    tracklets_1f = handler(tracklets_1f, motion_state_config)

with cache(refine_tracklets, output / f"tracklets_{preds_1f_path.name}_refine.pkl") as handler:
    tracklets_1f = handler(tracklets_1f, tracklets_refine_config)

# with cache(fuse_tta_results, output / f"all_{preds_16f_path.name}.pkl") as handler:
#     _, _, pred_boxes_16f = handler(preds_16f_path)
# with cache(gen_tracklets, output / f"tracklets_{preds_16f_path.name}_world.pkl") as handler:
#     tracklets_16f = handler(dataset, pred_boxes_16f, tracker_16f_config)
