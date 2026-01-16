from pathlib import Path
from easydict import EasyDict

output = Path('tools/ms3d/data/good')
preds_1f_path = output / "raw_preds_1f"
preds_16f_path = output / "raw_preds_16f"

visualization = False
MOTION_STATE_STATIC = 0
MOTION_STATE_DYNAMIC = 1
MAX_SEQUENCE_LENGTH = 200

preprocess_config = EasyDict(
    score_threshold=0.3,
    roi_threshold=2
)

kde_fusion_config = EasyDict(
    # MATCH=dict(name='euclidean', discard=3, radius=2.0),
    MATCH=dict(name='dbscan', discard=4, radius=2.0),  # 2 is good to cluster Vehicle
    # MATCH=dict(name='iou', discard=4, radius=0.1),
    BANDWIDTH=dict(yaw_opt=True, bw_loc=1.0, bw_size=2.0, bw_yaw=0.1, bw_score=2.0, bw_label=0.5)
)

# 可以修改 asso_thres(official:1.5) 和 det_dist_threshold(official:-0.5)
tracker_1f_config = EasyDict(
    running=dict(
        covariance='default',
        score_threshold=0.6,  # one-stage: use high score threshold to associate pred boxes and tracklets.
        max_age_since_update=MAX_SEQUENCE_LENGTH,  # an alive tracklet with # frames no associated to pred will dead.
        min_hits_to_birth=1,  # associated # frames to make it birth.
        match_type='bipartite',  # association method.
        asso='giou',  # association metric.
        has_velo=False,  # no use boxes without velocity information .
        motion_model='kf',  # kalman filter to predict motion state of boxes.
        asso_thres=dict(giou=1.5)  # the smaller, the stricter.
    ),
    redundancy=dict(
        mode='mm',  # use motion model to predict low confident pred boxes.
        max_redundancy_age=3,
        det_score_threshold=dict(giou=0.1),  # lower threshold for second stage association.
        # -0.5产生更多轨迹，但使得轨迹不容易断续。且会输出纯卡尔曼滤波预测的box，使得静止的车辆会有漂移的框。
        det_dist_threshold=dict(giou=-0.5)  # lower threshold for second stage association.
    )
)
# score_threshold=0.01 asso=giou asso_thres=dict(giou=1.5)
# score_threshold=0.5, asso=iou, asso_thres=dict(iou=0.9)
immortal_tracker_1f_config = EasyDict(
    running=dict(
        covariance='default',
        tracker='immortal',
        score_threshold=0.5,  # one-stage: use high score threshold to associate pred boxes and tracklets.
        max_age_since_update=2,  # an alive tracklet with # frames no associated to pred will dead.
        min_hits_to_birth=dict(
            immortal=1.5
        ),  # associated # frames to make it birth.
        match_type='bipartite',  # association method.
        asso='iou',  # association metric.
        asso_thres=dict(iou=0.9)  # the smaller, the stricter. using 1-iou as metric.
    )
)

tracker_16f_config = EasyDict(
    running=dict(
        covariance='default',
        score_threshold=0.5,
        max_age_since_update=4,
        min_hits_to_birth=1,
        match_type='bipartite',
        asso='iou',
        has_velo=False,
        motion_model='kf',
        asso_thres=dict(iou=0.5)
    ),
    redundancy=dict(
        mode='mm',
        max_redundancy_age=3,
        det_score_threshold=dict(iou=0.1),
        det_dist_threshold=dict(iou=0.1)  # Here it is threshold (not 1-threshold!!)
    )
)
tracklets_preprocess_config = EasyDict(
    score_threshold=0.6,
    min_confident_frame=4,
)

tracklets_refine_config = EasyDict(
    rolling_kde_windows=16,
    min_predicted_tracklet_box_score=0.1,
    min_static_refined_score=0.7,
)

# distance_threshold=1.0
# variance_threshold=0.1
motion_state_config = EasyDict(
    score_threshold=0.6,
    distance_threshold=2.0,
    variance_threshold=0.1,
)
