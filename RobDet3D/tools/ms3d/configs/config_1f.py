experiment = "raw_preds_1f"

launcher = "accelerate-launch"

models = [
    "--cfg configs/iassd/target_custom/iassd_kitti_3cls.py "
    "--ckpt model_zoo/iassd_kitti_3cls_e80.pth",
    #
    # "--cfg configs/iassd/target_custom/iassd_nuscenes.py "
    # "--ckpt model_zoo/iassd_nuscenes_20e.pth",
    #
    "--cfg configs/voxel_rcnn/target_custom/voxel_rcnn_kitti_3cls_da.py "
    "--ckpt model_zoo/voxel_rcnn_kitti_3cls_da_e80.pth",
    #
    # "--cfg configs/voxelnext/target_custom/voxelnext_nuscenes.py "
    # "--ckpt model_zoo/voxelnext_nuscenes_kernel1.pth",

    "--cfg configs/iassd/target_custom/iassd_havsx2_waymo.py "
    "--ckpt model_zoo/iassd_havsx2_waymo_e30.pth",

    "--cfg configs/voxel_rcnn/target_custom/voxel_rcnn_ctr_head_dyn_voxel_waymo.py "
    "--ckpt model_zoo/voxel_rcnn_ctr_head_dyn_voxel_waymo_05_30e.pth"
]
args = "--set DATASET.SWEEPS.RANGE=[0,0]"
tta = [
    "--eval-tag raw",
    "--eval-tag aug0 --tta --set DATASET.TEST_TIME_AUGMENTOR.ENABLE_INDEX=0",
    "--eval-tag aug1 --tta --set DATASET.TEST_TIME_AUGMENTOR.ENABLE_INDEX=1",
    "--eval-tag aug2 --tta --set DATASET.TEST_TIME_AUGMENTOR.ENABLE_INDEX=2",
]

workspace = f"output/ms3d/{experiment}"
