experiment = "raw_preds_16f"

launcher = "accelerate-launch"

models = [
    # "--cfg configs/iassd/iassd_4x8_80e_custom.py  --ckpt  model_zoo/iassd_kitti_3cls_e80.pth"
    "--cfg configs/centerpoint/centerpoint_4x4_custom.py --ckpt model_zoo/centerpoint_4x4_kitti_3cls_da_e80.pth",
    "--cfg configs/voxel_rcnn/voxel_rcnn_4x2_80e_custom.py --ckpt model_zoo/voxel_rcnn_kitti_3cls_da_e80.pth",
]
args = ""
tta = [
    "--eval-tag raw",
    "--eval-tag aug0 --tta --set DATASET.TEST_TIME_AUGMENTOR.ENABLE_INDEX=0",
    "--eval-tag aug1 --tta --set DATASET.TEST_TIME_AUGMENTOR.ENABLE_INDEX=1",
    "--eval-tag aug2 --tta --set DATASET.TEST_TIME_AUGMENTOR.ENABLE_INDEX=2",
]

workspace = f"output/ms3d/{experiment}"
