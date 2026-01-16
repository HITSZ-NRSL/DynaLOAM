import os
from pathlib import Path
from rd3d import PROJECT_ROOT

kitti_raw_sequences = Path('/home/nrsl/dataset/kitti_raw')
output_path = PROJECT_ROOT / 'data/custom'


def run_lidar_odom(sequence_path):
    try:
        import kiss_icp
    except ImportError as e:
        os.system("pip install kiss-icp")

    _ = os.system(f"cd {str(sequence_path)} && kiss_icp_pipeline pointclouds")
    result_path = sequence_path / 'results'
    odom_path = result_path / 'latest/pointclouds_poses.npy'
    save_path = sequence_path / 'pointclouds_poses.npy'
    os.system(f"mv {str(odom_path)} {str(save_path)}")
    os.system(f"rm -rf {str(result_path)}")


def load_timestamps(pth):
    output_timestamps = []
    with open(str(pth), 'r') as f:
        timestamps = [line.split('\n')[0] for line in f.readlines()]
    for idx in range(len(timestamps)):
        if timestamps[idx]:
            output_timestamps.append(''.join(c for c in timestamps[idx] if c.isdigit()))
    return output_timestamps


for sequence in kitti_raw_sequences.iterdir():
    sequence_output_path = output_path / 'sequences' / sequence.name
    sequence_output_path.mkdir(exist_ok=False, parents=True)
    (sequence_output_path / 'pointclouds').symlink_to(sequence / 'velodyne_points/data')
    (sequence_output_path / 'image_02').symlink_to(sequence / 'image_02')
    timestamps = load_timestamps(sequence / 'velodyne_points/timestamps.txt')
    with open(sequence_output_path / 'timestamps.txt', 'w') as f:
        f.write('\n'.join(timestamps))
    # run_lidar_odom(sequence_output_path)
