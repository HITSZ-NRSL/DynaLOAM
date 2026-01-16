"""
Examples:
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda-11.3/
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel

cd Grounded-Segment-Anything
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

Notes:
1. for better performance, you can download Gounding DINO Swin-B model from GoundingDINO official repo.https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
2. Its corresponding config file is stored in GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py.
3. It may take a few time to download BERT from huggingface hub at the first time you run this model.

References:
1. consuming 8.2G GPU MEMS.
2. takeing 1.15 second per image. (3087 imgs / 74 mins)
"""
import torch
import json
import os
from tqdm import tqdm
from pathlib import Path

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
TEXT_PROMPT = "Car. Pedestrian. Cyclist."
IMAGE_DIR = "data/kitti_sparse/training/image_2"
MASK_DIR = "/home/nrsl/dataset/image_2_mask"

DEVICE = 'cuda'
GROUNDING_DINO_CKPT = "/home/nrsl/workspace/temp/Grounded-Segment-Anything/groundingdino_swinb_cogcoor.pth"
SEGMENT_ANTHING_CKPT = "/home/nrsl/workspace/codebase/segment-anything/models/sam_vit_h_4b8939.pth"

if __name__ == "__main__":

    output_dir = Path(MASK_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_list = list(Path(IMAGE_DIR).iterdir())
    file_list.sort()
    for i, image_path in enumerate(tqdm(iterable=file_list)):
        """
        convert [x1,y1,x2,y2] -> [c1,c2,w,h]
        """
        frame_dir = output_dir / image_path.stem

        with open(os.path.join(frame_dir, 'mask.json'), 'r') as f:
            saved_data = json.load(f)
        # import cv2
        # image = cv2.imread(image_path.__str__())
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(saved_data)
        for i in range(1, saved_data.__len__()):
            box = saved_data[i]['box']
            x = (box[0] + box[2]) / 2
            y = (box[1] + box[3]) / 2
            w = box[2] - box[0]
            h = box[3] - box[1]
            saved_data[i]['box'] = [x, y, w, h]
            # print((int(x), (int)(y)))
            # cv2.line(image, ((int(x), (int)(y-h/2))), ((int(x), (int)(y+h/2))), (255, 0, 0), 5)

        # cv2.imshow("123", image)
        # cv2.waitKey()
        # break
        # print(saved_data)
        # break
        with open(os.path.join(frame_dir, 'mask.json'), 'w') as f:
            json.dump(saved_data, f)
