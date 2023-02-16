from argparse import ArgumentParser
import json
import os
import warnings
import torch
from detect import detect_output_dict

from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import TracedModel

import glob

warnings.simplefilter('ignore')

def detect(target_image_dir, json_path, model, device):
    conf_thres = 0.01
    iou_thres = 0.01
    image_size = 64

    target_images = os.path.join(target_image_dir, "*.png")

    stride = int(model.stride.max())  # type: ignore # model stride
    imgsz = check_img_size(image_size, s=stride)  # check img_size
    model = TracedModel(model, device, image_size)

    res = detect_output_dict(model, target_images, imgsz, stride, conf_thres,
                 iou_thres, device, lambda path : os.path.splitext(os.path.basename(path))[0])
    with open(json_path, "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--target', help="target images dir", required=True)
    parser.add_argument('-o', '--output', help="output json path", required=True)
    parser.add_argument('-m', '--model', help="yolov7 model path (default: 'models/yolov7.pt')", default='models/yolov7.pt')
    args = parser.parse_args()
    target_image_dir = args.target
    json_path = args.output
    model_path = args.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(model_path, map_location=device)
    detect(target_image_dir, json_path, model, device)
