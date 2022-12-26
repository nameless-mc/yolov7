import json
import os
import warnings
import torch
from yolov7.detect import detect_output_dict

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size
from yolov7.utils.torch_utils import TracedModel

warnings.simplefilter('ignore')

if __name__ == '__main__':
    model_path = "yolov7.pt"
    conf_thres = 0.01
    iou_thres = 0.01
    image_size = 64
    target_image_dir = f"dataset/data/2022-12-21T18:22:03+09:00/filtered/img"
    json_path = f"dataset/data/2022-12-21T18:22:03+09:00/filtered/detect.json"

    target_images = os.path.join(target_image_dir, "*.png")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = attempt_load(model_path, map_location=device)  # load FP32 model

    stride = int(model.stride.max())  # type: ignore # model stride
    imgsz = check_img_size(image_size, s=stride)  # check img_size
    model = TracedModel(model, device, image_size)

    res = detect_output_dict(model, target_images, imgsz, stride, conf_thres,
                 iou_thres, device, lambda path : os.path.splitext(os.path.basename(path))[0])  # dict(path, list[list[label, pred]])
    with open(json_path, "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
