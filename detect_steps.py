import json
import warnings
import torch
from detect import detect_output_dict

from models.experimental import attempt_load
from utils.general import check_img_size
from utils.torch_utils import TracedModel

warnings.simplefilter('ignore')

if __name__ == '__main__':
    model_path = "yolov7.pt"
    conf_thres = 0.01
    iou_thres = 0.01
    image_size = 64
    target_step = 0
    target_images = f"dataset/result_for_humans/img/*/steps/*/image_steps/{target_step}.png"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = attempt_load(model_path, map_location=device)  # load FP32 model

    stride = int(model.stride.max())  # type: ignore # model stride
    imgsz = check_img_size(image_size, s=stride)  # check img_size
    model = TracedModel(model, device, image_size)

    res = detect_output_dict(model, target_images, imgsz, stride, conf_thres,
                 iou_thres, device, lambda path : path.split('/')[-3])  # dict(path, list[list[label, pred]])
    with open(f"./dataset/detect_step_{target_step}.json", "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)
