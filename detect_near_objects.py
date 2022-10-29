import warnings
import torch

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import  TracedModel

warnings.simplefilter('ignore')

def detect(model, target_image, image_size, stride, conf_thres, iou_thres, device):
    model(torch.zeros(1, 3, image_size, image_size).to(device).type_as(next(model.parameters())))
    dataset = LoadImages(target_image, img_size=image_size, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names

    results = {}

    for path, img, im0s, _ in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        for det in pred: 
            if len(det):
                labels = []
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *_, conf, cls in reversed(det):
                    name = names[int(cls)]
                    # print(f'{name} {conf:.2f}')
                    labels.append([name, conf.item()])
                results[path] = labels
    
    return results

if __name__ == '__main__':
    model_path = "yolov7.pt"
    conf_thres = 0.25
    iou_thres = 0.45
    image_size = 640
    target_image = "inference/images/horses.jpg"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = attempt_load(model_path, map_location=device)  # load FP32 model

    stride = int(model.stride.max())  # type: ignore # model stride
    image_size = check_img_size(image_size, s=stride)  # check img_size
    model = TracedModel(model, device, image_size)

    res = detect(model, target_image, image_size, stride, conf_thres, iou_thres, device) # dict(path, list[label, pred])
    print(res)