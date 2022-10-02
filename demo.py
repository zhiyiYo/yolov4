# coding:utf-8
from net import VOCDataset
from utils.detection_utils import image_detect

# 模型文件和图片路径
model_path = 'model/2022-10-02_16-52-03/Yolo_51.pth'
image_path = 'resource/image/聚餐.jpg'

# 检测目标
image = image_detect(model_path, image_path, VOCDataset.classes, conf_thresh=0.3)
image.show()