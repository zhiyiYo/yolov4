# coding:utf-8
from net import VOCDataset
from utils.detection_utils import image_detect

# 模型文件和图片路径
model_path = 'model/yolo4_voc_weights.pth'
image_path = 'resource/image/街道.jpg'

# 检测目标
image = image_detect(model_path, image_path, VOCDataset.classes, conf_thresh=0.4)
image.show()