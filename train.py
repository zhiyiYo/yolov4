# coding:utf-8
from net import TrainPipeline, VOCDataset
from utils.augmentation_utils import YoloAugmentation


# train config
config = {
    "n_classes": len(VOCDataset.classes),
    "image_size": 416,
    "anchors": [
        [142, 110], [192, 243], [459, 401],
        [36, 75], [76, 55], [72, 146],
        [12, 16], [19, 36], [40, 28],
    ],
    "darknet_path": "model/CSPdarknet53.pth",
    "batch_size": 4
}

# load dataset
root = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
dataset = VOCDataset(
    root,
    'trainval',
    YoloAugmentation(config['image_size']),
    keep_difficult=True
)


# train
train_pipeline = TrainPipeline(dataset=dataset, **config)
train_pipeline.train()
