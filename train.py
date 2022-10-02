# coding:utf-8
from net import TrainPipeline, VOCDataset
from utils.augmentation_utils import YoloAugmentation, ColorAugmentation


# train config
config = {
    "n_classes": len(VOCDataset.classes),
    "image_size": 416,
    "anchors": [
        [[142, 110], [192, 243], [459, 401]],
        [[36, 75], [76, 55], [72, 146]],
        [[12, 16], [19, 36], [40, 28]],
    ],
    "darknet_path": "model/CSPdarknet53.pth",
    "lr": 1e-2,
    "batch_size": 4,
    "freeze_batch_size": 8,
    "freeze": True,
    "freeze_epoch": 30,
    "max_epoch": 100,
    "start_epoch": 0,
    "num_workers": 4
}

# load dataset
root = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
dataset = VOCDataset(
    root,
    'train',
    transformer=YoloAugmentation(config['image_size']),
    color_transformer=ColorAugmentation(config['image_size']),
    keep_difficult=True,
    use_mosaic=True,
    use_mixup=True,
    image_size=config["image_size"]
)

if __name__ == '__main__':
    train_pipeline = TrainPipeline(dataset=dataset, **config)
    train_pipeline.train()
