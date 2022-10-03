# yolov4
An implementation of yolov4 using pytorch.


## Quick start
1. Create virtual environment:

    ```shell
    conda create -n yolov4 python=3.8
    conda activate yolov4
    pip install -r requirements.txt
    ```

2. Install [PyTorch](https://pytorch.org/), refer to the [blog](https://blog.csdn.net/qq_23013309/article/details/103965619) for details.


## Train
1. Download VOC2007 dataset from following website and unzip them:

    * http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    * http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tarnload.py

2. Download pre-trained `CSPDarknet53.pth` model from [Google Drive](https://drive.google.com/file/d/12oV8QL937S1JWFQhzLNPoqyYc_bi0lWT/view?usp=sharing).


3. Modify the value of `root` in `train.py`, please ensure that the directory structure of the `root` folder is as follows:

    ```txt
    root
    ├───Annotations
    ├───ImageSets
    │   ├───Layout
    │   ├───Main
    │   └───Segmentation
    ├───JPEGImages
    ├───SegmentationClass
    └───SegmentationObject
    ```

4. start training:

    ```shell
    conda activate yolov4
    python train.py
    ```

## Evaluation
### one model
1. Modify the value of `root` and `model_path` in `eval.py`.
2. Calculate mAP:

    ```sh
    conda activate yolov4
    python eval.py
    ```

### multi models
1. Modify the value of `root` and `model_dir` in `evals.py`.
2. Calculate and plot mAP:

    ```shell
    conda activate yolov4
    python evals.py
    ```

### mAP curve
I trained the model on VOC2012 dataset and the mAP curse is shown in the figure below:

![mAP 曲线](resource/image/mAP_曲线.png)


### best mAP
```
+-------------+--------+--------+
|    class    |   AP   |  mAP   |
+-------------+--------+--------+
|  aeroplane  | 88.87% | 82.92% |
|   bicycle   | 90.69% |        |
|     bird    | 85.72% |        |
|     boat    | 66.17% |        |
|    bottle   | 71.61% |        |
|     bus     | 89.98% |        |
|     car     | 92.20% |        |
|     cat     | 91.61% |        |
|    chair    | 66.64% |        |
|     cow     | 91.90% |        |
| diningtable | 80.29% |        |
|     dog     | 89.20% |        |
|    horse    | 90.04% |        |
|  motorbike  | 88.59% |        |
|    person   | 88.20% |        |
| pottedplant | 51.06% |        |
|    sheep    | 86.86% |        |
|     sofa    | 75.07% |        |
|    train    | 89.05% |        |
|  tvmonitor  | 84.70% |        |
+-------------+--------+--------+
```

## Detection
1. Modify the `model_path` and `image_path` in `demo.py`.

2. Display detection results:

    ```shell
    conda activate yolov4
    python demo.py
    ```


## Reference
* [[Paper] YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)
* [[GitHub] bubbliiiing / yolov4-pytorch](https://github.com/bubbliiiing/yolov4-pytorch)


## Notes
1. 82.92% mAP may not be the limit of this project. You can try to train the model on the VOC2007 + VOC2012 dataset. The mAP should be higher. If you get a better result, please don't hesitate to tell me.
2. If you want to train custom dataset, here are some steps to follow:
   1. The label file must be in the same XML format as VOC2007, and the structure of dataset must be the same as follows:

        ```txt
        root
        ├───Annotations
        ├───ImageSets
        │   └───Main
        └───JPEGImages
        ```
   2. Put your `test.txt` and `train.txt` in the `Main` folder. These txt files must contain the names of the corresponding **jpg** format pictures. These names do not need a suffix.
   3. Modify the `classes` property of `VOCDataset` in `net/dataset.py` to include all the classes in your dataset.
   4. Change the `root` and `image_set` of `VOCDataset` in `train.py` and start training.

## License
```txt
MIT License

Copyright (c) 2022 Huang Zhengzhi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```