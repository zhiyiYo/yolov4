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

2. Download pre-trained `CSPDarknet53.pth` model from [Google Drive](https://drive.google.com/file/d/1ig5-fAfT2z3EBXEDlUsSJVxKPJRTftm7/view?usp=sharing).

3. Modify `config` in `train.py` and start training:

    ```shell
    conda activate yolov4
    python train.py
    ```

## Evaluation
1. Modify the `model_path` in `eval.py`.
2. Calculate mAP:

    ```shell
    conda activate yolov4
    python eval.py
    ```


## Detection
1. Modify the `model_path` and `image_path` in `demo.py`.

2. Display detection results:

    ```shell
    conda activate yolov4
    python demo.py
    ```


## Notes
1. Sometimes the data set downloaded through `download.py`' may be incomplete, so please check whether the number of pictures in the data set is correct after downloading, or you can download the data set directly through the browser in the following addresses:
   * http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   * http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar


## Reference
* [[Paper] YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)
* [[GitHub] bubbliiiing / yolov4-pytorch](https://github.com/bubbliiiing/yolov4-pytorch)
