# coding: utf-8
from typing import Tuple, List

import torch
from torch import Tensor, nn
from utils.box_utils import match, decode, ciou


class YoloLoss(nn.Module):
    """ 损失函数 """

    def __init__(self, anchors: list, n_classes: int, image_size: int, overlap_thresh=0.5, label_smooth=0):
        """
        Parameters
        ----------
        anchors: list of shape `(3, n_anchors, 2)`
            先验框列表，尺寸从大到小

        n_classes: int
            类别数

        image_size: int
            输入神经网络的图片大小

        overlap_thresh: float
            视为忽视样例的 IOU 阈值

        label_smooth: float
            标签平滑系数，取值在 0.01 以下
        """
        super().__init__()
        self.anchors = anchors
        self.n_classes = n_classes
        self.image_size = image_size
        self.overlap_thresh = overlap_thresh
        self.label_smooth = label_smooth

        # 损失函数各个部分的权重
        self.balances = [0.4, 1, 4]
        self.lambda_box = 0.05
        self.lambda_obj = 5*image_size**2/416**2
        self.lambda_noobj = self.lambda_obj
        self.lambda_cls = n_classes / 80

        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, preds: Tuple[Tensor], targets: List[Tensor]):
        """
        Parameters
        ----------
        preds: Tuple[Tensor]
            Yolo 神经网络输出的各个特征图，维度为:
            * `(N, (n_classes+5)*n_anchors, 13, 13)`
            * `(N, (n_classes+5)*n_anchors, 26, 26)`
            * `(N, (n_classes+5)*n_anchors, 52, 52)`

        targets: List[Tensor]
            标签数据，每个标签张量的维度为 `(N, n_objects, 5)`，最后一维的第一个元素为类别，剩下为边界框 `(cx, cy, w, h)`

        Returns
        -------
        loc_loss: Tensor
            定位损失

        conf_loss: Tensor
            置信度损失

        cls_loss: Tensor
            分类损失
        """
        loc_loss = 0
        conf_loss = 0
        cls_loss = 0

        for anchors, pred, balance in zip(self.anchors, preds, self.balances):
            N, _, img_h, img_w = pred.shape

            # 对预测结果进行解码，shape: (N, n_anchors, H, W, n_classes+5)
            pred = decode(pred, anchors, self.n_classes, self.image_size)

            # 获取特征图最后一个维度的每一部分
            conf = pred[..., 4]
            cls = pred[..., 5:]

            # 匹配边界框
            step_h = self.image_size/img_h
            step_w = self.image_size/img_w
            anchors = [[i/step_w, j/step_h] for i, j in anchors]
            p_mask, n_mask, gt, _ = match(
                anchors, targets, img_h, img_w, self.n_classes, self.overlap_thresh)

            p_mask = p_mask.to(pred.device)
            n_mask = n_mask.to(pred.device)
            gt = gt.to(pred.device)

            # 定位损失
            m = p_mask == 1
            iou = ciou(pred[..., :4], gt[..., :4])
            loc_loss += torch.mean((1-iou)[m])*self.lambda_box

            # 置信度损失
            conf_loss += (self.bce_loss(conf*p_mask, p_mask)*self.lambda_obj +
                          self.bce_loss(conf*n_mask, 0*n_mask)*self.lambda_noobj)*balance

            # 分类损失
            cls_loss += self.bce_loss(cls[m], gt[..., 5:][m])*self.lambda_cls

        return loc_loss, conf_loss, cls_loss
