# Plotting utils

import glob
import math
import os
import random
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, filtfilt

from 物体检测.YOLO_LEARN.utils.datasets import xyxy2xywh

# Settings
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only


from 物体检测.YOLO_LEARN.utils.general import xywh2xyxy
from 物体检测.YOLO_LEARN.utils.torch_utils.utils import color_list, plot_one_box


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
#     '''
#
#         该函数 `plot_images` 的功能是将一批图像（`images`）及其对应的目标标注或预测框（`targets`）绘制在一个拼接图（mosaic）上，用于可视化目标检测的结果。
#
# **具体功能如下：**
#
# 1. **输入处理与格式转换**：
#    - 如果输入是 PyTorch 张量（`torch.Tensor`），则将其转换为 NumPy 数组；
#    - 若图像数据被归一化到 [0,1] 范围，则反归一化到 [0,255]（即恢复为像素值范围）；
#
# 2. **参数设置**：
#    - 设置线条厚度 `tl` 和字体厚度 `tf`；
#    - 获取图像的批次大小 [bs](file://D:\项目\咕泡项目\深度网络架构pytorch\神经网络实战分类与回归任务\神经网路分类任务.py#L56-L56)、高度 [h](file://D:\项目\咕泡项目\大模型\调用mindspore\resnet_pet_classification-main\mindcv\models\volo.py#L0-L0)、宽度 [w](file://D:\项目\咕泡项目\大模型\调用mindspore\resnet_pet_classification-main\mindcv\models\volo.py#L0-L0)；
#    - 限制最多显示的子图数量为 `max_subplots`；
#    - 计算每个子图的行列数 `ns`（近似平方布局）；
#
# 3. **图像缩放控制**：
#    - 根据 [max_size](file://D:\项目\咕泡项目\物体检测\detr-master\datasets\transforms.py#L0-L0) 控制最大尺寸，计算缩放因子 `scale_factor`；
#    - 若需缩小图像，则对图像和对应的边界框进行缩放；
#
# 4. **初始化画布**：
#    - 使用 [color_list()](file://D:\项目\咕泡项目\物体检测\YOLO_LEARN\utils\torch_utils\utils.py#L105-L110) 创建颜色列表；
#    - 初始化一个白色背景的大图 [mosaic](file://D:\项目\咕泡项目\物体检测\YOLO_LEARN\utils\datasets.py#L0-L0)，用于拼接所有图像；
#
# 5. **图像拼接与标注绘制**：
#    - 遍历每张图像，将其调整尺寸后放入大图的指定位置；
#    - 若存在目标标注（`targets` 不为空），提取当前图像的标注信息；
#      - 将标注框从归一化坐标转为像素坐标；
#      - 若图像被缩放，则同步缩放边界框；
#      - 将边界框和类别标签绘制在对应图像区域；
#    - 若提供图像路径 [paths](file://D:\项目\咕泡项目\大模型\调用mindspore\resnet_pet_classification-main\mindcv\models\layers\selective_kernel.py#L0-L0)，则在图像左上角添加文件名标签；
#    - 绘制图像边框以区分不同图像；
#
# 6. **保存结果图像**：
#    - 对最终拼接图再次缩放以控制输出大小；
#    - 使用 PIL 库将图像保存为文件（默认为 `images.jpg`）；
#
# 7. **返回值**：
#    - 返回拼接后的图像 [mosaic](file://D:\项目\咕泡项目\物体检测\YOLO_LEARN\utils\datasets.py#L0-L0)（NumPy 数组格式）。
#     :param images:
#     :param targets:
#     :param paths:
#     :param fname:
#     :param names:
#     :param max_size:
#     :param max_subplots:
#     :return:
#     :param images:
#     :param targets:
#     :param paths:
#     :param fname:
#     :param names:
#     :param max_size:
#     :param max_subplots:
#     :return:
#     '''

    # Plot image grid with labels

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    colors = color_list()  # list of colors
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            labels = image_targets.shape[1] == 6  # labels if no conf column
            conf = None if labels else image_targets[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= w  # scale to pixels
                    boxes[[1, 3]] *= h
                elif scale_factor < 1:  # absolute coords need scale if image scales
                    boxes *= scale_factor
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = colors[cls % len(colors)]
                cls = names[cls] if names else cls
                if labels or conf[j] > 0.25:  # 0.25 conf thresh
                    label = '%s' % cls if labels else '%s %.1f' % (cls, conf[j])
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # Draw image filename labels
        if paths:
            label = Path(paths[i]).name[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)  # ratio to limit image size
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))  # cv2 save
        Image.fromarray(mosaic).save(fname)  # PIL save
    return mosaic


def plot_results(start=0, stop=0, bucket='', id=(), labels=(), save_dir=''):
    # Plot training 'results*.txt'. from utils.plots import *; plot_results(save_dir='runs/train/exp')
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['Box', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val Box', 'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']
    if bucket:
        # files = ['https://storage.googleapis.com/%s/results%g.txt' % (bucket, x) for x in id]
        files = ['results%g.txt' % x for x in id]
        c = ('gsutil cp ' + '%s ' * len(files) + '.') % tuple('gs://%s/results%g.txt' % (bucket, x) for x in id)
        os.system(c)
    else:
        files = list(Path(save_dir).glob('results*.txt'))
    assert len(files), 'No results.txt files found in %s, nothing to plot.' % os.path.abspath(save_dir)
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
            n = results.shape[1]  # number of rows
            x = range(start, min(stop, n) if stop else n)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # don't show zero loss values
                    # y /= y[0]  # normalize
                label = labels[fi] if len(labels) else f.stem
                ax[i].plot(x, y, marker='.', label=label, linewidth=2, markersize=8)
                ax[i].set_title(s[i])
                # if i in [5, 6, 7]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (f, e))

    ax[1].legend()
    fig.savefig(Path(save_dir) / 'results.png', dpi=200)

def output_to_target(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls in o.cpu().numpy():
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
    return np.array(targets)

def plot_study_txt(path='', x=None):  # from utils.plots import *; plot_study_txt()
    # Plot study.txt generated by test.py
    fig, ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)
    # ax = ax.ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    # for f in [Path(path) / f'study_coco_{x}.txt' for x in ['yolor-p6', 'yolor-w6', 'yolor-e6', 'yolor-d6']]:
    for f in sorted(Path(path).glob('study*.txt')):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_inference (ms/img)', 't_NMS (ms/img)', 't_total (ms/img)']
        # for i in range(7):
        #     ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
        #     ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(y[6, 1:j], y[3, 1:j] * 1E2, '.-', linewidth=2, markersize=8,
                 label=f.stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
             'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')

    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(30, 55)
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    plt.savefig(str(Path(path).name) + '.png', dpi=300)


