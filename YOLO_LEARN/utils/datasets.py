# Dataset utils and dataloaders

import glob
import logging
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from 物体检测.YOLO_LEARN.utils.general import xywhn2xyxy, xyn2xy, resample_segments, segment2box
from 物体检测.YOLO_LEARN.utils.torch_utils.utils import torch_distributed_zero_first

help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
logger = logging.getLogger(__name__)
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':###图像的方向
        break


def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix=''):
    with torch_distributed_zero_first(rank):#确保只有主进程（rank == 0）先执行数据集加载，其余进程等待，避免重复下载或缓存冲突
        '''
        使用 LoadImagesAndLabels 加载图像和标签。
        rect: 控制是否使用矩形训练（保持原始比例）。
        cache_images: 控制是否将图像缓存进内存或磁盘。
        image_weights: 是否按类别分布进行加权抽样。
        prefix: 日志前缀，便于调试输出。
        '''
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # 训练增强,是否启用数据增强（如随机透视变换、翻转）
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.single_cls,  #是否将所有类别视为单一类别
                                      stride=int(stride),##下采样步长（grid size），用于调整图像为该值的倍数
                                      pad=pad,
                                      image_weights=image_weights,
                                      prefix=prefix)


    batch_size = min(batch_size, len(dataset))
    '''
    os.cpu_count() 获取 CPU 核心数。
    分布式训练时，每个进程只使用部分线程。
    最终线程数是三者中的最小值，防止资源占用过高。
        '''
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    '''
    在 DDP 模式下，每个进程只能访问属于自己的一份数据子集。
使用 DistributedSampler 来划分数据，保证不同进程之间不重复。
#     '''
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    '''
    默认使用 PyTorch 自带的 DataLoader。
如果启用了 image_weights，则使用自定义的 InfiniteDataLoader，它可以无限循环地迭代数据。'''
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    '''
#
#     作用
#     dataset 已经加载好的 LoadImagesAndLabels 对象
#     batch_size 每个 batch 包含的图像数量
#     num_workers  并行加载图像的线程数
#     sampler 分布式训练使用的样本划分器
#     pin_memory=True 将数据加载到 pinned memory，提升 GPU 拷贝效率
#     collate_fn:自定义拼接函数，控制如何组合一个 batch 的数据,LoadImagesAndLabels正常batch，LoadImagesAndLabels.collate_fn4: 若启用 --quad，每个 batch 是原来的 4x 大小，适合测试加速。
#     '''
    dataloader = loader(dataset,
                        batch_size=batch_size,
                        num_workers=nw,
                        sampler=sampler,
                        pin_memory=True,
                        collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset

class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    InfiniteDataLoader 是一个 可以无限迭代的数据加载器，它能：
永远不会耗尽数据（即使数据量不足一个 epoch）
在分布式训练中表现良好（DDP）
更好地支持图像权重抽样（image weights sampling）
避免传统 DataLoader 最后一个 batch 被舍弃的问题
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
class _RepeatSampler(object):#是一个 无限重复的采样器
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return ['txt'.join(x.replace(sa, sb, 1).rsplit(x.split('.')[-1], 1)) for x in img_paths]
def exif_size(img):
    '''
    函数的主要功能是读取图像的 EXIF 数据中的方向信息，并根据方向调整图像的宽高，确保返回的尺寸与实际显示一致。以下是完整的函数逻辑：

    :param img:
    :return:
    '''
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  #  # 顺时针旋转270度
            s = (s[1], s[0])# 宽高互换
        elif rotation == 8:  # 顺时针旋转90度
            s = (s[1], s[0]) # 宽高互换
    except:
        pass

    return s
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        print(s,x,y)

        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh
def get_hash(files):
    # Returns a single hash value of a list of files获取文件大小
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))

def load_image(self, index):
    # loads 1 image from dataset, returns img, original hw, resized hw
    img = self.imgs[index]###尝试从缓存中获取图像
    if img is None:  # 判断图像是否已缓存
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # 计算缩放比例 r，使得图像的长边等于目标尺寸 self.img_size。
        if r != 1:  # 只有当缩放比例不为 1 时才进行缩放操作。训练时可以上采样，验证/测试阶段只下采样。
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR      #采样（缩小）：使用 cv2.INTER_AREA（效果更好），上采样（放大）或训练增强时：使用 cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)  #按比例 r 缩放图像，保持纵横比不变。
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        '''img: 缩放后的图像
(h0, w0): 原始图像高度和宽度
img.shape[:2]: 缩放后的图像高度和宽度'''
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized
def bbox_ioa(box1, box2):
    # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

    # Intersection over box2 area
    return inter_area / box2_area
def copy_paste(img, labels, segments, probability=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    '''实现了 Copy-Paste 数据增强技术，其核心思想是将一张图像中的目标对象（例如检测框和对应的分割区域）复制并粘贴到另一张图像的不同位置，从而生成更加多样化的训练样本
        提高小目标的检测能力
        增加背景复杂度
        防止模型对某些固定场景过拟合。'''
    n = len(segments)
    if probability and n:
        h, w, c = img.shape  # height, width, channels
        im_new = np.zeros(img.shape, np.uint8)   #创建一个与原图相同大小的黑色图像模板，用于绘制待粘贴的目标区域。
        for j in random.sample(range(n), k=round(probability * n)): #3. 随机选择部分目标进行粘贴
            l, s = labels[j], segments[j] #4. 获取标签和分割信息
            box = w - l[3], l[2], w - l[1], l[4] #4. 获取标签和分割信息
            ioa = bbox_ioa(box, labels[:, 1:5])  #  计算 IoA（交并比）
            if (ioa < 0.30).all():  # 如果 IoA < 0.3，表示遮挡较小，允许粘贴
                '''将新粘贴的目标添加进 labels 和 segments
                    对分割点也做相应的水平翻转处理'''
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                '''. 绘制掩码并融合图像'''
                cv2.drawContours(im_new, [segments[j].astype(int)], -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=img, src2=im_new)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        img[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug

    return img, labels, segments
def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

def random_perspective(img, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    '''

    :param img: 输入图像 (H x W x C)
    :param targets: 标签数组
    :param segments: 实例分割点集
    :param degrees: 图像旋转角度范围（±度数）
    :param translate: 平移比例（相对于图像大小）
    :param scale:缩放因子（如 0.9~1.1）
    :param shear:剪切角度（±度数）
    :param perspective: 透视变换系数（0 表示关闭）
    :param border: 边界填充大小
    :return:
    '''
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    '''在图像四周添加边框，防止变换后裁剪掉重要内容。'''
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # . 定义变换矩阵：中心平移
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # 透视变换（可选）
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # 旋转和缩放
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1.1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # . 剪切变换添加剪切变换，使图像产生“斜拉”效果。
    # 将角度转换为弧度并取正切值，构造剪切矩
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # 最终平移 添加最终的随机平移，使图像位置发生变化。
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # or. 组合所有变换 将所有变换矩阵按顺序组合成一个总变换矩阵 ：矩阵乘法顺序是 从右到左（先中心平移 → 透视 → 旋转缩放 → 剪切 → 最终平移）。
    '''应用变换到图像,如果有变化（如边框不为 0 或变换矩阵不是单位矩阵），则应用变换。
cv2.warpPerspective 用于透视变换，cv2.warpAffine 用于仿射变换。
默认填充颜色为灰色 (114, 114, 114)。'''
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    '''如果有标签，则使用变换矩阵更新其坐标。
        分为两种情况：
        使用分割掩码：对每个分割点进行变换，然后转为边界框。
        仅使用边界框：将边界框的四个角点展开为 8 个点，变换后再重新计算新的边界框。
        最后通过 box_candidates 筛选出有效的目标（面积足够大、宽高比合理等）。
        步骤  功能
        C将图像中心移到原点
        P添加透视变换
        R添加旋转和缩放
        S添加剪切变换
        T添加最终平移
        M所有变换的组合
        warp*应用变换到图像
        @ M.T应用变换到标签坐标
        box_candidates过滤无效边界框'''
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets
def load_mosaic(self, index):##4图拼接的 Mosaic 增强，用于训练时的数据增强。
    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # 马赛克中心点坐标，通过随机方式生成
    indices = [index] + random.choices(self.indices, k=3)  #当前图像索引 + 其他三张随机选取的图像索引
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)  放置的位置
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image) 表示从原始图像中裁剪的位置。
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  #  归一化格式 (normalized xywh) 转换为 绝对像素坐标 (pixel xyxy)。
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]  #对每个分割点集应用 xyn2xy，将归一化坐标转换为像素坐标
        labels4.append(labels)
        segments4.extend(segments)
    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)##将多个图像的标签合并成一个 (n, 5) 的 NumPy 数组。labels4[:, 1:] 表示所有目标的边界框坐标（x, y, w, h）。
    for x in (labels4[:, 1:], *segments4): #对标签框和分割点进行坐标裁剪，确保它们不超出图像范围。
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    #img4, labels4, segments4 = remove_background(img4, labels4, segments4)
    #sample_segments(img4, labels4, segments4, probability=self.hyp['copy_paste'])
    img4, labels4, segments4 = copy_paste(img4, labels4, segments4, probability=self.hyp['copy_paste'])#数据增强
    img4, labels4 = random_perspective(img4, labels4, segments4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  #  随机透视变换
    return img4, labels4
def load_mosaic9(self, index):
    # loads images in a 9-mosaic

    labels9, segments9 = [], []
    s = self.img_size
    indices = [index] + random.choices(self.indices, k=8)  # 8 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img9
        if i == 0:  # center
            img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
        labels9.append(labels)
        segments9.extend(segments)

        # Image
        img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
        hp, wp = h, w  # height, width previous
    # Offset
    yc, xc = [int(random.uniform(0, s)) for _ in self.mosaic_border]  # mosaic center x, y
    img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

    # Concat/clip labels
    labels9 = np.concatenate(labels9, 0)
    labels9[:, [1, 3]] -= xc
    labels9[:, [2, 4]] -= yc
    c = np.array([xc, yc])  # centers
    segments9 = [x - c for x in segments9]

    for x in (labels9[:, 1:], *segments9):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img9, labels9 = replicate(img9, labels9)  # replicate

    # Augment
    #img9, labels9, segments9 = remove_background(img9, labels9, segments9)
    img9, labels9, segments9 = copy_paste(img9, labels9, segments9, probability=self.hyp['copy_paste'])
    img9, labels9 = random_perspective(img9, labels9, segments9,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    return img9, labels9
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    '''用于 将图像缩放并填充为固定大小，同时保持其原始宽高比。这个函数通常在推理阶段使用，以确保输入图像的尺寸与模型期望的一致，而不会导致图像变形。'''
    '''
    auto:自动选择填充方式（默认启用）,scaleFill:是否拉伸填满图像（忽略纵横比）,scaleup:是否允许上采样（放大图像）

,
    '''
    # 获取原图尺寸
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old) 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)只下采样，不上采样
        r = min(r, 1.0)

    # Compute padding计算新图像尺寸（保持纵横比）
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle缩放图像
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch填充边缘
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    '''
    图像左右/上下各填充一半，使图像在新尺寸中居中。
避免图像偏向一侧。
    '''
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    '''果原图缩放后的尺寸与目标尺寸不同，则进行缩放。'''
    if  shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    '''计算上下左右边框'''
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    '''
    使用 OpenCV 的 copyMakeBorder 函数添加边框。.不改变图像内容，仅在四周添加灰度边框
    img: 经过缩放和填充后的图像。
ratio: 缩放比例（可用于还原原始坐标）。
(dw, dh): 水平和垂直方向的总填充像素（可用于后续边界框坐标的调整）。
    '''
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def augment_hsv(img,hgain=0.5,sgain=0.5,vgain=0.5):
    '''功能：生成三个通道（Hue、Saturation、Value）的随机增益。
np.random.uniform(-1, 1, 3)：生成 [rh, rs, rv] 三个随机数。
* [hgain, sgain, vgain] + 1：
hgain: Hue 偏移范围（如 0.015）
sgain: Saturation 缩放因子（如 0.7）
vgain: Value 缩放因子（如 0.4）
最终得到类似 [1.01, 0.9, 1.3] 的变换参数。'''
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # r生成随机增益
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))#将图像从 BGR 转换为 HSV 并分离通道
    dtype = img.dtype  # uint8保存原始图像类型
    '''功能：创建一个 0~255 的数组，用于构建查找表（Look-Up Table）。
所有像素值都会根据这个数组做映射。'''
    x = np.arange(0, 256, dtype=np.int16)
    '''
    功能：构建色调变换的查找表。
注意：Hue 通道取值范围是 [0, 180]，所以要模 180。
r[0] 是 Hue 的增益系数。

构建饱和度变换的查找表。
使用 np.clip(..., 0, 255) 防止数值越界。

功能：构建明度变换的查找表。
和 Saturation 类似，也使用 clip 防止溢出
    '''
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
    #应用 LUT 变换到各通道
    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    ###将 HSV 转换回 BGR
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def sample_segments(img, labels, segments, probability=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    '''

    :param img: 经过 Mosaic 拼接后的图像
    :param labels: 所有拼接图像的标签 [cls, x_center, y_center, w, h]
    :param segments: 分割点集，每个元素是一个 (n_points, 2) 的坐标数组
    :param probability: 控制采样的概率，例如 0.5 表示 50% 的目标会被采样
    :return:
    '''
    n = len(segments)
    sample_labels = []
    sample_images = []
    sample_masks = []
    if probability and n:
        h, w, c = img.shape  # height, width, channels
        for j in random.sample(range(n), k=round(probability * n)):#. 随机选择部分目标进行采样
            l, s = labels[j], segments[j]
            box = l[1].astype(int).clip(0, w - 1), l[2].astype(int).clip(0, h - 1), l[3].astype(int).clip(0, w - 1), l[
                4].astype(int).clip(0, h - 1)#裁剪边界框以防止越界。

            # print(box)
            if (box[2] <= box[0]) or (box[3] <= box[1]):
                continue

            sample_labels.append(l[0])
            '''创建一个与原图大小相同的全黑掩码图像。
在对应分割点绘制白色填充轮廓（255, 255, 255）
这样得到的是目标对象的二值掩码'''
            mask = np.zeros(img.shape, np.uint8)

            cv2.drawContours(mask, [segments[j].astype(int)], -1, (255, 255, 255), cv2.FILLED)
            #. 裁剪掩码和图像区域
            '''
            将掩码根据边界框裁剪出来，用于粘贴。
使用 bitwise_and 提取掩码覆盖的图像区域。
将提取的图像区域也保存下来，便于后续融合
            '''
            sample_masks.append(mask[box[1]:box[3], box[0]:box[2], :])

            result = cv2.bitwise_and(src1=img, src2=mask)
            i = result > 0  # pixels to replace
            mask[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug
            # print(box)
            sample_images.append(mask[box[1]:box[3], box[0]:box[2], :])

    return sample_labels, sample_images, sample_masks
def load_samples(self, index):
    # loads images in a 4-mosaic

    labels4, segments4 = [], []
    s = self.img_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels, segments = self.labels[index].copy(), self.segments[index].copy()
        if labels.size:
            labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padw, padh)  # normalized xywh to pixel xyxy format
            segments = [xyn2xy(x, w, h, padw, padh) for x in segments]
        labels4.append(labels)
        segments4.extend(segments)

    # 凹形/剪切标签
    labels4 = np.concatenate(labels4, 0)
    for x in (labels4[:, 1:], *segments4):
        np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
    # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    #img4, labels4, segments4 = remove_background(img4, labels4, segments4)从当前图像中随机采样一些分割目标
    sample_labels, sample_images, sample_masks = sample_segments(img4, labels4, segments4, probability=0.5)

    return sample_labels, sample_images, sample_masks


def pastein(image, labels, sample_labels, sample_images, sample_masks):
    '''
    添加了来自其他图像的目标，提升背景复杂度

    开始
 │
 ├─ 获取当前图像尺寸 → h, w
 │
 ├─ 定义多尺度粘贴策略 → scales
 │
 ├─ 遍历每个 scale
 │     ├── 随机决定是否跳过
 │     ├── 计算粘贴区域 box
 │     └── 计算 IoA，判断是否遮挡过多
 │
 ├─ 随机选择一个样本进行粘贴
 │     ├── 缩放样本图像和掩码 → resize
 │     ├── 融合图像区域 → bitwise_and
 │     └── 应用掩码 → mask > 0
 │
 ├─ 更新标签信息
 │     ├── 已有标签 → concatenate
 │     └── 无标签 → 创建新标签数组
 │
 └─ 返回最终图像和标签
    :param image:
    :param labels:
    :param sample_labels:
    :param sample_images:
    :param sample_masks:
    :return:
    '''
    # img: 当前图像 (H x W x C)
    # labels: 当前图像原有的标签 [cls, x1, y1, x2, y2]
    # sample_labels: {0}#L906-L906)  # sample_images 和 sample_masks 是从其他图像中提取的目标区域
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    # create random masks
    scales = [0.75] * 2 + [0.5] * 4 + [0.25] * 4 + [0.125] * 4 + [0.0625] * 6  # image size fraction
    for s in scales:
        if random.random() < 0.2:
            continue
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        if len(labels):
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
        else:
            ioa = np.zeros(1)

        if (ioa < 0.30).all() and len(sample_labels) and (xmax > xmin + 20) and (
                ymax > ymin + 20):  # allow 30% obscuration of existing labels
            sel_ind = random.randint(0, len(sample_labels) - 1)
            # print(len(sample_labels))
            # print(sel_ind)
            # print((xmax-xmin, ymax-ymin))
            # print(image[ymin:ymax, xmin:xmax].shape)
            # print([[sample_labels[sel_ind], *box]])
            # print(labels.shape)
            hs, ws, cs = sample_images[sel_ind].shape
            r_scale = min((ymax - ymin) / hs, (xmax - xmin) / ws)
            r_w = int(ws * r_scale)
            r_h = int(hs * r_scale)

            if (r_w > 10) and (r_h > 10):
                r_mask = cv2.resize(sample_masks[sel_ind], (r_w, r_h))
                r_image = cv2.resize(sample_images[sel_ind], (r_w, r_h))
                temp_crop = image[ymin:ymin + r_h, xmin:xmin + r_w]
                m_ind = r_mask > 0
                if m_ind.astype(np.int32).sum() > 60:
                    temp_crop[m_ind] = r_image[m_ind]
                    # print(sample_labels[sel_ind])
                    # print(sample_images[sel_ind].shape)
                    # print(temp_crop.shape)
                    box = np.array([xmin, ymin, xmin + r_w, ymin + r_h], dtype=np.float32)
                    if len(labels):
                        labels = np.concatenate((labels, [[sample_labels[sel_ind], *box]]), 0)
                    else:
                        labels = np.array([[sample_labels[sel_ind], *box]])

                    image[ymin:ymin + r_h, xmin:xmin + r_w] = temp_crop

    return labels
class LoadImagesAndLabels(Dataset):
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.stride = stride
        self.path = path
        self.mosaic = self.augment and not self.rect  # 一次加载4张图像到马赛克中（仅在训练期间）
        self.mosaic_border = [-img_size // 2, -img_size // 2]

        '''读取图片txt文件的路径'''

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('**/*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, 'r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                else:
                        raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in img_formats])

        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

        self.label_files = img2label_paths(self.img_files)  # 使用 img2label_paths 函数将图像路径替换为对应的 .txt 标签路径。
        ''''''
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')  # cached labels

        if cache_path.is_file():
            cache, exists = torch.load(cache_path,weights_only=False), True  # load
            #if cache['hash'] != get_hash(self.label_files + self.img_files) or 'version' not in cache:  # changed
            #    cache, exists = self.cache_labels(cache_path, prefix), False  # re-cache
        else:
            cache, exists = self.cache_labels(cache_path, prefix), False  # cache
        ##cache_labes:标签坐标，图片大小，图片路径，hash,( nf, nm, ne, nc, n),版本
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupted, total
        '''缓存图像（加速训练）'''
        if exists:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted"
            tqdm(None, desc=prefix + d, total=n, initial=n)  # display cache results
        assert nf > 0 or not augment, f'{prefix}No labels in {cache_path}. Can not train without labels. See {help_url}'

        # 读取缓存，去掉不要的字段
        cache.pop('hash')  # remove hash
        cache.pop('version')  # remove version
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)##标签坐标
        self.shapes = np.array(shapes, dtype=np.float64)##图片大小
        self.img_files = list(cache.keys())  # 图像文件
        self.label_files = img2label_paths(cache.keys())  # 标签文件
        if single_cls:##判断是否启用了“单类检测”模式。
            for x in self.labels:
                x[:, 0] = 0
        n = len(shapes)## 是图像总数
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # 为每张图像计算其所属的 batch 编号。
        nb = bi[-1] + 1  # 计算总共有多少个 batch。
        self.batch = bi
        self.n = n
        self.indices = range(n)#将 batch 索引、图像总数、索引范围保存到类变量中供后续使用。
        '''矩形训练，提升推理用于验证数据'''
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # 是宽高比 h/w（注意：在代码中顺序是 wh，所以取 s[:, 1]/s[:, 0]）
            irect = ar.argsort()  ##则按照宽高比（aspect ratio）对图像进行排序。
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect] ##根据宽高比重新排序所有图像路径、标签路径、标签内容和图像尺寸。确保每个 batch 内部的图像具有相似的长宽比，减少 padding 浪费，加快训练速度。

            # 为每个 batch 设置统一的目标尺寸（保持长宽比一致）。让同一批图像 resize 后的 shape 更接近，节省内存并提高 GPU 利用率。
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            '''基于目标比例和步长（stride）计算最终每个 batch 的目标尺寸。
            mg_size 是网络输入大小（如 640）;stride 是模型下采样步长（如 32）;pad 是额外填充量
            最终 batch_shapes 是 (nb, 2) 形状的数组，表示每个 batch 的 (height, width)'''
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride

        #图像缓存 Image Caching（可选，用于加速训练）
        self.imgs = [None] * n #初始化一个长度为 n 的列表，用于缓存加载后的图像数据。
        print(cache_images)
        if cache_images:
            if cache_images == 'disk':
                self.im_cache_dir = Path(Path(self.img_files[0]).parent.as_posix() + '_npy')
                self.img_npy = [self.im_cache_dir / Path(f).with_suffix('.npy').name for f in self.img_files]
                self.im_cache_dir.mkdir(parents=True, exist_ok=True)
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: load_image(*x), zip(repeat(self), range(n)))
            pbar = tqdm(enumerate(results), total=n) #使用线程池并发加载图像
            for i, x in pbar: #将图像缓存到内存或磁盘。
                if cache_images == 'disk':
                    if not self.img_npy[i].exists():
                        np.save(self.img_npy[i].as_posix(), x[0])
                    gb += self.img_npy[i].stat().st_size
                else:
                    self.imgs[i], self.img_hw0[i], self.img_hw[i] = x
                    gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, index):
        index = self.indices[index]  # linear, shuffled, or image_weights
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:###数据增强
            # Load mosaic
            if random.random() < 0.8:###随机选择 4 图或 9 图马赛克增强
                img, labels = load_mosaic(self, index)
            else:
                img, labels = load_mosaic9(self, index)
            shapes = None##清除形状信息，表示图像尺寸不一致

            '''
            开始
         │
         ├─ 随机选择 Mosaic 类型：
         │     └── 80% → load_mosaic (4图拼接)
         │     └── 20% → load_mosaic9 (9图拼接)
         │
         ├─ shapes = None （表示图像尺寸不统一）
         │
         ├─ 是否启用 Mixup?
         │     ├── 否 → 返回当前 img, labels
         │     └── 是：
         │           ├─ 再选一张图（80% mosaic / 20% mosaic9）
         │           ├─ 用 beta 分布混合两张图像
         │           └─ 合并两个图像的标签
         │
        返回最终图像 img 和标签 labels
            '''
            if random.random() < hyp['mixup']:#判断是否启用 Mixup 增强
                '''再选一张图像用于 Mixup'''
                if random.random() < 0.8:
                    img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
                else:
                    img2, labels2 = load_mosaic9(self, random.randint(0, len(self.labels) - 1))
                '''生成 Mixup 权重系数'''
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                '''执行图像像素级融合'''
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                '''合并两个图像的标签'''
                labels = np.concatenate((labels, labels2), 0)
        else:
            # Load image
            '''
                         │
             ├─ 加载图像及原始尺寸 → load_image
             │
             ├─ 获取目标尺寸 → self.batch_shapes 或 self.img_size
             │
             ├─ 缩放并填充图像 → letterbox
             │
             ├─ 保存图像变换信息 → shapes
             │
             ├─ 获取标签 → self.labels[index]
             │
             └─ 转换标签坐标 → xywhn2xyxy（考虑缩放和填充）
            '''
            img, (h0, w0), (h, w) = load_image(self, index)

            # 缩放填充图像
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self.labels[index].copy()
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
        if self.augment:  ##如果训练器增强
            # Augment imagespace
            if not mosaic:
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])  ###对输入图像进行 随机透视变换 / 仿射变换

            # img, labels = self.albumentations(img, labels)

            #扩大色彩空间
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

            if random.random() < hyp['paste_in']:
                sample_labels, sample_images, sample_masks = [], [], []
                while len(sample_labels) < 30:##样本标签太少的时候
                    sample_labels_, sample_images_, sample_masks_ = load_samples(self, random.randint(0,
                                                                                                      len(self.labels) - 1))
                    sample_labels += sample_labels_
                    sample_images += sample_images_
                    sample_masks += sample_masks_
                    # print(len(sample_labels))
                    if len(sample_labels) == 0:
                        break
                '''是 YOLO 模型中用于 数据增强 的关键操作之一，它的作用是将之前通过 sample_segments 提取的目标图像和掩码 粘贴（paste）到当前图像上，
                并更新标签信息。这种增强方式叫做 PasteIn 增强，与 Copy-Paste 类似，但 PasteIn 更强调从多张图像中提取目标后融合。'''
                labels = pastein(img, labels, sample_labels, sample_images, sample_masks)

        nL = len(labels)  # number of labels 计算当前图像中目标的数量
        if nL: #将边界框转换为 (x_center, y_center, w, h) 格式
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        if self.augment:#判断是否启用训练增强
            # flip up-down
            if random.random() < hyp['flipud']:  #随机上下翻转翻转概率
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if random.random() < hyp['fliplr']:#随机左右翻转，labels[:, 1]: x_center 坐标，翻转后也需要取反（1 - x_center）
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nL, 6))#构建输出标签张量
        if nL:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416 BGR → RGB 并调整通道顺序.transpose(2, 0, 1): 将图像维度从 (H, W, C) 转换为 (C, H, W)，便于 PyTorch 处理
        img = np.ascontiguousarray(img)##确保内存连续性

        return torch.from_numpy(img), labels_out, self.img_files[index], shapes






    def cache_labels(self, path=Path('./labels.cache'), prefix=''):
        '''
        该函数的作用是将标签（label）信息预先解析并缓存到内存中，避免每次迭代时重复读取和解析文件，从而提升性能。
        :param path:
        :param prefix:
        :return:
        nm（no label missing）：
        作用：记录标签文件缺失的数量。
        解释：当找不到与图像对应的标签文件时，此值会增加。
        nf（number found labels）：
        作用：记录成功找到的标签文件数量。
        解释：每当正确读取一个标签文件时，此值会增加。
        ne（number empty labels）：
        作用：记录空标签文件的数量。
        解释：如果标签文件存在但没有任何内容（即没有标注目标），此值会增加。
        nc（number corrupted labels）：
        作用：记录损坏或无效的图像/标签对的数量。
        解释：当图像或标签文件由于某些原因无法正常处理时（如尺寸小于阈值、格式不支持等），此值会增加。
        '''
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0  # 数字丢失、找到、为空、重复
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                im = Image.open(im_file)
                im.verify()#对图像进行验证的过程，确保图像数据的完整性和正确性。
                shape = exif_size(im)#函数来校正图像方向后返回图像的宽高。
                segments = []#用于存储实例分割信息。
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'#确保图像尺寸大于 10x10 像素。
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'
                if os.path.isfile(lb_file):#检查是否存在对应的标签文件，如果存在则增加 nf 计数器。
                    nf += 1  # label found
                    with open(lb_file, 'r') as f:
                        l = [x.split() for x in f.read().strip().splitlines()]

                        '''暂时没有分割信息所以用不到'''
                        if any([len(x) > 8 for x in l]):  # 如果标签包含分割信息（每行超过 8 个数值），则将其转换为边界框。
                            classes = np.array([x[0] for x in l], dtype=np.float32)#classes 提取类别信息。
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]  #  提取分割点并转换为 NumPy 数组。 提取分割点并转换为 NumPy 数组。
                            print('classes:',classes,'segments:',segments,classes.reshape(-1, 1), segments2boxes(segments))
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)#将分割点转换为边界框坐标。

                        l = np.array(l, dtype=np.float32)

                    '''
                    作用：对标签进行一系列验证：
                    标签必须有 5 列（类别 + 4 个归一化坐标）。
                    所有标签值必须非负。
                    归一化坐标必须在 [0, 1] 范围内。
                    确保没有重复的标签。
                    '''
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1  # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                else:
                    nm += 1  # label missing
                    l = np.zeros((0, 5), dtype=np.float32)
                x[im_file] = [l, shape, segments]
                print('segments--,',segments)

            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')
            pbar.desc = f"{prefix}Scanning '{path.parent / path.stem}' images and labels... " \
                        f"{nf} found, {nm} missing, {ne} empty, {nc} corrupted"
        pbar.close()
        if nf == 0:
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')
        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version
        torch.save(x, path)  # save for next time
        logging.info(f'{prefix}New cache created: {path}')

        return x

    @staticmethod
    def collate_fn(batch):#数据加载器，多个样本为一个batch,标准的 batch 合并方式（每张图一张标签）,hapes: 缩放信息
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):#四图拼接增强（Quad）下的 batch 合并方式
        img, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4#计算实际有多少组四图拼接的图像（例如：batch_size=16 → n=4 组 quad）
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        '''每次处理 4 张图：
一半概率放大第一张图（用插值）
一半概率把 4 张图横向拼接为 1 张图'''
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW主循环：组合 4 张图
            i *= 4
            if random.random() < 0.5:
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2., mode='bilinear', align_corners=False)[
                    0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i  # add target image index for build_targets()

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4

