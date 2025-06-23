import math
import random
import time
from copy import copy, deepcopy
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import requests
import torch
import torchvision
from PIL import Image
from click.core import F
from pybaseutils.coords_utils import xywh2xyxy, xyxy2xywh
from torch import nn, amp
from torch.nn import SiLU
from torchvision.ops import box_iou

from 深度网络架构pytorch.text import logger
from 物体检测.YOLO_LEARN.models.dataseets import letterbox
from 物体检测.YOLO_LEARN.utils.general import increment_path, non_max_suppression, make_divisible, scale_coords
from 物体检测.YOLO_LEARN.utils.torch_utils.utils import time_synchronized, color_list, plot_one_box


##padding
class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad  添加padding
    return p
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):   #groups （int， optional） – 从输入通道到输出通道的阻塞连接数。默认值：1
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self,x):  ##普通前向传播 (forward
        return self.act(self.bn(self.conv(x)))
    def fuseforward(self,x):###正对yolo里面的已经融合的模型前向传播
        return self.act(self.conv(x))
class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))
class RepConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()
        self.deploy=deploy   ###测试
        self.groups=g
        self.in_channels=c1
        self.out_channels=c2

        assert k==3
        assert autopad(k,p)==1

        padding_1= autopad(k, p) - k // 2
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)#残差连接分支

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )##3*3卷积提取局部特征

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_1, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self,inputs):
        if hasattr(self, "rbr_repgram"):return self.act(self.rbr_reparam(inputs))##推理阶段（已融合）：
        if self.rbr_identity is None:id_out = 0
        else:id_out = self.rbr_identity(inputs)##训练阶段：
        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:  return 0
        else:nn.functional.pad(kernel1x1,[1,1,1,1])
    def _fuse_bn_tensor(self, branch):
        '''
        融合权重和偏执
        输入：一个包含 Conv + BN 的分支。
输出：融合后的卷积核 [weight, bias]。
        :param branch:
        :return:
        '''
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),)
    def fuse_conv_bn(self,conv,bn):
        std=(bn.running_var+bn.eps).sqrt()
        bias=bn.bias- bn.running_mean * bn.weight / std
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t
        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)
        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv


    def fuse_repvgg_block(self):  #来启用推理优化路径。
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])  #2. 融合 rbr_dense 分支（3x3 Conv + BN）

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        '''获取 1x1 分支的偏置，并扩展其卷积核为 3x3'''
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])
        '''处理 Identity 分支（即残差连接）'''
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                        nn.modules.batchnorm.SyncBatchNorm)):
            # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()###去除大小为1的对角线。最终变成二维矩阵，方便设置对角线值。
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
            identity_conv_1x1.weight.data.fill_(0.0)#将整个权重矩阵填充为 0。，初始化为零矩阵后，再手动设置对角线为 1，构造单位矩阵。
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)##将权重矩阵的主对角线元素设为 1。得到一个单位矩阵，表示输入通道与输出通道一一对应。C_in == C_out == 64，那么现在权重是一个 64x64 的单位矩阵。
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)#将二维的单位矩阵 [C_out, C_in] 扩展为四维的卷积核 [C_out, C_in, 1, 1]。，.unsqueeze(2) → 在第 2 维增加一个维度 → [C_out, C_in, 1]，unsqueeze(3) → 在第 3 维增加一个维度 → [C_out, C_in, 1, 1]，构造出一个 1x1 卷积核
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            '''将所有分支的权重和偏置融合进 rbr_dense'''
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))
        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True
        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)
class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold  置信度
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)
class autoShape(nn.Module):
    '''
    这是一个输入鲁棒性的模型封装器，
    支持传入 OpenCV（cv2）、NumPy（np）、PIL（图像）
    或 Torch 张量作为输入。它包含了预处理、推理和非极大值抑制（NMS）功能。
    '''
    conf = 0.25  #置信度阈值，默认为 0.25。
    iou = 0.45  # IOU 阈值，默认为 0.45。
    classes = None  #可选参数，用于按类别过滤检测结果。

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()
    def autoshape(self):
        '''这个方法是为了兼容性设计的，如果已经启用了 autoshape，则打印提示并返回自身实例，避免重复包装。'''
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self
    '''使用 @torch.no_grad() 装饰器，表示在该函数中不会计算梯度，节省内存和计算资源。'''

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        '''
        :param imgs: 输入图像，可以是多种格式（如文件路径、OpenCV 图像、PIL 图像、NumPy 数组、Torch 张量等）。
        :param size: 输入图像的目标尺寸，默认为 640x640。
        :param augment:  是否启用数据增强，默认不启用。
        :param profile: 是否启用性能分析，默认不启用。
        :return:
        '''
        t= [time_synchronized()] #初始化一个时间戳列表 t，记录不同阶段的时间点，用于后续统计耗时。
        p = next(self.model.parameters())  # for device and type
        '''如果输入已经是 torch.Tensor 类型，则直接进行推理：
        使用混合精度 (amp.autocast) 加速推理（除非使用 CPU）。
        将输入张量移动到与模型相同的设备，并保持相同的数据类型。
        调用模型进行推理。'''
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference
        '''初始化三个空列表：
shape0: 原始图像形状。
shape1: 缩放后的图像形状。
files: 图像文件名（如果有）。'''
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames

        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # 如果输入是 PIL 图像对象，则将其转换为 NumPy 数组，并尝试获取图像的文件名
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)  #将文件名加入 files 列表，并确保后缀为 .jpg
            if im.shape[0] < 5:  # image in CHW 如果图像是通道优先（CHW）格式，则将其转置为高度宽度通道（HWC）格式。
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  #如果图像是三维（HWC），只取前三个通道（RGB）。
            s = im.shape[:2]  # HWC 如果是二维图像（灰度图），则将其扩展为三通道图像。
            shape0.append(s)  #获取当前图像的高度和宽度（HWC 格式），并存入 shape0。
            g = (size / max(s))  # 计算缩放因子 g，使图像的最大边等于目标尺寸 size。
            shape1.append([y * g for y in s])  #根据缩放因子 g 计算缩放后的图像尺寸 shape1
            imgs[i] = im  # 更新 imgs[i] 为处理后的图像数组。
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # 对所有图像的缩放后尺寸取最大值，并使用 make_divisible 使其可被最大步长整除，以满足模型对输入尺寸的要求。对所有图像的缩放后尺寸取最大值，并使用 make_divisible 使其可被最大步长整除，以满足模型对输入尺寸的要求。 shape

        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad 使用 letterbox 函数对每张图像进行填充和缩放，使其达到目标尺寸 shape1。
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack 如果有多张图像，则堆叠成一个批次；否则将单张图像增加一个批次维度。
        '''将图像从 BHWC（Batch, Height, Width, Channels）格式转换为 BCHW（Batch, Channels, Height, Width）格式，符合 PyTorch 输入要求'''
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        '''将 NumPy 数组转换为 PyTorch 张量。
        移动到模型所在设备（GPU 或 CPU）。
        数据类型与模型参数一致（通常是 float16 或 float32）。
        归一化像素值（0-255 → 0-1）。'''
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        '''记录预处理结束时间。'''
        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS 对模型输出执行 非极大值抑制（NMS），去除重叠过多的预测框。
            for i in range(n): #将预测框坐标从缩放后的尺寸恢复到原始图像尺寸（反归一化）。
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)

class Detections:
    ''' 封装模型推理后的检测结果，并提供一系列便捷的方法用于展示、保存、处理和分析这些检测结果。'''
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # 存储原始图像数组（numpy 格式）
        self.pred = pred  # 存储 NMS 后的检测结果列表，每个元素是一个张量，包含 of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class 类别名称列表，用于将类别编号映射为可读名称。
        self.files = files  #图像文件名列表。
        self.xyxy = pred  # 检测框的像素坐标（左上角 + 右下角），格式为 [x1, y1, x2, y2]。
        self.xywh = [xyxy2xywh(x) for x in pred]  # 检测框的像素坐标（中心点 + 宽高），格式为 [cx, cy, w, h]。
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # 归一化的 xyxy 坐标（相对于原始图像尺寸）。
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh 归一化的 xywh 坐标（相对于原始图像尺寸）
        self.n = len(self.pred)  # number of images (batch size)批次中的图像数量。
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)记录三个阶段的时间（预处理、推理、NMS），单位为毫秒
        self.s = shape  # inference BCHW shape推理输入的形状（BCHW）。

    def display(self, pprint=False, show=False, save=False, render=False, save_dir=''):##打印检测结果和各阶段耗时
        '''显示/保存/绘制检测结果。
        如果启用 pprint=True，则打印每张图的检测对象及数量。
        如果启用 render=True，则将带有检测框的图像渲染回原图。'''
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                img.show(self.files[i])  # show
            if save:
                f = self.files[i]
                img.save(Path(save_dir) / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp')  # increment save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.display(save=True, save_dir=save_dir)  # save results
    def render(self):#渲染检测结果，返回图像列表（可用于进一步处理或展示）
        self.display(render=True)  # render results
        return self.imgs
    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n

