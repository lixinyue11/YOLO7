import datetime
import math
import os
import random
import subprocess
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import cv2
import matplotlib
import torch
from torch import nn

from 深度网络架构pytorch.text import logger


def select_device(device='',batch_size=None):
    cpu = device.lower() == 'cpu'
    if cpu:os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability
    cuda=not  cpu and torch.cuda.is_available()
    return torch.device('cuda:0' if cuda else 'cpu')
def time_synchronized():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()
import torch.nn.functional as F
def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean

def check_anchor_order(m):###网格大小，对比预测和真实的
    # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)
def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True
def model_info(model,verbose=False,img_size=400):#verbose: 是否输出详细信息，默认为 False
        n_p=sum(x.numel() for x in model.parameters())#返回张量中的元素总数。
        n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # 需要计算梯度的总和
        if verbose:
            print('%5s %40s %9s %12s %20s %10s %10s' % (
            'layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
            for i, (name, p) in enumerate(model.named_parameters()):##p.numel()参数数量
                name = name.replace('module_list.', '')
                print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                      (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
        try:  # FLOPS
            from thop import profile
            stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32#是模型中特征图的下采样步长。
            img = torch.zeros((1, model.yaml.get('ch', 3), stride, stride),
                              device=next(model.parameters()).device)  # 是一个虚拟输入张量，用于进行 FLOPS 计算。
            flops = profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1E9 * 2  # FLOPS 计，评估模型在不同输入尺寸下的计算需求。
            img_size = img_size if isinstance(img_size, list) else [img_size, img_size]  #整 FLOPS 的比例。
            fs = ', %.1f GFLOPS' % (flops * img_size[0] / stride * img_size[1] / stride)  # 640x640 GFLOPS
        except (ImportError, Exception):
            fs = ''

        logger.info(f"Model Summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}")
def fuse_conv_and_bn(conv,bn):
    fusedconv=nn.Conv2d(conv.in_channels,conv.out_channels,kernel_size=conv.kernel_size,
                        stride=conv.stride,padding=conv.padding,groups=conv.groups,bias=True).requires_grad_(False).to(conv.weight.device)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv
def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)
def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

def intersect_dicts(da,db,exclude=()):
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}
def is_parallel(mode):
    return type(mode) in (nn.parallel.DataParallel,nn.parallel.DistributedDataParallel)
class ModelEMA:
    def __init__(self,model,decay=0.9999,updates=0):  ##deca移动衰减率，updates更新次数
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.updates=updates
        self.decay=lambda  x:decay*(1-math.exp(-x/2000))##这是一个自适应衰减函数，随着 updates 增加，衰减因子会逐渐接近 decay
        for p in  self.ema.parameters():##冻结 ：防止反向传播过程中修改 EMA 模型的参数
            p.requires_grad_(False)
    def update(self,model):#更新 EMA 模型的 参数（parameters
        '''使用当前模型参数更新 EMA 模型'''
        with torch.no_grad():
            self.updates+=1###更新计数器 updates，每次更新 +1
            d=self.decay(self.updates)#计算当前衰减因子 d
            msd=model.module.state_dict() if is_parallel(model) else model.state_dict()##获取当前模型的参数字典
            for k,v ,in self.ema.state_dict().items():#遍历 EMA 模型的所有参数和缓冲区
                if v.dtype.is_floating_point:##只处理浮点类型的参数（忽略整型、计数器等）
                    v *= d#将 EMA 参数按衰减因子缩小
                    v += (1. - d) * msd[k].detach()#加入当前模型参数的加权更新
    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        '''同步主模型的属性到 EMA 模型'''
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)###这个函数的作用是从 model 中复制属性到 self.ema 中，支持白名单/黑名单过滤。
@contextmanager
def torch_distributed_zero_first(local_rank:int):
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()###同步操作，
    yield
    if local_rank == 0:
        torch.distributed.barrier()

class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
        # is this method that is overwritten by the sub-class
        # This original goal of this method was for tensor sanity checks
        # If you're ok bypassing those sanity checks (eg. if you trust your inference
        # to provide the right dimensional inputs), then you can just use this method
        # for easy conversion from SyncBatchNorm
        # (unfortunately, SyncBatchNorm does not store the original class - if it did
        #  we could return the one that was originally created)
        return
def revert_sync_batchnorm(module):
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        new_cls = BatchNormXd
        module_output = BatchNormXd(module.num_features,
                                               module.eps, module.momentum,
                                               module.affine,
                                               module.track_running_stats)
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output

class TracedModel(nn.Module):

    def __init__(self, model=None, device=None, img_size=(640, 640)):
        super(TracedModel, self).__init__()

        print(" Convert model to Traced-model... ")
        self.stride = model.stride
        self.names = model.names
        self.model = model

        self.model = revert_sync_batchnorm(self.model)
        self.model.to('cpu')
        self.model.eval()

        self.detect_layer = self.model.model[-1]
        self.model.traced = True

        rand_example = torch.rand(1, 3, img_size, img_size)

        traced_script_module = torch.jit.trace(self.model, rand_example, strict=False)
        # traced_script_module = torch.jit.script(self.model)
        traced_script_module.save("traced_model.pt")
        print(" traced_script_module saved! ")
        self.model = traced_script_module
        self.model.to(device)
        self.detect_layer.to(device)
        print(" model is traced! \n")

    def forward(self, x, augment=False, profile=False):
        out = self.model(x)
        out = self.detect_layer(out)
        return out
