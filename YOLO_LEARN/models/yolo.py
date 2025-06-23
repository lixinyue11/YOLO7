import argparse
import logging
import math
from copy import deepcopy
from pathlib import Path

import thop
from 物体检测.YOLO_LEARN.models.common import *

import torch
import torch.nn as nn

from 物体检测.YOLO_LEARN.utils.general import set_logging, check_file, make_divisible
from 物体检测.YOLO_LEARN.utils.torch_utils.utils import select_device, time_synchronized, scale_img, check_anchor_order, \
    initialize_weights, model_info, fuse_conv_and_bn, copy_attr

logger = logging.getLogger(__name__)
class IDetect(nn.Module):
    stride=None
    export=False
    end2end=False
    include_nms=False
    concat=False
    def __init__(self,nc=80,anchors=(),ch=()):##ch 是RepConv获得的输出维度
        super(IDetect,self).__init__()
        self.nc=nc
        self.no=nc+5###每个锚点的输出维度，包括边界框的4个坐标值和类别概率
        self.nl=len(anchors)
        self.na=len(anchors[0])//2
        self.grid=[torch.zeros(1)]*self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)锚框注册到缓冲区
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)锚框的网格化处理

        self.m=nn.ModuleList(nn.Conv2d(x,self.no*self.na,1) for x in ch)##通常是一组检测头，用于将特征图映射到最终的检测输出（如边界框坐标、类别概率等）。每一组RepConv的特征维度*类别*候选框作为边界款的坐标和最终输出
        self.ia=nn.ModuleList(ImplicitA(x) for x in ch)  ##通过可学习的偏移量来调整网络的中间特征。通常用于在网络中引入隐式的可学习参数，这些参数可以动态地调整特征图的分布。
        self.im=nn.ModuleList(ImplicitM(self.no*self.na) for x in ch)
        '''
        self.m  Module/Model 的缩写，表示一组子模块 用于定义检测头（Detection Heads）、特征变换层等
        self.ia Implicit Addition 的缩写，表示隐式加法 用于在网络中引入可学习的偏移量，增强模型表现力
        self.im  ✅ 功能： ImplicitM 实现了一个可学习的乘法操作：对输入张量 x 每个通道施加一个可学习的缩放因子 self.implicit。
            self.implicit 是一个形状为 (1, channel, 1, 1) 的参数，表示每个通道的缩放系数。
            ⚙️ 初始化：
            使用正态分布初始化 self.implicit，均值为 mean=1.，标准差为 std=0.02。
            这意味着初始时，self.im 不会对输入产生明显影响（因为乘以 1 相当于无变化），但它是可学习的，在训练过程中会动态调整。
        '''
    def forward(self,x):
        z=[]
        self.training |=self.export
        for i in range(self.nl):
            x[i]=self.m[i](self.ia[i](x[i]))
            x[i]=self.im[i](x[i])
            bs,_,ny,nx=x[i].shape
            x[i]=x[i].view(bs,self.na,self.no,ny,nx).permute(0,1,3,4,2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4]!=x[i].shape[2:4]:
                    self.grid[i]=self._make_grid(nx,ny).to(x[i].device)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    def fuseforward(self, x):
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy,wh,conf=y.split((2,2,self.nc+1),4)
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))
        if self.training:
            out = x
        elif self.end2end:  ###如果启用了端到端推理（例如部署时 ONNX 推理）。
            out = torch.cat(z, 1)
        elif self.include_nms:  #如果需要包含 NMS（非极大值抑制）逻辑。
            z = self.convert(z)
            out = (z, )
        elif self.concat:  #如果启用了输出拼接模式。
            out = torch.cat(z, 1)
        else:  #适用于普通推理流程。
            out = (torch.cat(z, 1), x)
        return out

    def fuse(self):
        print("IDetect.fuse")
        # fuse ImplicitA and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.m[i].weight.shape
            c1_, c2_, _, _ = self.ia[i].implicit.shape
            self.m[i].bias += torch.matmul(self.m[i].weight.reshape(c1, c2),
                                           self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

        # fuse ImplicitM and Convolution
        for i in range(len(self.m)):
            c1, c2, _, _ = self.im[i].implicit.shape
            self.m[i].bias *= self.im[i].implicit.reshape(c2)
            self.m[i].weight *= self.im[i].implicit.transpose(0, 1)
    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix
        return (box, score)

    @staticmethod
    def _make_grid(nx=20,ny=20):
        yv,xv=torch.meshgrid([torch.arange(ny),torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
class Detect(nn.Module):
    strode=None
    export=None
    end2end=None
    include_nms = False
    concat = False
    def __init__(self, nc=80, anchors=(), ch=()):
        super(Detect,self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    xy, wh, conf = y.split((2, 2, self.nc + 1), 4)  # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        if self.training:
            out = x
        elif self.end2end:
            out = torch.cat(z, 1)
        elif self.include_nms:
            z = self.convert(z)
            out = (z, )
        elif self.concat:
            out = torch.cat(z, 1)
        else:
            out = (torch.cat(z, 1), x)

        return out

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def convert(self, z):
        z = torch.cat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=z.device)
        box @= convert_matrix
        return (box, score)

def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)

class IKeypoint(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), nkpt=17, ch=(), inplace=True, dw_conv_kpt=False):  # detection layer
        super(IKeypoint, self).__init__()
        self.nc = nc  # number of classes
        self.nkpt = nkpt
        self.dw_conv_kpt = dw_conv_kpt
        self.no_det = (nc + 5)  # number of outputs per anchor for box and class
        self.no_kpt = 3 * self.nkpt  ## number of outputs per anchor for keypoints
        self.no = self.no_det + self.no_kpt
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.flip_test = False
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no_det * self.na, 1) for x in ch)  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im = nn.ModuleList(ImplicitM(self.no_det * self.na) for _ in ch)

        if self.nkpt is not None:
            if self.dw_conv_kpt:  # keypoint head is slightly more complex
                self.m_kpt = nn.ModuleList(
                    nn.Sequential(DWConv(x, x, k=3), Conv(x, x),
                                  DWConv(x, x, k=3), Conv(x, x),
                                  DWConv(x, x, k=3), Conv(x, x),
                                  DWConv(x, x, k=3), Conv(x, x),
                                  DWConv(x, x, k=3), Conv(x, x),
                                  DWConv(x, x, k=3), nn.Conv2d(x, self.no_kpt * self.na, 1)) for x in ch)
            else:  # keypoint head is a single convolution
                self.m_kpt = nn.ModuleList(nn.Conv2d(x, self.no_kpt * self.na, 1) for x in ch)

        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            if self.nkpt is None or self.nkpt == 0:
                x[i] = self.im[i](self.m[i](self.ia[i](x[i])))  # conv
            else:
                x[i] = torch.cat((self.im[i](self.m[i](self.ia[i](x[i]))), self.m_kpt[i](x[i])), axis=1)

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            x_det = x[i][..., :6]
            x_kpt = x[i][..., 6:]

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                kpt_grid_x = self.grid[i][..., 0:1]
                kpt_grid_y = self.grid[i][..., 1:2]

                if self.nkpt == 0:
                    y = x[i].sigmoid()
                else:
                    y = x_det.sigmoid()

                if self.inplace:
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    if self.nkpt != 0:
                        x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1, 1, 1, 1, 17)) * \
                                           self.stride[i]  # xy
                        x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1, 1, 1, 1, 17)) * \
                                           self.stride[i]  # xy
                        # x_kpt[..., 0::3] = (x_kpt[..., ::3] + kpt_grid_x.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                        # x_kpt[..., 1::3] = (x_kpt[..., 1::3] + kpt_grid_y.repeat(1,1,1,1,17)) * self.stride[i]  # xy
                        # print('=============')
                        # print(self.anchor_grid[i].shape)
                        # print(self.anchor_grid[i][...,0].unsqueeze(4).shape)
                        # print(x_kpt[..., 0::3].shape)
                        # x_kpt[..., 0::3] = ((x_kpt[..., 0::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,17) * self.stride[i]  # xy
                        # x_kpt[..., 1::3] = ((x_kpt[..., 1::3].tanh() * 2.) ** 3 * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,17) * self.stride[i]  # xy
                        # x_kpt[..., 0::3] = (((x_kpt[..., 0::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,0].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_x.repeat(1,1,1,1,17) * self.stride[i]  # xy
                        # x_kpt[..., 1::3] = (((x_kpt[..., 1::3].sigmoid() * 4.) ** 2 - 8.) * self.anchor_grid[i][...,1].unsqueeze(4).repeat(1,1,1,1,self.nkpt)) + kpt_grid_y.repeat(1,1,1,1,17) * self.stride[i]  # xy
                        x_kpt[..., 2::3] = x_kpt[..., 2::3].sigmoid()

                    y = torch.cat((xy, wh, y[..., 4:], x_kpt), dim=-1)

                else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                    if self.nkpt != 0:
                        y[..., 6:] = (y[..., 6:] * 2. - 0.5 + self.grid[i].repeat((1, 1, 1, 1, self.nkpt))) * \
                                     self.stride[i]  # xy
                    y = torch.cat((xy, wh, y[..., 4:]), -1)

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Model(nn.Module):
    def __init__(self, cfg='yolor-csp-c.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        self.traced = False
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name

            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:###修改原来的输出标签列别
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model,self.save=parse_model(deepcopy((self.yaml)),ch=[ch])##单纯之构建模型的大框架，构建模型框架
        self.names=[str(i) for i in range(self.yaml['nc'])]
        m=self.model[-1]
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())
        if isinstance(m,IDetect):##对检测头进行初始化,计算并设置步长；,调整锚框尺寸；初始化偏置；确保检测层能正确地进行目标检测任务
            s=256 # 计算后续的步长
            '''
            使用一个全零张量 torch.zeros(1, ch, s, s) 输入到模型前向传播，得到各输出特征图的尺寸。
            然后通过 s / x.shape[-2] 计算每个特征图相对于输入图像的下采样倍数（即步长）。
            最终将这些步长值赋给 m.stride，表示该检测层在不同尺度下的步长。
            '''
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward（8，16，32）
            '''
            确保 anchors 的顺序与 stride 一致，避免因顺序错误导致检测结果异常。
            '''
            check_anchor_order(m)
            '''将锚框的宽高值除以对应的步长，将其从特征图坐标转换为原始图像坐标。'''
            m.anchors /= m.stride.view(-1, 1, 1)
            '''把检测层的步长信息保存到整个模型中，供其他部分使用。'''
            self.stride = m.stride
            '''调用 _initialize_biases() 方法，对检测层中的偏置项进行初始化，提升训练初期的稳定性。
                注释说明此操作仅执行一次'''
            self._initialize_biases()  # only run once
        if isinstance(m, IKeypoint):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases_kpt()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)#对卷积层进行权重初始化，仅对特定模块生效，对 BatchNorm 层进行初始化
        self.info()
        logger.info('')
    def _print_biases(self):#函数用于打印 YOLO 模型中检测层（Detect）的偏置值，这些偏置值通常与目标检测任务中的先验框（anchor boxes）相关
        m=self.model[-1]
        for mi in m.m:#m.m: 是一个包含多个卷积层的模块列表。
            b=mi.bias.detach().view(m.na,-1).T
            '''置值被重塑为 (na, -1) 的形状，并进行转置以方便处理'''
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))
    def fuse(self):
        print('Fusing layers------')
        for m in self.model.modules():
            if isinstance(m,RepConv):
                m.fuse_repvgg_block()
            elif type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 函数会将卷积（Conv2d）和批量归一化（BatchNorm2d）合并成一个等效的卷积层。
                delattr(m, 'bn')  # 删除原始的 bn 层，并替换 forward 方法为 fuseforward，以便使用融合后的结构进行前向传播
                m.forward = m.fuseforward  #并替换 forward 方法为 fuseforward，以便使用融合后的结构进行前向传播。
            elif isinstance(m, (IDetect)):
                m.fuse()
                m.forward = m.fuseforward
        self.info()
        return self

    def nms(self, mode=True):  # a添加或移除 NMS（非极大值抑制）模块。
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            print('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            print('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self
    def autoshape(self):  # 封装模型为标准输入输出格式，使其像 OpenCV 的 DNN 模块一样使用。自动适配图像尺寸、通道顺序等。
        '''
        是一种封装模型的方法，使得模型在推理时能够自动处理输入图像的尺寸问题。YOLO 模型通常要求输入图像具有固定的尺寸（例如 640x640），但实际应用中，输入图像的大小可能是任意的。AutoShape 的作用包括：
        自动将输入图像缩放到模型期望的尺寸。
        在缩放的同时保持宽高比，避免图像变形。
        在必要时进行零填充（zero-padding）以确保图像比例正确。
        处理多种输入格式（如 NumPy 数组、PIL 图像等）。
        :return:
        '''
        print('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m
    def info(self, verbose=False, img_size=640):# 打印模型结构信息，包括层数、参数量、FLOPS、输入输出形状
        model_info(self, verbose, img_size)

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from##遍历每个检测头及其步长
            b = mi.bias.view(m.na, -1)  # 将偏置 reshape 为 (na, 85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # 初始化目标置信度偏置，基于目标密度
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # 初始化类别偏置，使用默认或类别频率

            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)#将新偏置写回卷积层




    def forward(self,x,profile=None, augment=False):
        if augment:###如果启用数据增强
            img_size = x.shape[-2:]  # height, width，
            s = [1, 0.83, 0.67]  # scales，缩放比例
            f = [None, 3, None]  # flips (2-ud, 3-lr)翻转方式，
            y = []  # outputs
            for si, fi in zip(s, f):
                '''#对每种尺度进行前向传播并保存结果'''
                xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
                '''缩放回原始尺寸并处理翻转后的坐标'''
                yi = self.forward_once(xi)[0]  # forward#
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if not hasattr(self, 'traced'):
                self.traced = False

            if self.traced:
                if isinstance(m, Detect) or isinstance(m, IDetect) or isinstance(m, IKeypoint) :
                    break

            if profile:
                c = isinstance(m, (Detect, IDetect))
                o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
                for _ in range(10):
                    m(x.copy() if c else x)
                t = time_synchronized()
                for _ in range(10):
                    m(x.copy() if c else x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run

            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x


def parse_model(d,ch):
    '''

    构建 YOLO 模型的网络结构。它会读取配置中的模块信息，动态地创建模型层，并统计参数数量和连接关系。
    :param d:输出三层
    :param ch:三个候选框
    :return:
    '''
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # print(f, n, m, args)

        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j ,a in enumerate(args):
            try: args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:pass
        n=max(round(n*gd),1) if n>1 else n
        if m in [nn.Conv2d, Conv,SPPCSPC,RepConv]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)  ##特征/d['width_multiple']
            args=[c1,c2,*args[1:]]##初始3，输出通道数是输入通道数的整数倍。【第一个特征数】，组【特征，卷积，步长】
            # print(args,'------')
            if m in [SPPCSPC]:
                args.insert(2,n)
            n = 1
        elif m is Concat:     #输入通道数是多个来源层的通道数之和。
            c2=sum([ch[x] for x in f])
        elif m in[Detect,IDetect,IKeypoint]:  ####将输入通道数传给检测层；如果是整数，则构造默认 anchor 列表。
            args.append([ch[x] for x in f])
            if isinstance(args[1],int):args[1]=[list(range(args[1]*2))]*len(f)
        else:
            c2=ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module 调用模型
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # 统计模块参数数量。
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='D:/PycharmProject/yolov7-main/cfg//training/yolov7.yaml',
                        help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    if opt.profile:
        img = torch.rand(1, 3, 640, 640).to(device)
        # y = model(img, profile=True)
        model = Model('D:/PycharmProject/yolov7-main/cfg//training/yolov7.yaml', ch=3, nc=6, ).to(device)  # create
