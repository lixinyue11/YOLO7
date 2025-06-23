import torch

tensor1 = torch.tensor([[1, 2], [3, 4]])
tensor2 = torch.tensor([[5, 6], [7, 8]])
result = torch.cat([tensor1, tensor2], dim=1)
print(result,tensor1)

import torch
from torch import nn


class IDetect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):
        super(IDetect, self).__init__()
        self.nc = nc  # 类别
        self.no = nc + 5  # 输出的候选框的特征点 (xywh + obj_conf + class_probs)
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors per layer
        self.grid = [torch.zeros(1)] * self.nl  # grid coordinates for each detection layer
        anchors = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', anchors)  # shape(nl, na, 2)
        self.register_buffer('anchor_grid', anchors.clone().view(self.nl, 1, -1, 1, 1, 2))  # broadcasting shape

    def forward(self, x):
        z = []  # 存储最终的检测结果
        for i in range(self.nl):  # 遍历每个检测层

            bs, _, ny, nx = x[i].shape  # 获取当前特征图尺寸
            print( bs, _, ny, nx)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # reshape & permute
            print(self.grid[i].shape[2:4])
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)  # 生成网格坐标
                y = x[i].sigmoid()  # 对输出应用 sigmoid 函数
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # 计算 xy 坐标
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # 计算 wh 尺寸
                z.append(y.view(bs, -1, self.no))  # 扁平化输出
        return x if self.training else (torch.cat(z, 1), x)  # 返回训练输出或推理结果


# 初始化 IDetect 模块
detect_layer = IDetect(nc=80,
                       anchors=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]],
                       ch=[256, 512, 1024])

# 输入三个检测层的输出
inputs = [torch.randn(1, 255, 80, 80), torch.randn(1, 255, 40, 40), torch.randn(1, 255, 20, 20)]
# inputs = [torch.randn(1, 4, 3, 3), torch.randn(1, 4, 2, 2), torch.randn(1, 4, 1, 1)]

# 前向传播
outputs = detect_layer(inputs)
'''
第一个检测层输入特征图的通道数是 256
第二个检测层输入特征图的通道数是 512
第三个检测层输入特征图的通道数是 1024
✅ 总结：
参数
含义
ch
输入特征图的通道数（Channels），用于定义每个检测层的输入维度
nc
分类数量（例如 COCO 数据集是 80 类）
no
每个锚框的输出维度，通常是 nc + 5（5 表示 x, y, w, h, objectness）
nl
检测层的数量（通常有 3 个尺度：大中小目标）
na
每个检测层对应的锚框数量
anchors
锚框尺寸（预先聚类得到的先验框）
grid
用于计算坐标偏移的网格坐标矩阵

✅ 步骤分解：
特征图重塑：
输入形状 (batch_size, channels, height, width) 被重塑为 (batch_size, num_anchors, num_outputs, height, width)
再通过 permute 变换为 (batch_size, num_anchors, height, width, num_outputs)，便于后续处理
网格生成：
使用 _make_grid(nx, ny) 生成网格坐标，用于计算边界框中心点相对于网格的位置
坐标解码：
应用 sigmoid 激活函数对输出进行归一化
利用公式 xy = (sigmoid(xy) * 2 - 0.5) + grid 和 wh = (sigmoid(wh) * 2)^2 * anchor 解码边界框坐标和尺寸
输出拼接：
将所有检测层的结果拼接在一起，形成 (batch_size, total_anchors, num_outputs) 的最终输出格式

📌 作用总结
解码模型输出：将原始网络输出转换为目标检测所需的边界框坐标、对象置信度、类别概率。
适配 ONNX 推理：在导出 ONNX 模型时保留后处理逻辑，便于部署到 ONNX Runtime、TensorRT 等推理引擎中。
支持多尺度检测：处理多个检测头（不同感受野）的输出，提升小目标和大目标的检测效果。
'''

# 输出：(batch_size, total_anchors, 85)，其中 85 = 4(xywh) + 1(obj_confidence) + 80(class_probs)
