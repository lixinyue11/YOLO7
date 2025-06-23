import torch
from torch import nn
from torchvision.ops import box_iou

from 物体检测.YOLO_LEARN.utils.general import xywh2xyxy, bbox_iou
import torch.nn.functional   as F

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps
class FocalLoss(nn.Module):
    '''

    是一种常用于解决类别不平衡问题的损失函数，尤其在目标检测任务中效果显著。
    它通过引入两个调节因子 alpha 和 gamma，使得模型更关注 难分类样本，同时减少对易分类负样本的过度惩罚。
    解决正负样本不均衡的问题（如目标检测中背景远多于前景）。
    让模型更加关注 困难样本（hard examples），提升整体性能。
    在 YOLO 等检测器中用于分类损失或目标置信度损失。
    \text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
    $ p_t $：预测概率（经过 sigmoid 后的结果）。
    $ \alpha_t $：类别权重，平衡正负样本比例。
    $ \gamma $：聚焦参数，控制难易样本的关注程度
    '''
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()基础损失函数
        self.gamma = gamma#聚焦参数，控制难易样本的区分
        self.alpha = alpha# 类别权重，用于平衡正负样本。
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # 将基础损失函数的 reduction 设置为 'none'，以便对每个样本单独应用 Focal Loss

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)




class ComputeLossOTA:
    ''''

    输入: p (preds), targets, imgs
│
├── build_targets: 找到与预测匹配的真实框及其位置
│
├── 遍历每一层 YOLO 输出
│   ├── 提取对应的预测子集
│   ├── 计算 bounding box loss (CIoU)
│   ├── 计算 objectness loss (BCE or FocalLoss)
│   └── 计算 classification loss (BCE or FocalLoss)
│
├── 加权合并所有层的损失
├── 自动调整各层权重 (optional)
└── 返回总损失和分项损失
    '''
    def __init__(self,model,autobalance=False):#autobalance：是否启用自动平衡不同层的损失权重。
        super(ComputeLossOTA, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        '''
        使用 BCEWithLogitsLoss（二元交叉熵损失 + logits）分别定义分类损失 (BCEcls) 和目标置信度损失 (BCEobj)
        pos_weight: 正样本的权重，防止类别不平衡
        '''
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        '''对分类标签进行平滑处理，提升泛化能力。
        cp, cn: 分别是正负样本的目标值（默认为 1.0 和 0.0，经过平滑后会略有调整）。'''
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
        # 可选 Focal Loss
        #如果启用了 Focal Loss（通过 fl_gamma > 0），则将分类和目标置信度损失包装成 Focal Loss，缓解正负样本不平衡问题。
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # 获取模型的最后一层检测头（Detect 模块）。
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # 不同输出层的损失权重系数（用于多尺度检测）
        self.ssi = list(det.stride).index(16) if autobalance else 0  # 自动平衡时参考的步长索引（默认为 stride=16）。
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        '''
        检测模块中的以下属性复制到当前类中：
        na: anchor 数量。
        nc: 类别数。
        nl: 输出层数。
        anchors: anchor 框尺寸。
        stride: 各层下采样步长。
        '''
        for k in 'na', 'nc', 'nl', 'anchors', 'stride':
            setattr(self, k, getattr(det, k))
    def __call__(self, p, targets, imgs):  # predictions(（batch,anchor个数，输出大小，坐标点4+置信度1+分类数量）), targets（多少标签，几个值）, model

        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
       #初始化分类损失 lcls、边界框损失 lbox、目标置信度损失 lobj。
        bs, as_, gjs, gis, targets, anchors = self.build_targets(p, targets, imgs)
        #将预测张量的 shape 转换为 [w, h, w, h] 格式，用于坐标归一化。
        '''
        bs: 图像索引。
        as_: 锚框索引。
        gjs, gis: 网格坐标。
        targets: 对应的目标。
        anchors: 锚框大小。
        '''
        pre_gen_gains = [torch.tensor(pp.shape, device=device)[[3, 2, 3, 2]] for pp in p]

        # Losses
        for i, pi in enumerate(p):  # 遍历每个 YOLO 输出层（通常有多个尺度）。
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # 提取该层对应的图像索引、锚框索引、网格坐标。
            tobj = torch.zeros_like(pi[..., 0], device=device)  # 创建目标置信度张量 tobj，初始化为 0。

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  #预测结果中提取对应于当前目标的预测子集。

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = ps[:, :2].sigmoid() * 2. - 0.5#预测中心点坐标（经过 sigmoid 归一化后调整）。
                # pxy = ps[:, :2].sigmoid() * 3. - 1.
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # 组合得到预测框 [x, y, w, h]。
                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]# 真实框坐标乘以图像尺寸（取消归一化）并减去网格偏移。还原到特征图
                selected_tbox[:, :2] -= grid
                iou = bbox_iou(pbox.T, selected_tbox, x1y1x2y2=False, CIoU=True)  #并累加 (1 - IoU) 到 lbox。中心点距离，宽高，CIOU重叠面积都加入了
                lbox += (1.0 - iou).mean()  # iou loss

                # 目标置信度损失（Objectness Loss）,使用 IoU 值作为目标置信度标签（soft label）。,self.gr 是一个可学习参数，默认为 1.0。
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # 分类损失（Classification Loss）
                selected_tcls = targets[i][:, 1].long()#前目标的类别。如果多类，则使用 BCEWithLogitsLoss（或 FocalLoss）进行分类损失计算。
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), selected_tcls] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
            '''用 BCEWithLogitsLoss（或 FocalLoss）计算目标置信度损失。
不同层的损失加权不同（由 self.balance 控制）。'''
            obji = self.BCEobj(pi[..., 4], tobj)###大部分是背景
            lobj += obji * self.balance[i]  # obj loss
            ##自动平衡权重（Autobalance）
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
    def build_targets(self, p, targets, imgs):
        '''

        在目标检测训练过程中，我们需要将模型输出的大量预测框与真实的标签框进行匹配。传统方法如基于 IoU 阈值的静态匹配方式可能不够高效或准确，而 YOLOv7 中采用的是 基于最优传输理论（Optimal Transport）的 SimOTA 动态匹配策略。
        :param p:模型输出的所有预测框
        :param targets:真实标签框
        :param imgs:输入图像列表，
        :return:
        '''

        # indices, anch = self.find_positive(p, targets)
        indices, anch = self.find_3_positive(p, targets)####获取初始的 anchor 匹配位置（grid 坐标、anchor 编号等）
        # indices, anch = self.find_4_positive(p, targets)
        # indices, anch = self.find_5_positive(p, targets)
        # indices, anch = self.find_9_positive(p, targets)
        device = torch.device(targets.device)
        '''
        为每一层创建空列表，用于保存：
        图像编号 (matching_bs)
        anchor 编号 (matching_as)
        网格 y 坐标 (matching_gjs)
        网格 x 坐标 (matching_gis)
        真实目标 (matching_targets)
        anchor 尺寸 (matching_anchs)
        '''
        matching_bs = [[] for pp in p]
        matching_as = [[] for pp in p]
        matching_gjs = [[] for pp in p]
        matching_gis = [[] for pp in p]
        matching_targets = [[] for pp in p]
        matching_anchs = [[] for pp in p]

        nl = len(p)

        for batch_idx in range(p[0].shape[0]):
        #提取当前 batch 的目标框。
            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue
            ##将真实框从归一化坐标转为像素坐标
            '''
            txywh: 归一化的 [x, y, w, h] → 转换为像素坐标。
txyxy: 转换为 [x1, y1, x2, y2] 格式用于后续 IoU 计算。
            '''
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh2xyxy(txywh)
            #提取对应层的有效预测框并转换为实际坐标
            pxyxys = []#预测框坐标（格式为 [x1, y1, x2, y2]）
            p_cls = []#分类得分（class scores）
            p_obj = []#目标置信度得分（objectness score）
            from_which_layer = []#预测框来自哪一层输出层（YOLO head）
            all_b = []#图像索引（batch index）
            all_a = []#anchor 编号（index of anchor）
            all_gj = []#网格行坐标（grid y-index）
            all_gi = []#：网格列坐标（grid x-index）
            all_anch = []#anchor 尺寸（width, height）

            for i, pi in enumerate(p):
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append((torch.ones(size=(len(b),)) * i).to(device))

                fg_pred = pi[b, a, gj, gi]    # 获取有效预测框
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2. - 0.5 + grid) * self.stride[i]  # / 8.# 预测的中心点坐标
                # pxy = (fg_pred[:, :2].sigmoid() * 3. - 1. + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * anch[i][idx] * self.stride[i]  # / 8. # 预测的宽高
                pxywh = torch.cat([pxy, pwh], dim=-1)#
                pxyxy = xywh2xyxy(pxywh)# 转换为 [x1,y1,x2,y2] 格式
                pxyxys.append(pxyxy)

            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue
            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_anch = torch.cat(all_anch, dim=0)
        # 计算预测框与真实框的 IoU 和损失
            '''box_iou: 计算所有真实框与预测框之间的 IoU。
    pair_wise_iou_loss: 转换为损失形式（IoU 越小损失越大）'''
            pair_wise_iou = box_iou(txyxy, pxyxys)###真实数值和所有的候选框

            pair_wise_iou_loss = -torch.log(pair_wise_iou + 1e-8)##计算损失
            #选择 top-k 最优预测框用于匹配,dynamic_ks: 动态决定每个真实框需要多少个预测框参与匹配。
            top_k, _ =   torch.topk(pair_wise_iou, min(10, pair_wise_iou.shape[1]), dim=1)##多的选择10哥，少的拿一个
            dynamic_ks = torch.clamp(top_k.sum(1).int(), min=1)##累加，太小的不需要
            ##构造类别损失矩阵
            '''
            构造 one-hot 类别标签。
将预测的类别概率和目标置信度相乘，作为联合分类得分。
计算每个预测框与每个真实框之间的分类损失。
            '''
            gt_cls_per_image = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.nc)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )##重复候选框数量次数

            num_gt = this_target.shape[0]

            cls_preds_ = (
                    p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )##预测情况类别

            y = cls_preds_.sqrt_()

            pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)), gt_cls_per_image, reduction="none"
            ).sum(-1)##类别差异
            del cls_preds_

        # 构建总代价矩阵并进行匹配
            '''
            cost: 总代价 = 分类损失 + 权重 × IoU 损失。
    构建匹配矩阵，表示每个真实框应由哪些预测框负责预测。
            '''
            cost = (
                    pair_wise_cls_loss
                    + 3.0 * pair_wise_iou_loss
            )##类别损失和IOU损失累加

            matching_matrix = torch.zeros_like(cost, device=device)##全0矩阵，没有匹配上的是0

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(
                    cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
                )
                matching_matrix[gt_idx][pos_idx] = 1.0##实际匹配上的

            del top_k, dynamic_ks

        #处理多个预测框匹配同一个目标的情况
            '''
            处理多个预测框匹配同一个目标的情况,如果某个预测框被多个目标选中，则只保留代价最小的那个。
            '''

            anchor_matching_gt = matching_matrix.sum(0)##竖着
            if (anchor_matching_gt > 1).sum() > 0:##一个样本匹配多个GT
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)##只拿损失最小的
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0##删除其他的
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0##保留最下的
            '''
                提取出所有有匹配的预测框。
                每个预测框对应的真实框索引。
            '''
            fg_mask_inboxes = (matching_matrix.sum(0) > 0.0).to(device)##哪些正样本
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]##batch索引
            all_a = all_a[fg_mask_inboxes]##anchor索引
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]##匹配到正样本的GT
            #将匹配结果按层划分，便于后续损失函数计算。
            for i in range(nl):#合并
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])
        #合并匹配结果
        for i in range(nl):
            if matching_targets[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_targets[i] = torch.cat(matching_targets[i], dim=0)
                matching_anchs[i] = torch.cat(matching_anchs[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_as[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_targets[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)
                matching_anchs[i] = torch.tensor([], device='cuda:0', dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs

    def find_3_positive(self, p, targets):
        '''

        :param p: 模型输出的所有预测张量（list of tensors），每一层对应一个 YOLO head 输出。
        :param targets: 真实目标框 [image_index, class_id, x_center, y_center, width, height]
        :return:
        '''
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # anchor 数量（通常为 3，YOLO 使用多尺度 anchor）;当前 batch 中的目标框数量
        indices, anch = [], []#存储 (b, a, gj, gi)，即图像编号、anchor 编号、网格 y 坐标、网格 x 坐标;存储对应的 anchor 宽高
        gain = torch.ones(7, device=targets.device).long()  # 是归一化坐标转实际网格坐标的缩放因子。(长度为 7 是为了适配目标框格式 [img_idx, cls, x, y, w, h, anchor_idx])
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  #创建 anchor 索引，shape (na, nt),表示每个目标框都会被复制 na 次（每个 anchor 分配一次）
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # 将原始目标框复制 na 次（每个 anchor 对应一个目标框）,并将 anchor 索引附加到最后一维，形成新的 shape: (na, nt, 7),主要用来确定属于那个候选框

        ###targets：batch,分类，xyzh，候选框索引
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  #  偏移量矩阵，用于在当前网格基础上扩展周围的网格。

        for i in range(self.nl):#遍历每一层输出（YOLO 多尺度检测）
            anchors = self.anchors[i]# 当前层使用的 anchor 尺寸（如 [10,13], [16,30], [33,23]）
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  #用于将归一化的 [x, y, w, h] 转换为当前层的特征图坐标

            # Match targets to anchors
            t = targets * gain#将目标框坐标乘以 gain，转换为当前层的特征图空间坐标
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # 目标框宽高与 anchor 宽高的比例
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  #  判断是否满足宽高比阈值（默认 anchor_t=4）
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # 目标框中心点在当前层的网格坐标,到左上角的位置
                gxi = gain[[2, 3]] - gxy  # i中心点距离右侧/下边界的距离（用于反向偏移），到右下角
                '''
                判断目标框中心点是否接近当前网格点（通过模运算判断是否靠近网格边缘）
j, k, l, m 是布尔掩码，表示是否使用额外的偏移来扩大匹配区域，1%用于确定到坐标的格子
                '''
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T##
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                '''
                j: 构建索引，选择主网格点及其周围四个方向的网格点
t: 根据索引选择需要处理的目标框
offsets: 添加偏移量，使目标框可以匹配多个网格点
                '''
                j = torch.stack((torch.ones_like(j), j, k, l, m))##自身的位置是True
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # 提取图像编号、类别编号、网格坐标
            b, c = t[:, :2].long().T  # 图像编号（batch index）, 类别编号（class id）
            gxy = t[:, 2:4]  # 目标框中心点坐标（归一化后）
            gwh = t[:, 4:6]  # 加上偏移后的整数网格坐标
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # 提取 x、y 坐标作为网格索引

            # Append
            a = t[:, 6].long()  # 提取 anchor 编号并限制网格范围
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  #将图像编号、anchor 编号、网格 y、x 坐标加入 indicesclamp_: 保证网格坐标不越界
            anch.append(anchors[a])  # anchors

        return indices, anch


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        # self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.1, .05])  # P3-P7
        # self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.5, 0.4, .1])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    # t[t==self.cp] = iou.detach().clamp(0).type(t.dtype)
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        '''

        是 YOLO 模型中用于计算训练过程中损失函数的类。
        它的主要作用是将模型预测结果与真实标签进行对比，
        计算边界框（bounding box）损失、目标置信度（objectness）
        损失和分类（classification）损失，并最终合并为总损失用于反向传播优化。
        :param model:
        :param autobalance: 否自动调整各层损失权重
        '''
        super(ComputeLoss,self).__init__()
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # 使用 BCEWithLogitsLoss（二元交叉熵 + logits）
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # C分类标签进行平滑处理，提升泛化能力。 https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # 如果启用 Focal Loss（通过 fl_gamma > 0），则将分类和目标置信度损失包装成 Focal Loss，缓解正负样本不平衡问题。
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7  不同输出层的损失权重系数（用于多尺度检测）。
        # self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.1, .05])  # P3-P7
        # self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.5, 0.4, .1])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index自动平衡时参考的步长索引（默认为 stride=16）。
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance #将损失函数、梯度因子、超参数、是否启用自动平衡保存到类属性中
        for k in 'na', 'nc', 'nl', 'anchors':
            '''
            将检测模块中的以下属性复制到当前类中：
na: anchor 数量。
nc: 类别数。
nl: 输出层数。
anchors: anchor 框尺寸。
            '''
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device
        #初始化分类损失 lcls、边界框损失 lbox、目标置信度损失 lobj。
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        '''
        tcls: 目标类别。
tbox: 目标框坐标。
indices: 图像编号、anchor 编号、网格坐标。
anchors: 锚框大小。
        '''
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # 提取该层对应的图像索引、锚框索引、网格坐标。
            tobj = torch.zeros_like(pi[..., 0], device=device)  # t创建目标置信度张量 tobj，初始化为 0。

            n = b.shape[0]  # 获取该层的目标数量。
            if n:
                ps = pi[b, a, gj, gi]  # 从预测结果中提取对应于当前目标的预测子集。

                # Regression
                '''
                计算预测框中心点坐标和宽高。
使用 sigmoid 归一化并调整偏移量。
最终组合成预测框 [x, y, w, h]。
                '''
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # 使用 IoU 值作为目标置信度标签（soft label）。

                # 如果多类，则使用 BCEWithLogitsLoss 进行分类损失计算。
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    # t[t==self.cp] = iou.detach().clamp(0).type(t.dtype)
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  #计算目标置信度损失，并乘以对应层的权重。
            if self.autobalance:#自动调整各层损失权重。
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:#标准化损失权重。
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']#应用超参数对损失加权。
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        #构建目标框匹配信息。

        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # na: anchor 数量；nt: 当前 batch 中的目标框数量
        #将原始目标框复制 na 次（每个 anchor 对应一个目标框），并将 anchor 索引附加到最后一维。
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # 定义偏移矩阵，用于在当前网格基础上扩展周围的网格
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
        #遍历每一层输出，获取当前层使用的 anchor，并将目标框坐标转换为当前层特征图空间坐标。
        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # 判断目标框宽高与 anchor 的比例是否满足阈值，过滤不匹配的目标框。
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # 判断目标框中心点是否接近当前网格点，并添加偏移量，使目标框可以匹配多个网格点。
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # 提取图像编号、类别编号、网格坐标。
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

