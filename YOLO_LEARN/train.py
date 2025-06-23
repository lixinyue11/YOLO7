import argparse
import datetime
import logging
import math
import os
import pickle
import random
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models.yolo import Model
from 物体检测.YOLO_LEARN import test
from 物体检测.YOLO_LEARN.experimental import attempt_load
from 物体检测.YOLO_LEARN.utils.autoanchor import check_anchors
from 物体检测.YOLO_LEARN.utils.datasets import create_dataloader
from 物体检测.YOLO_LEARN.utils.general import set_logging, check_file, increment_path, colorstr, init_seed, \
    check_dataset, one_cycle, check_img_size, labels_to_class_weights, labels_to_image_weights, strip_optimizer
from 物体检测.YOLO_LEARN.utils.loss import ComputeLossOTA, ComputeLoss
from 物体检测.YOLO_LEARN.utils.metrics import fitness
from 物体检测.YOLO_LEARN.utils.plots import plot_images, plot_results
from 物体检测.YOLO_LEARN.utils.torch_utils.utils import select_device, torch_distributed_zero_first, intersect_dicts, \
    ModelEMA, is_parallel
from 物体检测.YOLO_LEARN.utils.wandb_logging.wandb_utils import check_wandb_resume,WandbLogger
os.environ["WANDB_API_KEY"] = "4b41bf5fa66650642626b5b93501482551768eb0"
logger = logging.getLogger(__name__)
def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    print(opt.save_dir)
    save_dir,epochs,batch_size,total_batch_size,weights,rank,freeze=Path(opt.save_dir),opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank, opt.freeze

    # Directories
    wdir=save_dir/'weights'
    wdir.mkdir(parents=True,exist_ok=True)
    last=wdir/'last.pt'
    best=wdir/'best.pt'
    results_file=save_dir/'result.txt'
    # Save run settings
    with open(save_dir/'hyp.yaml','w' ) as f:yaml.dump(hyp,f,sort_keys=False)
    with open(save_dir/'opt.yaml','w') as f:yaml.dump(opt,f,sort_keys=False)
    # Configure
    plots=not opt.evolve
    cuda=device.type!='cpu'
    '''随机种子'''
    init_seed(2+rank)
    with   open(opt.data) as f:data_dict=yaml.load(f,Loader=yaml.SafeLoader)
    is_coco = opt.data.endswith('coco.yaml')


    # '''wandb'''
    loggers={'wandb':None}
    if rank in [-1,0]:
        opt.hyp=hyp
        run_id=None
        wandb_logger=WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb']=wandb_logger.wandb
        data_dict=wandb_logger.data_dict
        if wandb_logger.wandb:  weights,epochs,hyp=opt.weights,opt.epochs,opt.hyp
    '''获取标签'''
    nc=int(data_dict['nc'])##标签个数
    names =data_dict['names']  # 分类的种类

    print(data_dict)


    '''models'''
    pretrained = weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(weights, map_location=device,weights_only=False)##加载训练权重文件
        model = Model(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)##根据配置文件创建一个新的模型
        exclude=['anchor']  if (opt.cfg or hyp.get('anchors'))  and not opt.resume  else [] #如果指定了 opt.cfg 或设置了自定义的 anchors，并且不是继续训,就不加载锚框相关的权重。
        state_dict=ckpt['model'].float().state_dict()#获取预训练模型的权重，并将其转换为浮点格式
        '''
        将预训练模型的权重和当前新模型的权重进行匹配，只保留可以复用的部分。
        函数功能：
        只保留名称匹配且形状一致的层权重。
        排除掉在 exclude 中指定的层（如锚框相关权重）。
        参数说明：
        第一个参数是预训练权重字典。
        第二个参数是当前模型的权重字典。
        exclude：不需要加载权重的层名列表。
        结果：得到一个仅包含可迁移权重的 state_dict，用于后续加载。
        '''
        state_dict=intersect_dicts(state_dict,model.state_dict(), exclude=exclude)
        model.load_state_dict(state_dict, strict=False)  # state_dict：上一步筛选出的预训练权重。,strict=False：表示不要求完全匹配，忽略不匹配的层。
    else: model = Model(opt.cfg, ch=3, nc=6, anchors=hyp.get('anchors')).to(device)  # create
    with torch_distributed_zero_first(rank):#使用上下文管理器的方式调用这个类，保证在进入 with 块时：
        check_dataset(data_dict)  # check 检查并下载数据集，确保训练所需文件存在

    train_path = data_dict['train']
    test_path = data_dict['val']

    freeze = [f'model.{x}.' for x in
              (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    print('freeze--:', freeze)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer  累加的梯度
    nbs = 64  # batch
    accumulate = max(round(nbs / total_batch_size), 1)##几个batch进行优化
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs   ##为了保持正则化效果的一致性，当使用梯度累积时，需要按比例调整  .实际大小/批次大小
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")
    print(hyp)
    print(hyp['weight_decay'],total_batch_size*accumulate/nbs,'-----')

    '''无数据，后面要再看看'''
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups,设置三组权重，不同的衰减策略
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        if hasattr(v, 'im'):
            if hasattr(v.im, 'implicit'):
                pg0.append(v.im.implicit)
            else:
                for iv in v.im:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imc'):
            if hasattr(v.imc, 'implicit'):
                pg0.append(v.imc.implicit)
            else:
                for iv in v.imc:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imb'):
            if hasattr(v.imb, 'implicit'):
                pg0.append(v.imb.implicit)
            else:
                for iv in v.imb:
                    pg0.append(iv.implicit)
        if hasattr(v, 'imo'):
            if hasattr(v.imo, 'implicit'):
                pg0.append(v.imo.implicit)
            else:
                for iv in v.imo:
                    pg0.append(iv.implicit)
        if hasattr(v, 'ia'):
            if hasattr(v.ia, 'implicit'):
                pg0.append(v.ia.implicit)
            else:
                for iv in v.ia:
                    pg0.append(iv.implicit)
        if hasattr(v, 'attn'):
            if hasattr(v.attn, 'logit_scale'):
                pg0.append(v.attn.logit_scale)
            if hasattr(v.attn, 'q_bias'):
                pg0.append(v.attn.q_bias)
            if hasattr(v.attn, 'v_bias'):
                pg0.append(v.attn.v_bias)
            if hasattr(v.attn, 'relative_position_bias_table'):
                pg0.append(v.attn.relative_position_bias_table)
        if hasattr(v, 'rbr_dense'):
            if hasattr(v.rbr_dense, 'weight_rbr_origin'):
                pg0.append(v.rbr_dense.weight_rbr_origin)
            if hasattr(v.rbr_dense, 'weight_rbr_avg_conv'):
                pg0.append(v.rbr_dense.weight_rbr_avg_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_pfir_conv'):
                pg0.append(v.rbr_dense.weight_rbr_pfir_conv)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_idconv1'):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_idconv1)
            if hasattr(v.rbr_dense, 'weight_rbr_1x1_kxk_conv2'):
                pg0.append(v.rbr_dense.weight_rbr_1x1_kxk_conv2)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_dw'):
                pg0.append(v.rbr_dense.weight_rbr_gconv_dw)
            if hasattr(v.rbr_dense, 'weight_rbr_gconv_pw'):
                pg0.append(v.rbr_dense.weight_rbr_gconv_pw)
            if hasattr(v.rbr_dense, 'vector'):
                pg0.append(v.rbr_dense.vector)


    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    ###学习率衰减的方式
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:  ##这是默认的
        lf = one_cycle(1, hyp['lrf'],
                       epochs)  # cosine 1->hyp['lrf'] One Cycle的学习率变化过程是从lr0=0.01呈余弦变化衰退到lr0*lrf = 0.01*0.1 = 0.001
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    #EMA 指数移动平均，优化训练模型，进行平滑处理，提升繁华能力，，MA 会对模型参数进行加权平均，使得训练过程中参数更新更加稳定，减少噪声对模型的影响。
    #提升测试性能：使用 EMA 参数进行推理时，通常可以获得比未平滑参数更好的检测精度
    #增强鲁棒性：EMA 能够缓解训练过程中因学习率波动或数据分布变化带来的不稳定性。

    ema = ModelEMA(model) if rank in [-1, 0] else None
    # Resume  ##继续训练
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # 优化器
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Results
        if ckpt.get('training_results') is not None:
            results_file.write_text(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict
    ##图片大小

    gs = max(int(model.stride.max()), 32)  # 计算模型的最大下采样步长（grid size8,16,32），确保输入图像的大小是该步长的倍数。
    nl = model.model[-1].nl  # 获取检测层的数量  ##下采样位数
    '''
    关键变量：
        opt.img_size：用户指定的图像尺寸（例如，默认值为 [640, 640]）。
        check_img_size(x, gs)：一个函数，确保 x 是 gs 的倍数。如果不是，则向上取整到最接近 gs 的倍数。
        imgsz：训练时使用的图像尺寸。
        imgsz_test：测试时使用的图像尺寸。
    '''
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # 调整训练和测试时的图像尺寸，确保它们是 gs 的倍数。
    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # 训练数据的参数：gs:网格步长（stride，opt:所有训练参数,,hpy:超参数（如数据增强、loss 权重）,
    #augment=True启用训练增强,rect:保持图像原始比例,,rank:,分布式训练中的进程编号,world_size:总进程数,works:数据加载线程数,
    #image_weights:是否使用权重抽样,quad:加载 quad batch（4x 大小）,prefix:日志前缀
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))

    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class  ，类别的个数，所有图像的标签拼接成一个二维数组，并提取最大类别 ID。
    nb = len(dataloader)  # number of batches
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)#断言检查是否存在非法类别 ID。

    if rank in [-1, 0]:#创建验证数据加载器（仅主进程执行）
        testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        if not opt.resume:#非恢复训练时的初始化操作，如果不是继续训练，则执行以下初始化操作。
            labels = np.concatenate(dataset.labels, 0)#是所有图像的标签拼接结果。
            c = torch.tensor(labels[:, 0])  # classes#是所有标签的类别 ID，转换为 PyTorch 张量。
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                #plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)#统计各类别出现的频率。

            # Anchors
            if not opt.noautoanchor:###如果未禁用自动锚框检测，则调用 check_anchors 根据数据集重新计算锚框。
                ''''''
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision float32转化为float16
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    # nn.MultiheadAttention incompatibility with DDP https://github.com/pytorch/pytorch/issues/26698
                    find_unused_parameters=any(isinstance(layer, nn.MultiheadAttention) for layer in model.modules()))
    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers调整边界框损失权重 (box)，根据层数 (nl) 进行缩放。
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers分类损失权重 (cls) 根据类别数 (nc) 和层数 (nl) 进行调整。
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers对象损失权重 (obj) 根据图像大小 (imgsz) 和层数 (nl) 缩放。
    hyp['label_smoothing'] = opt.label_smoothing#将标签平滑值从命令行参数 (opt) 设置到超参数 (hyp) 中。
    model.nc = nc  # attach number of classes to model将类别数量 (nc) 绑定到模型中。
    model.hyp = hyp  # attach hyperparameters to model将超参数字典 (hyp) 绑定到模型中。
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)设置 IoU 损失比例，默认为 1.0，表示使用 IoU 计算对象损失。
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights根据数据集中各类别的分布计算类别权重，并绑定到模型上，用于缓解类别不平衡问题。
    model.names = names##将类别名称列表 (names) 绑定到模型中，便于后续输出结果可视化。

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # 计算预热（warmup）阶段的迭代次数，最少为 1000 次
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    maps = np.zeros(nc)  # mAP per class初始化每个类别的 mAP 数组，用于记录验证结果。
    results = (0, 0, 0, 0, 0, 0, 0)  # 初始化评估指标结果，包括 P（精确率）、R（召回率）、mAP@0.5、mAP@0.5-0.95 以及损失值等。
    scheduler.last_epoch = start_epoch - 1  # 设置学习率调度器的起始轮次，确保训练连续性。
    scaler = amp.GradScaler(enabled=cuda)#初始化梯度缩放器，用于混合精度训练，提高训练效率。
    '''
    作用：初始化一个支持 OTA（Optimal Transport Assignment） 的损失函数。
OTA 简介： OTA 是一种更先进的标签分配策略，它将目标检测中的正样本选择问题建模为一个“最优传输”问题，通过计算预测框与真实框之间的匹配代价，动态地为每个真实框选择最佳的预测框。相比传统的基于 IoU 或阈值的静态分配方式，OTA 能够实现更精确、更合理的正样本匹配。
适用场景：
多用于训练后期或高质量训练阶段。
可以提升模型精度，尤其是对密集目标场景表现更好。
    '''
    compute_loss_ota = ComputeLossOTA(model)  # 初始化 OTA 版本的损失计算器，用于优化匹配预测与真实框。
    '''
    作用：初始化一个 基础版本的标准损失函数。
    标准损失组成：
    包括边界框损失（box loss）
    目标置信度损失（objectness loss）
    分类损失（classification loss）
    标签分配方式：
    使用传统的 anchor 匹配策略（基于 IoU 阈值）
    对每个真实框分配一个或多个 anchor 作为正样本
    适用场景：
    默认使用的损失函数
    训练初期或快速迭代时使用
    计算效率高
    '''

    compute_loss = ComputeLoss(model)  # 初始化基础版本的损失计算器，用于训练过程中的损失计算。
    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    # torch.save(model, wdir / 'init.pt')

    with open(wdir / 'init.pt', 'wb') as f:
        pickle.dump(model, f, protocol=4)


    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        #这段代码是 YOLO 模型训练的主循环部分，负责迭代每一个 epoch 并处理每个 batch 的数据加载、预处理、学习率预热等操作。
        model.train()

        #  1. 图像权重更新（可选）
        if opt.image_weights:
            '''
            则根据类别权重和当前 mAP 动态调整训练图像的采样概率。
主进程（rank 0）计算每个图像的权重并随机采样。
使用 dist.broadcast 向其他进程广播采样索引（适用于分布式训练 DDP）。
            '''
            # Generate indices
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(4, device=device)  #  初始化损失记录器,mloss[0]: 边界框损失（box loss）,mloss[1]: 目标置信度损失（objectness loss）,mloss[2]: 分类损失（classification loss）
        if rank != -1:#🔄 4. 分布式训练中重置采样器,在 DDP 模式下，为了保证不同 GPU 上的采样顺序不同但均匀分布，需要在每轮开始时调用 set_epoch(epoch)。
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()#⚙️ 6. 清空优化器梯度

        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            #内层循环，处理每个 batch 的数据。，ni是全局批次索引，nb是总批次数，epoch是当前轮数，total_batch_size是总批次大小。
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0将图像数据移动到指定设备（GPU/CPU），并将像素值归一化到 [0, 1]。

            # Warmup  预热，学习率
            if ni <= nw:
                '''
                使用线性插值逐步增加学习率（bias 组从 warmup_bias_lr 增加到初始 lr，其他组从 0 增加）。
momentum 也随训练步数线性增长。
accumulate 表示多少个 batch 后才更新一次参数，模拟大 batch size 效果。
📌 作用：防止模型初期因学习率过大导致不稳定，提高收敛速度与稳定性。

                '''
                xi = [0, nw]  # x interp
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            #  多尺度训练（Multi-scale Training），
            '''
            作用：在训练过程中随机改变输入图像的尺寸，提高模型对不同尺度目标的鲁棒性。
imgsz 是基础图像大小（如 640），gs 是模型的最大下采样步长（如 32）。
随机选择一个介于 imgsz * 0.5 到 imgsz * 1.5 的图像大小，并确保其为 gs 的整数倍。
计算缩放因子 sf，并根据该因子重新计算图像的新尺寸 ns（保持与 gs 对齐）。
使用双线性插值将图像调整到新尺寸。
            '''
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            '''
            使用 amp.autocast() 启用混合精度训练（FP16/FP32），减少显存占用并加速训练。
将处理后的图像数据 imgs 输入模型，得到预测结果 pred（通常是多个特征层的输出列表）。
            '''
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                if hyp['loss_ota'] == 1:#💥 3. 损失计算（Loss Computation）
                    print(f'---------{device}---------------')
                    loss, loss_items = compute_loss_ota(pred, targets.to(device), imgs)  #ComputeLossOTA: 支持 OTA（Optimal Transport Assignment）的损失函数，用于更精确的正样本分配，适用于高质量训练。
                else:
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size基础版本，使用传统 anchor 匹配策略，计算效率更高。
                if rank != -1:#分布式训练中的损失平均,在 DDP（分布式训练）模式下，总损失乘以 world_size，以便后续梯度平均。
                    loss *= opt.world_size  # gradient averaged between devices in DDP mode
                if opt.quad:#如果启用 --quad（加载四倍大小的 batch），则损失放大 4 倍，模拟更大 batch size 的效果。
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()#使用 GradScaler 缩放损失，防止 FP16 下的梯度溢出。执行反向传播，计算梯度。

            # Optimize
            '''
            梯度累积：每 accumulate 步更新一次权重。
scaler.step(optimizer)：执行优化器更新。
scaler.update()：更新缩放因子。
optimizer.zero_grad()：清空当前梯度。
ema.update(model)：更新 EMA（指数移动平均）模型，提升模型稳定性与推理性能。
            '''
            if ni % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            '''
            更新当前 epoch 中各损失项的平均值。
显示当前使用的显存大小。
构建并显示日志信息，包括：
当前 epoch / 总 epoch；
显存占用；
各项损失；
目标数量；
图像尺寸。
            '''
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 10:# 图像可视化（Plotting）
                    f = save_dir / f'train_batch{ni}.jpg'  # filename
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()
                    # if tb_writer:
                    #     tb_writer.add_image(f, result, dataformats='HWC', global_step=epoch)
                    #     tb_writer.add_graph(torch.jit.trace(model, imgs, strict=False), [])  # add model graph
                elif plots and ni == 10 and wandb_logger.wandb:
                    wandb_logger.log({"Mosaics": [wandb_logger.wandb.Image(str(x), caption=x.name) for x in
                                                  save_dir.glob('train*.jpg') if x.exists()]})

            # end batch ------------------------------------------------------------------------------------------------
        # end epoch ----------------------------------------------------------------------------------------------------

        # Scheduler是一个学习率调度器
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:#主进程或单 GPU
            # mAP
            '''
            test.test(...)：调用测试函数，返回指标如：
results: [P, R, mAP@0.5, mAP@0.5:0.95, ...]
maps: 每个类别的 mAP 值；
times: 推理时间；
model=ema.ema：使用指数移动平均（EMA）模型进行推理，提升稳定性；
imgsz_test：测试图像大小；
dataloader=testloader：验证集数据加载器；
plots=True：保存混淆矩阵、PR 曲线等可视化结果；
is_coco：如果是 COCO 数据集，会启用相应的评估方式
            '''
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights']) # 更新 EMA 模型属性（如类别数、超参数等）
            final_epoch = epoch + 1 == epochs# 判断是否为最后一个 epoch
            if not opt.notest or final_epoch:  # Calculate mAP # 如果不是禁用测试或最后一轮
                wandb_logger.current_epoch = epoch + 1
                results, maps, times = test.test(data_dict, # 调用 test.py 进行验证集测试
                                                 batch_size=batch_size * 2,
                                                 imgsz=imgsz_test,
                                                 model=ema.ema,
                                                 single_cls=opt.single_cls,
                                                 dataloader=testloader,
                                                 save_dir=save_dir,
                                                 verbose=nc < 50 and final_epoch,
                                                 plots=plots and final_epoch,
                                                 wandb_logger=wandb_logger,
                                                 compute_loss=compute_loss,
                                                 is_coco=is_coco)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 7 % results + '\n')  # append metrics, val_loss
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss',  # train loss
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'val/box_loss', 'val/obj_loss', 'val/cls_loss',  # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})  # W&B

            # Update best mAP  更新最佳模型（Best Fitness）
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model  💾 六、
            if (not opt.nosave) or (final_epoch and not opt.evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': results_file.read_text(),
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': wandb_logger.wandb_run.id if wandb_logger.wandb else None}

                # Save last, best and delete
                with open(last, 'wb') as f:
                    pickle.dump(ckpt, f, protocol=4)
                # torch.save(ckpt, last)
                if best_fitness == fi:
                    with open(best, 'wb') as f:
                        pickle.dump(ckpt, f, protocol=4)
                    # torch.save(ckpt, best)
                if (best_fitness == fi) and (epoch >= 200):
                    # torch.save(ckpt, wdir / 'best_{:03d}.pt'.format(epoch))
                    with open( wdir / 'best_{:03d}.pt'.format(epoch), 'wb') as f:
                        pickle.dump(ckpt, f, protocol=4)

                if epoch == 0:
                    # torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                    with open(wdir / 'epoch_{:03d}.pt'.format(epoch), 'wb') as f:
                        pickle.dump(ckpt, f, protocol=4)
                elif ((epoch+1) % 25) == 0:
                    # torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                    with open( wdir / 'epoch_{:03d}.pt'.format(epoch), 'wb') as f:
                        pickle.dump(ckpt, f, protocol=4)
                elif epoch >= (epochs-5):
                    # torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                    with open(wdir / 'epoch_{:03d}.pt'.format(epoch), 'wb') as f:
                        pickle.dump(ckpt, f, protocol=4)
                if wandb_logger.wandb:
                    if ((epoch + 1) % opt.save_period == 0 and not final_epoch) and opt.save_period != -1:
                        wandb_logger.log_model(
                            last.parent, opt, epoch, fi, best_model=best_fitness == fi)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training  训练结束后的操作
    '''
    plot_results(...)：绘制训练过程中的 loss、precision、recall、mAP 曲线；
如果启用了 WandB，则上传这些图表。
    '''
    if rank in [-1, 0]:
        # Plots
        # if plots:
        #     plot_results(save_dir=save_dir)  # save as results.png
        #     if wandb_logger.wandb:
        #         files = ['results.png', 'confusion_matrix.png', *[f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R')]]
        #         wandb_logger.log({"Results": [wandb_logger.wandb.Image(str(save_dir / f), caption=f) for f in files
        #                                       if (save_dir / f).exists()]})
        # Test best.pt
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

        #最终测试（Final Test）,在 COCO 数据集上执行最终测试；,输出 JSON 文件用于官方提交。
        if opt.data.endswith('coco.yaml') and nc == 80:  # if COCO
            for m in (last, best) if best.exists() else (last):  # speed, mAP tests
                results, _, _ = test.test(opt.data,
                                          batch_size=batch_size * 2,
                                          imgsz=imgsz_test,
                                          conf_thres=0.001,
                                          iou_thres=0.7,
                                          model=attempt_load(m, device).half(),
                                          single_cls=opt.single_cls,
                                          dataloader=testloader,
                                          save_dir=save_dir,
                                          save_json=True,
                                          plots=False,
                                          is_coco=is_coco)

        # Strip optimizers 删除优化器
        final = best if best.exists() else last  # final model
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers 、清理优化器（Strip Optimizers）,对部署和推理更友好。
        if opt.bucket:
            os.system(f'gsutil cp {final} gs://{opt.bucket}/weights')  # upload 上传云端，如果设置了 --bucket，则自动上传训练结果文件。
        if wandb_logger.wandb and not opt.evolve:  # Log the stripped model
            wandb_logger.wandb.log_artifact(str(final), type='model',
                                            name='run_' + wandb_logger.wandb_run.id + '_model',
                                            aliases=['last', 'best', 'stripped']) #WandB 模型上传
        wandb_logger.finish_run()
    else:
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',type=str,default='./yolov7.pt')
    parser.add_argument('--cfg',type=str,default='')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')  ##数据地址
    parser.add_argument('--hyp',type=str,default='data/hyp.scratch.p5.yaml')
    parser.add_argument('--epochs', type=int, default=300)  ####循环几次
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')  ###一次有多少个图片
    parser.add_argument('--img-size',nargs='+',type=int,default=[640,640])
    parser.add_argument('--rect', action='store_true')
    parser.add_argument('--resume',nargs='?',const=True,default=False,)
    parser.add_argument('--nosave',action='store_true')
    parser.add_argument('--notest',action='store_true')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')  ##自动检查
    parser.add_argument('--evole',action='store_true')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')  ###学习进行迭代
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')  ##图像保存
    parser.add_argument('--image-weights', action='store_true',
                        help='use weighted image selection for training')  ##图像权重
    parser.add_argument('--device',default='')
    parser.add_argument('--multi-scale',action='store_true')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')  ###优化器
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')  ###跨卡
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')  #
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')  ###项目保存的路径
    parser.add_argument('--entity', default=None, help='W&B entity')  ##可视化相关的，用于日志
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')  ##学习率
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')  ##平滑标签处理
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1,
                        help='Set bounding-box image logging interval for W&B')  ##也是日志用到的
    parser.add_argument('--save_period', type=int, default=-1,
                        help='Log model after every "save_period" epoch')  ##保存最好的一次
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0],
                        help='Freeze layers: backbone of yolov7=50, first3=0 1 2')  ##冻结，一般只冻结前几层
    opt = parser.parse_args()
    ''' 
    --workers 1
--device 0
--batch-size 4
--data  data/neu.yaml
--cfg cfg/training/yolov7.yaml
--weights yolov7.pt
--name yolov7
--hyp data/hyp.scratch.p5.yaml
--epochs 20   '''
    opt.world_size=1
    opt.global_rank =-1  ###单程
    set_logging(opt.global_rank)
    # wandb_run=check_wandb_resume(opt)
    # print(wandb_run,opt.resume)
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check
    opt.name = 'evolve' if opt.evolve else opt.name   ##不进行学习的迭代

    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  ##文件保存的路径
    # DDP mode # DDP mode
    opt.total_batch_size = opt.batch_size
    device=select_device(opt.device,batch_size=opt.batch_size)
    print(device)

    '''以上是准备，超参数文件，数据文件，模型文件，使用的设备，是否使用分布式，保存的路径'''
    with open(opt.hyp) as f: hyp=yaml.load(f,Loader=yaml.SafeLoader)

    ###训练数据
    logging.info(opt)
    if not opt.evolve:  ####没有进行学习迭代的模型训练
        tb_writer = None  # init loggers
        train(hyp,opt,device,tb_writer)


