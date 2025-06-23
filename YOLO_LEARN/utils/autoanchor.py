import numpy as np
import torch
import yaml
from scipy.cluster.vq import kmeans
from tqdm import tqdm

from 物体检测.YOLO_LEARN.utils.datasets import LoadImagesAndLabels
from 物体检测.YOLO_LEARN.utils.general import colorstr

def check_anchor_order(m):
    '''确保锚框的顺序与特征图下采样步长（stride）一致。如果顺序不匹配，则翻转锚框顺序。
🔍 解释：
    a: 锚框面积。
    da, ds: 分别为锚框面积和 stride 的差值。
    如果符号不同（即增长方向相反），说明锚框顺序错误。
    使用 flip(0) 翻转锚框顺序以保持一致性。
 '''
    # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # Check anchor fit to data, recompute if necessary
    '''

    是 YOLOv7 中用于 自动计算最优锚框（anchor boxes） 的函数。它会根据训练数据集中的标注框尺寸，使用聚类算法（如 K-means）重新生成与数据匹配度更高的锚框，以提升模型的检测性能。
    自动分析训练集中所有目标框的宽高分布
使用聚类算法（K-means 或类似方法）生成最合适的锚框
将生成的锚框与当前模型使用的锚框进行对比
如果差异较大，则提示用户更新锚框
    :param dataset: 数据集对象，包含所有图像的标签信息（如 [class_id, x_center, y_center, width, height]）
    :param model: 模型对象，用于获取当前模型使用的锚框
    :param thr: 锚框匹配阈值（IoU），用于判断是否需要更新锚框
    :param imgsz: 图像尺寸，用于归一化标注框尺寸
    :return:
    '''
    prefix = colorstr('autoanchor: ')
    print(f'\n{prefix}Analyzing anchors... ', end='')
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()模型的最后一层
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)#图片尺寸图像归一化后的尺寸。【640，640】
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale数据增强用的缩放因子。
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh所有标注框的宽高（已归一化）。图片尺寸*缩放因子*标注框宽高。

    def metric(k):  # compute metric  定义评估指标函数
        r = wh[:, None] / k[None]#wh所有标注框的宽高（已归一化）/候选框
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric小比例误差。
        best = x.max(1)[0]  # best_x 个目标框最佳匹配锚框。
        aat = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold平均每个目标框有多少锚框匹配良好。
        bpr = (best > 1. / thr).float().mean()  # best possible recall最佳可能召回率（Best Possible Recall）。
        return bpr, aat

    anchors = m.anchor_grid.clone().cpu().view(-1, 2)  # current anchors
    bpr, aat = metric(anchors)
    print(f'anchors/target = {aat:.2f}, Best Possible Recall (BPR) = {bpr:.4f}', end='')
    if bpr < 0.98:  # threshold to recompute
        print('. Attempting to improve anchors, please wait...')
        na = m.anchor_grid.numel() // 2  # number of anchors
        try:
            anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)#调用 kmean_anchors 生成新的锚框。
        except Exception as e:
            print(f'{prefix}ERROR: {e}')
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)#将新锚框保存到模型中。
            m.anchor_grid[:] = anchors.clone().view_as(m.anchor_grid)  # for inference
            check_anchor_order(m)#检查并调整锚框顺序。
            m.anchors[:] = anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss根据 stride 缩放锚框用于损失函数计算。
            print(f'{prefix}New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print(f'{prefix}Original anchors better than new anchors. Proceeding with original anchors.')
    print('')  # newline

def kmean_anchors(path='./data/coco.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: 数据集配置文件路径或直接传入 dataset 对象
            n: 生成的锚框数量（通常为 9
            img_size: 图像尺寸
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0锚框匹配阈值
            gen: generations to evolve anchors using genetic algorithm进化代数
            verbose: print all results是否打印详细信息

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    thr = 1. / thr#将阈值倒数处理便于后续计算。
    prefix = colorstr('autoanchor: ')

    def metric(k, wh):  # compute metrics计算锚框与标注框的匹配度。
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness计算锚框适应度（fitness）。
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):#输出结果。
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print(f'{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr')
        print(f'{prefix}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, '
              f'past_thr={x[x > thr].mean():.3f}-mean: ', end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k
    '''获取分类数据和训练数据'''
    if isinstance(path, str):  # *.yaml file
        with open(path) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
    else:
        dataset = path  # dataset

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter 过滤异常框
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(f'{prefix}WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels
    # wh = wh * (np.random.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans calculation 6. K-means 聚类，对标注框进行标准化后聚类。，恢复原始尺度
    print(f'{prefix}Running kmeans for {n} anchors on {len(wh)} points...')
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    assert len(k) == n, print(f'{prefix}ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}')
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unfiltered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve  . 遗传算法优化使用变异和选择策略优化聚类中心，提升锚框对数据集的适配性。
    npr = np.random
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'{prefix}Evolving anchors with Genetic Algorithm:')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{prefix}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k)

    return print_results(k)