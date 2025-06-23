import os
import subprocess
from pathlib import Path

import requests
import torch
from torch import nn

from 物体检测.YOLO_LEARN.models.common import Conv


class Ensemble(nn.ModuleList):#多模型集成推理
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

def attempt_download(file, repo='WongKinYiu/yolov7'):#自动下载权重文件，加载模型权重并设置设备
    # Attempt file download if does not exist
    file = Path(str(file).strip().replace("'", '').lower())

    if not file.exists():
        try:
            response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()  # github api
            assets = [x['name'] for x in response['assets']]  # release assets
            tag = response['tag_name']  # i.e. 'v1.0'
        except:  # fallback plan
            assets = ['yolov7.pt', 'yolov7-tiny.pt', 'yolov7x.pt', 'yolov7-d6.pt', 'yolov7-e6.pt',
                      'yolov7-e6e.pt', 'yolov7-w6.pt']
            tag = subprocess.check_output('git tag', shell=True).decode().split()[-1]

        name = file.name
        if name in assets:
            msg = f'{file} missing, try downloading from https://github.com/{repo}/releases/'
            redundant = False  # second download option
            try:  # GitHub
                url = f'https://github.com/{repo}/releases/download/{tag}/{name}'
                print(f'Downloading {url} to {file}...')
                torch.hub.download_url_to_file(url, file)
                assert file.exists() and file.stat().st_size > 1E6  # check
            except Exception as e:  # GCP
                print(f'Download error: {e}')
                assert redundant, 'No secondary mirror'
                url = f'https://storage.googleapis.com/{repo}/ckpt/{name}'
                print(f'Downloading {url} to {file}...')
                os.system(f'curl -L {url} -o {file}')  # torch.hub.download_url_to_file(url, weights)
            finally:
                if not file.exists() or file.stat().st_size < 1E6:  # check
                    file.unlink(missing_ok=True)  # remove partial downloads
                    print(f'ERROR: Download failure: {msg}')
                print('')
                return
def attempt_load(weights, map_location=None):#加载模型权重并设置设备
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model 提取模型并进行优化处理，将模型转为 FP32 格式，确保计算精度；，行层融合（如 Conv + BN 融合），提升推理速度；设置为评估模式（关闭 Dropout、BatchNorm 的训练行为）；

    # Compatibility updates 模块兼容性修复
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:#则构建一个集成模型（Ensemble）并返回；
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble

