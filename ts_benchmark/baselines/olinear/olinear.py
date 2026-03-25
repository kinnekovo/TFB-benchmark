import torch
import torch.nn as nn
from torch import optim
import os

from .models.olinear_model import Model as OLinearModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# 定义 OLinear 标准超参数
MODEL_HYPER_PARAMS = {
    "d_model": 256,
    "d_ff": 512,
    "e_layers": 2,
    "dropout": 0.1,
    "embed_size": 1,
    "temp_patch_len": 1,
    "temp_stride": 1,
    "activation": "gelu",
    "CKA_flag": False,
    "Q_chan_indep": False,
    "q_mat_file": "dataset/ILI_Q.npy",
    "q_out_mat_file": "dataset/ILI_Q_out.npy",
    "root_path": "./",
    "lr": 0.001,
}

class OLinear(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        # 动态处理 root_path，优先使用传入参数或环境变量
        if 'root_path' not in kwargs:
            kwargs['root_path'] = os.getcwd()
        super(OLinear, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "OLinear"

    def _init_criterion_and_optimizer(self):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        return criterion, optimizer

    def _init_model(self):
        # 确保 Q 矩阵文件路径是绝对路径或相对于项目根目录
        return OLinearModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        # TFB 标准数据流转
        outputs = self.model(input)
        return {"output": outputs}