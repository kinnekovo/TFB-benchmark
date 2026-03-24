import torch
import torch.nn as nn
from torch import optim

# 导入我们刚才放进去的模型类
from ts_benchmark.baselines.olinear.models.olinear_model import Model as OLinearModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# 定义 OLinear 的默认超参数 (你可以根据 ILI.csv 的经验填入)
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
    # 下面这两个路径非常关键，建议先写你 D 盘的绝对路径测试
    "q_mat_file": "D:/10245501444ECNU/决策智能实验室/OLinear/dataset/ILI_Q.npy",
    "q_out_mat_file": "D:/10245501444ECNU/决策智能实验室/OLinear/dataset/ILI_Q_out.npy",
    "root_path": "./",
    "batch_size": 32,
    "lr": 0.001,
    "num_epochs": 10,
    "patience": 3,
    "use_amp": 0,  # 是否使用自动混合精度
}


class OLinear(DeepForecastingModelBase):
    def __init__(self, **kwargs):
        # 这里的 super 会自动处理 MODEL_HYPER_PARAMS 和外部传入参数的合并
        super(OLinear, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "OLinear"

    def _init_criterion_and_optimizer(self):
        # 定义损失函数和优化器，TFB 的训练循环会自动调用它们
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)
        return criterion, optimizer

    def _init_model(self):
        # 这里的 self.config 是一个 DotDict，包含了所有超参
        return OLinearModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        """
        这是 TFB 训练和预测时的统一入口。
        input: [B, seq_len, C]
        target: [B, pred_len, C]
        """
        # OLinear 的 forward 签名是 forward(x, x_mark_enc, x_dec, x_mark_dec, mask)
        # 虽然它内部只用了 x，但我们还是把参数传全，防止报错
        if self.config.use_amp == 1:
            with torch.cuda.amp.autocast():
                outputs = self.model(input)
        else:
            outputs = self.model(input)

        # 结果必须封装成带 "output" 键的字典
        return {"output": outputs}