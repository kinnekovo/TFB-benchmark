import os


def integrate():
    # 1. 修正路径与目录结构
    base_path = "ts_benchmark/baselines/olinear"
    sub_dirs = ["layers", "models", "utils"]

    for sd in sub_dirs:
        os.makedirs(os.path.join(base_path, sd), exist_ok=True)
        # 确保每个子目录都是 package
        with open(os.path.join(base_path, sd, "__init__.py"), "a") as f:
            pass

    # 2. 修正 __init__.py (处理之前的拼写错误 _init_.py)
    if os.path.exists(os.path.join(base_path, "_init_.py")):
        os.rename(os.path.join(base_path, "_init_.py"), os.path.join(base_path, "__init__.py"))

    with open(os.path.join(base_path, "__init__.py"), "w") as f:
        f.write("from .olinear import OLinear\\n")

    # 3. 核心适配器重写 (修复硬编码路径，支持 TFB 动态配置)
    olinear_py_content = """
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
"""
    with open(os.path.join(base_path, "olinear.py"), "w") as f:
        f.write(olinear_py_content.strip())

    # 4. 修正全局基准注册
    baselines_init = "ts_benchmark/baselines/__init__.py"
    with open(baselines_init, "r") as f:
        content = f.read()

    if "from .olinear import OLinear" not in content:
        with open(baselines_init, "a") as f:
            f.write("\\nfrom .olinear import OLinear\\n")

    print("✅ Baseline 结构集成完成！")
    print("👉 请注意：你需要将 OLinear 仓库中的 .py 文件拷贝到 ts_benchmark/baselines/olinear/ 对应的子目录中。")


if __name__ == "__main__":
    integrate()