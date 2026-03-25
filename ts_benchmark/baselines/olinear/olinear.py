from ts_benchmark.baselines.olinear.models.olinear_model import Model as OLinearModel
from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase

# model hyper params
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
    "batch_size": 32,
    "lr": 0.001,
    "num_epochs": 10,
    "loss": "MSE",
    "patience": 5,
}


class OLinear(DeepForecastingModelBase):
    """
    OLinear adapter class.

    Attributes:
        model_name (str): Name of the model for identification purposes.
        _init_model: Initializes an instance of OLinearModel.
        _process: Executes the model's forward pass and returns the output.
    """

    def __init__(self, **kwargs):
        super(OLinear, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "OLinear"

    def _init_model(self):
        return OLinearModel(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output = self.model(input)
        return {"output": output}
