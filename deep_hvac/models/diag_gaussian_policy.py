from easyrl.models import diag_gaussian_policy
import torch


class DiagGaussianPolicy(diag_gaussian_policy.DiagGaussianPolicy):
    """
    Diagonal Gaussian Policy that allows for setting the
    initial bias to save some time during training.
    """
    def __init__(self, initial_bias=None, **kwargs):
        """
        :param tensor initial_bias:
        """
        super().__init__(**kwargs)
        if (initial_bias is not None and
            len(initial_bias) != kwargs['action_dim']):
            raise ValueError(
                "Initial bias should match length of action space")
        if initial_bias is not None:
            self.head_mean.bias = torch.nn.Parameter(
                initial_bias.type(torch.FloatTensor))
