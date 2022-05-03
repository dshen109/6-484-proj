from easyrl.models.categorical_policy import CategoricalPolicy
from easyrl.models.diag_gaussian_policy import DiagGaussianPolicy


class CategoricalPolicyFiltering(CategoricalPolicy):

    def __init__(self, body_net, action_dim, state_indices, in_features=None):
        """
        Policy that filters out a subset of the observation space.

        :nn body_net: Should take in same amount of inputs as
            len(state_indices)
        """

        super().__init__(body_net=body_net, action_dim=action_dim,
                         in_features=in_features)
        self.state_indices = state_indices

    def forward(self, x, filtered=False, **kwargs):
        if not filtered:
            if len(x.shape) == 1:
                x = x.reshape((1, -1))[:, self.state_indices].squeeze(0)
            elif len(x.shape) == 3:
                x = x[:, :, self.state_indices]
            else:
                raise RuntimeError
        return super().forward(x=x, **kwargs)


class DiagGaussianPolicyFiltering(DiagGaussianPolicy):
    def __init__(self, body_net, action_dim, state_indices, in_features=None):
        """
        Policy that filters out a subset of the observation space.

        :nn body_net: Should take in same amount of inputs as
            len(state_indices)
        """

        super().__init__(body_net=body_net, action_dim=action_dim,
                         in_features=in_features)
        self.state_indices = state_indices

    def forward(self, x, filtered=False, **kwargs):
        if not filtered:
            if len(x.shape) == 1:
                x = x.reshape((1, -1))[:, self.state_indices].squeeze(0)
            elif len(x.shape) == 3:
                x = x[:, :, self.state_indices]
            else:
                raise RuntimeError
        return super().forward(x=x, **kwargs)
