import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

def weight_norm(module, name='weight', dim=0):
    r"""Applies weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    WeightNorm.apply(module, name, dim)
    return module

class CC_head(nn.Module):
    def __init__(self, indim, outdim,scale_cls=10.0, learn_scale=True, normalize=True):
        super().__init__()
        self.L = weight_norm(nn.Linear(indim, outdim, bias=False), name='weight', dim=0)
        self.scale_cls = nn.Parameter(
            torch.FloatTensor(1).fill_(scale_cls), requires_grad=learn_scale
        )
        self.normalize=normalize

    def forward(self, features):
        if features.dim() == 4:
            if self.normalize:
                features=F.normalize(features, p=2, dim=1, eps=1e-12)
            features = F.adaptive_avg_pool2d(features, 1).squeeze_(-1).squeeze_(-1)
        assert features.dim() == 2
        x_normalized = F.normalize(features, p=2, dim=1, eps = 1e-12)
        self.L.weight.data = F.normalize(self.L.weight.data, p=2, dim=1, eps = 1e-12)
        cos_dist = self.L(x_normalized)
        classification_scores = self.scale_cls * cos_dist

        return classification_scores



def create_model(indim, outdim,  
        scale_cls: int =10.0, 
        learn_scale: bool = True, 
        normalize: bool = True):
    return CC_head(indim, outdim, scale_cls, learn_scale, normalize)
    
if __name__ == "__main__":
    layer = nn.Linear(3, 5, bias=False)
    x= torch.zeros((2,3,4,4))
    print(layer(x))


