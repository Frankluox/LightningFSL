import torch
import torch.nn as nn
import torch.nn.functional as F
from architectures.utils import L2SquareDist
from torch import Tensor



class PN_head(nn.Module):
    r"""The metric-based protypical classifier from ``Prototypical Networks for Few-shot Learning''.

    Args:
        metric: Whether use cosine or enclidean distance.
        scale_cls: The initial scale number which affects the following softmax function.
        learn_scale: Whether make scale number learnable.
        normalize: Whether normalize each spatial dimension of image features before average pooling.
    """
    def __init__(
        self, 
        metric: str = "cosine", 
        scale_cls: int =10.0, 
        learn_scale: bool = True, 
        normalize: bool = True) -> None:
        super().__init__()
        assert metric in ["cosine", "enclidean"]
        if learn_scale:
            self.scale_cls = nn.Parameter(
                torch.FloatTensor(1).fill_(scale_cls), requires_grad=True
            )    
        else:
            self.scale_cls = scale_cls
        self.metric = metric
        self.normalize = normalize

    def forward(self, features_test: Tensor, features_train: Tensor, 
                way: int, shot: int) -> Tensor:
        r"""Take batches of few-shot training examples and testing examples as input,
            output the logits of each testing examples.

        Args:
            features_test: Testing examples. size: [batch_size, num_query, c, h, w]
            features_train: Training examples which has labels like:[abcdabcdabcd].
                            size: [batch_size, way*shot, c, h, w]
            way: The number of classes of each few-shot classification task.
            shot: The number of training images per class in each few-shot classification
                  task.
        Output:
            classification_scores: The calculated logits of testing examples.
                                   size: [batch_size, num_query, way]
        """
        if features_train.dim() == 5:
            if self.normalize:
                features_train=F.normalize(features_train, p=2, dim=2, eps=1e-12)
            features_train = F.adaptive_avg_pool2d(features_train, 1).squeeze_(-1).squeeze_(-1)
        assert features_train.dim() == 3

        batch_size = features_train.size(0)
        if self.metric == "cosine":
            features_train = F.normalize(features_train, p=2, dim=2, eps=1e-12)
        #prototypes: [batch_size, way, c]
        prototypes = torch.mean(features_train.reshape(batch_size, shot, way, -1),dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=2, eps=1e-12)

        if self.normalize:
            features_test=F.normalize(features_test, p=2, dim=2, eps=1e-12)
        if features_test.dim() == 5:
            features_test = F.adaptive_avg_pool2d(features_test, 1).squeeze_(-1).squeeze_(-1)
        assert features_test.dim() == 3

        if self.metric == "cosine":
            features_test = F.normalize(features_test, p=2, dim=2, eps=1e-12)
            #[batch_size, num_query, c] * [batch_size, c, way] -> [batch_size, num_query, way]
            classification_scores = self.scale_cls * torch.bmm(features_test, prototypes.transpose(1, 2))

        elif self.metric == "euclidean":
            classification_scores = -self.scale_cls * L2SquareDist(features_test, prototypes)
        return classification_scores

def create_model(metric: str = "cosine", 
        scale_cls: int =10.0, 
        learn_scale: bool = True, 
        normalize: bool = True):
    return PN_head(metric, scale_cls, learn_scale, normalize)
