from torch.nn import Module
import torch
from architectures import PN_head, get_backbone
import torch.nn.functional as F
from . import utils
from .base_module import BaseFewShotModule
from typing import Tuple, List, Optional, Union, Dict
class ProtoNet(BaseFewShotModule):
    r"""The datamodule implementing Prototypical Network.
    """
    def __init__(
        self,
        metric: str = "cosine",
        scale_cls: float = 10.,
        normalize: bool = True,
        backbone_name: str = "resnet12",      
        train_way: int = 5,
        val_way: int = 5,
        test_way: int = 5,
        train_shot: int = 5,
        val_shot: int = 5,
        test_shot: int = 5,
        num_query: int = 15,
        train_batch_size_per_gpu: int = 2,
        val_batch_size_per_gpu: int = 2,
        test_batch_size_per_gpu: int = 2,
        lr: float = 0.1,
        weight_decay: float = 5e-4,
        decay_scheduler: Optional[str] = "cosine",
        optim_type: str = "sgd",
        decay_epochs: Union[List, Tuple, None] = None,
        decay_power: Optional[float] = None,
        backbone_kwargs: Dict = {},
        **kwargs
    ) -> None:
        """   
        Args:
            metric: what metrics applied. "cosine" or "euclidean".
            scale_cls: The initial scale number which affects the 
                    following softmax function.
            normalize: Whether normalize each spatial dimension of image features before average pooling.
            backbone_name: The name of the feature extractor, 
                        which should match the correspond 
                        file name in architectures.feature_extractor
            train_way: The number of classes within one training task.
            val_way: The number of classes within one training task.
            test_way: The number of classes within one training task.
            train_shot: The number of samples within each few-shot 
                        support class during training. 
                        For meta-learning only.
            val_shot: The number of samples within each few-shot 
                    support class during validation.
            test_shot: The number of samples within each few-shot 
                    support class during testing.
            num_query: The number of samples within each few-shot 
                    query class.
            train_batch_size_per_gpu: The batch size of training per GPU.
            val_batch_size_per_gpu: The batch size of validation per GPU.
            test_batch_size_per_gpu: The batch size of testing per GPU.
            lr: The initial learning rate.
            weight_decay: The weight decay parameter.
            decay_scheduler: The scheduler of optimizer.
                            "cosine" or "specified_epochs".
            optim_type: The optimizer type.
                        "sgd" or "adam"
            decay_epochs: The list of decay epochs of decay_scheduler "specified_epochs".
            decay_power: The decay power of decay_scheduler "specified_epochs"
                        at eachspeicified epoch.
                        i.e., adjusted_lr = lr * decay_power
            backbone_kwargs: The parameters for creating backbone network.
        """
        super().__init__(
            backbone_name, train_way, val_way, test_way, train_shot, val_shot,
            test_shot, num_query, train_batch_size_per_gpu,
            val_batch_size_per_gpu, test_batch_size_per_gpu,
            lr, weight_decay, decay_scheduler, optim_type,
            decay_epochs, decay_power, backbone_kwargs
        )
        self.classifier = PN_head(metric, scale_cls, normalize=normalize)


    def forward(self, batch, batch_size, way, shot):
        r"""Since PN is a meta-learning method,
            the model forward process is the same for train, val and test.
        
        Args:
            batch: a batch from val_dataloader.
            batch_size: number of tasks during one iteration.
            way: The number of classes within one task.
            shot: The number of samples within each few-shot support class. 
        """
        num_support_samples = way * shot
        data, _ = batch
        data = self.backbone(data)
        data = data.reshape([batch_size, -1] + list(data.shape[-3:]))
        data_support = data[:, :num_support_samples]
        data_query = data[:, num_support_samples:]
        logits = self.classifier(data_query, data_support, way, shot)
        return logits

    def train_forward(self, batch, batch_size, way, shot):
        return self(batch, batch_size, way, shot)

    def val_test_forward(self, batch, batch_size, way, shot):
        return self(batch, batch_size, way, shot)

def get_model():
    return ProtoNet


    
        



