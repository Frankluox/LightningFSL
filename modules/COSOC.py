from .base_module import BaseFewShotModule
from architectures import CC_head, SOC
from typing import Tuple, List, Optional, Union, Dict
import torch.nn.functional as F
class COSOC(BaseFewShotModule):
    r"""The datamodule implementing COSOC after foreground extraction of COS.
    """
    def __init__(
        self,
        SOC_params: Dict,
        num_classes: int = 64,
        scale_cls: float = 10.,
        backbone_name: str = "resnet12",      
        train_way: int = 5,
        val_way: int = 5,
        test_way: int = 5,
        val_shot: int = 5,
        test_shot: int = 5,
        num_query: int = 15,
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
            SOC_params: hyperparameters in the SOC algorithm.
            num_classes: The number of classes of the training dataset.
            scale_cls: The initial scale number which affects the 
                        following softmax function.
            backbone_name: The name of the feature extractor, 
                        which should match the correspond 
                        file name in architectures.feature_extractor
            train_way: The number of classes within one training task.
            val_way: The number of classes within one val task.
            test_way: The number of classes within one testing task.
            val_shot: The number of samples within each few-shot 
                    support class during validation.
            test_shot: The number of samples within each few-shot 
                    support class during testing.
            num_query: The number of samples within each few-shot 
                    query class.
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
            backbone_name=backbone_name, train_way=train_way, val_way=val_way, test_way=test_way, val_shot=val_shot,
            test_shot=test_shot, num_query=num_query, 
            val_batch_size_per_gpu=val_batch_size_per_gpu, test_batch_size_per_gpu=test_batch_size_per_gpu,
            lr=lr, weight_decay=weight_decay, decay_scheduler=decay_scheduler, optim_type=optim_type,
            decay_epochs=decay_epochs, decay_power=decay_power, backbone_kwargs = backbone_kwargs
        )
        self.classifier = CC_head(self.backbone.outdim, num_classes, scale_cls)
        self.SOC_classifier = SOC(**SOC_params)

    def train_forward(self, batch):
        data, labels = batch
        features = self.backbone(data)
        logits = self.classifier(features)
        return logits, labels

    def val_test_forward(self, batch, batch_size, way, shot):
        data, _ = batch
        logits = self.SOC_classifier(self.backbone, data, way, shot, batch_size)
        return logits
    
    def training_step(self, batch, batch_idx):
        logits, labels = self.train_forward(batch)
        loss = F.cross_entropy(logits, labels)
        log_loss = self.train_loss(loss)
        accuracy = self.train_acc(logits, labels)
        self.log("train/loss", log_loss)
        self.log("train/acc", accuracy)
        return loss
        

def get_model():
    return COSOC

