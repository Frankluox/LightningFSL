from .base_module import BaseFewShotModule
import torch.nn as nn
from typing import Tuple, List, Optional, Union, Dict
import torch.nn.functional as F
from architectures import get_classifier
class CE_Pretrainer(BaseFewShotModule):
    r"""The strandard pretraining procedure adopted in many
        Few-Shot methods using a simple cross-entropy loss.
    """
    def __init__(
        self,
        num_classes: int = 64,
        task_classifier_name: str = "proto_head",
        task_classifier_params: Dict = {"learn_scale":False},
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
            num_classes: The number of classes of the training dataset.
            task_classifier_name: The name of the classifier for downstream (val, test)
                                  few-shot tasks. It should match the name of file that 
                                  contains the classifier class.
            task_classifier_params: The initial parameters of the classifier for 
                                    downstream (val, test) few-shot tasks.
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
        self.classifier = nn.Linear(self.backbone.outdim, num_classes)
        self.val_test_classifier = get_classifier(task_classifier_name, **task_classifier_params)

    def train_forward(self, batch):
        data, labels = batch
        features = self.backbone(data)
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).squeeze_(-1).squeeze_(-1)
        logits = self.classifier(features)
        return logits, labels

    def val_test_forward(self, batch, batch_size, way, shot):
        num_support_samples = way * shot
        data, _ = batch
        data = self.backbone(data)
        data = data.reshape([batch_size, -1] + list(data.shape[-3:]))
        data_support = data[:, :num_support_samples]
        data_query = data[:, num_support_samples:]
        logits = self.val_test_classifier(data_query, data_support, way, shot)
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
    return CE_Pretrainer
    
