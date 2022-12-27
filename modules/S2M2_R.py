from .base_module import BaseFewShotModule
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from architectures import CC_head
from architectures.classifier.finetune_classifier import Finetuner
from typing import Tuple, List, Optional, Union, Dict
import torch
from torchmetrics import Accuracy, AverageMeter
from . import utils
class S2M2_R(BaseFewShotModule):
    r"""Implement S2M2_R in articale:
        "Charting the Right Manifold: Manifold Mixup for Few-shot Learning"
    """
    def __init__(
        self,
        is_test: int = False,
        switch_epoch: int = 400,
        alpha: float = 2.0,
        num_classes: int = 64,
        scale_cls: float = 10.,
        ft_batchsize: int = 4,
        ft_epochs: int = 300,
        ft_lr: float = 0.1,
        ft_wd: float = 0.001,
        backbone_name: str = "resnet12",
        train_way: int = 5,
        val_way: int = 5,
        test_way: int = 5,
        train_shot: Optional[int] = None,
        val_shot: int = 5,
        test_shot: int = 5,
        num_query: int = 15,
        train_batch_size_per_gpu: Optional[int] = None,
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
        super().__init__(
            backbone_name=backbone_name, train_way=train_way, val_way=val_way, test_way=test_way, val_shot=val_shot,
            test_shot=test_shot, num_query=num_query, 
            val_batch_size_per_gpu=val_batch_size_per_gpu, test_batch_size_per_gpu=test_batch_size_per_gpu,
            lr=lr, weight_decay=weight_decay, decay_scheduler=decay_scheduler, optim_type=optim_type,
            decay_epochs=decay_epochs, decay_power=decay_power, backbone_kwargs = backbone_kwargs
        )
        self.switch_epoch = switch_epoch
        self.alpha = alpha
        self.rotate_classifier = nn.Linear(self.backbone.outdim, 4)
        self.cosine_classifier = CC_head(self.backbone.outdim, num_classes, scale_cls, learn_scale=False,normalize=False)
        ft_CC_params = {"indim":self.backbone.outdim, "outdim":train_way, "scale_cls":scale_cls, "learn_scale":False, "normalize": False}
        if is_test:
            self.finetune_classifier = Finetuner(ft_batchsize, ft_epochs,ft_lr, ft_wd, test_way, "CC_head", ft_CC_params)
        else:
            self.finetune_classifier = Finetuner(ft_batchsize, ft_epochs,ft_lr, ft_wd, val_way, "CC_head", ft_CC_params)
        
        self.rotate_labels = torch.tensor([0,1,2,3])

    def train_forward(self, batch):
        data, labels = batch
        batch_size = data.size(0)
        if self.trainer.current_epoch>=self.switch_epoch:
            lam = np.random.beta(self.alpha, self.alpha)
            features, labels_new = \
                self.backbone(data, labels, mixup_hidden=True,
                              mixup_alpha = self.alpha,lam = lam)
                                                                                                   
            mixed_up_logits = self.cosine_classifier(features)
            x = data[:int(batch_size/4)]
            batch_size = x.size(0)
            # import pdb
            # pdb.set_trace()
        else:
            x = data
        
        x90 = x.transpose(3,2).flip(2)
        x180 = x90.transpose(3,2).flip(2)
        x270 = x180.transpose(3,2).flip(2)
        rotate_labels = self.rotate_labels.repeat(batch_size).to(x.device)
        x = torch.stack((x,x90,x180,x270),dim=1).reshape([-1]+list(x.shape[-3:]))
        # import pdb
        # pdb.set_trace()
        features = self.backbone(x)
        logits_normal = self.cosine_classifier(features)
        normal_labels = labels[:batch_size].unsqueeze_(1).repeat(1,4).reshape(-1)
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).squeeze_(-1).squeeze_(-1)
        logits_rotate = self.rotate_classifier(features)
        if self.trainer.current_epoch>=self.switch_epoch:
            return logits_normal, normal_labels, logits_rotate, mixed_up_logits, labels, rotate_labels, labels_new, lam
        else:
            return logits_normal, normal_labels, logits_rotate, rotate_labels


        
    def training_step(self, batch, batch_idx):
        total_loss = 0.
        if self.trainer.current_epoch>=self.switch_epoch:
            logits_normal, normal_labels, logits_rotate, mixed_up_logits, labels, rotate_labels, labels_new, lam\
                = self.train_forward(batch)
            mixed_up_loss = lam*F.cross_entropy(mixed_up_logits, labels)+\
                        (1-lam)*F.cross_entropy(mixed_up_logits, labels_new)
            total_loss += mixed_up_loss

            log_mixed_up_loss = self.mixed_up_loss(mixed_up_loss)
            self.log("train/mixed_up_loss", log_mixed_up_loss)
        else:
            logits_normal, normal_labels, logits_rotate, rotate_labels = self.train_forward(batch)
        
        rotation_loss = F.cross_entropy(logits_rotate, rotate_labels)
        normal_loss = F.cross_entropy(logits_normal, normal_labels)
        total_loss += (rotation_loss+normal_loss)/2


        log_rotation_loss = self.rotation_loss(rotation_loss)
        log_normal_loss = self.normal_loss(normal_loss)
        log_total_loss = self.total_loss(total_loss)
        
        rotation_acc = self.rotation_acc(logits_rotate, rotate_labels)
        normal_acc = self.normal_acc(logits_normal, normal_labels)

        self.log("train/rotation_loss", log_rotation_loss)
        self.log("train/normal_loss", log_normal_loss)
        self.log("train/total_loss", log_total_loss)

        self.log("train/rotation_acc", rotation_acc)
        self.log("train/normal_acc", normal_acc)
        
        return total_loss

    def val_test_forward(self, batch, batch_size, way, shot):
        assert batch_size == 1
        num_support_samples = way * shot
        data, _ = batch
        data = self.backbone(data)
        data = data.reshape([batch_size, -1] + list(data.shape[-3:]))
        data_support = data[:, :num_support_samples]
        data_query = data[:, num_support_samples:]
        logits = self.finetune_classifier.forward(data_query, data_support, shot)
        return logits

    def validation_step(self, batch, batch_idx):
        if self.trainer.current_epoch>=self.switch_epoch:
            _ = self.shared_step(batch, "val")

    def training_epoch_end(self, outs):
        losses = ["rotation", "total", "normal"]
        for loss_name in losses:
            value = getattr(self, f"{loss_name}_loss").compute()
            self.log(f"train/{loss_name}_loss_epoch", value)
            getattr(self, f"{loss_name}_loss").reset()
        acc_names = ["rotation", "normal"]
        for acc_name in acc_names:
            value = getattr(self, f"{acc_name}_acc").compute()
            self.log(f"train/{acc_name}_acc_epoch", value)
            getattr(self, f"{acc_name}_acc").reset()
        
    def validation_epoch_end(self, outs):
        if self.trainer.current_epoch>=self.switch_epoch:
            utils.epoch_wrapup(self, 'val')

    def set_metrics(self):
        """Set logging metrics."""
        self.mixed_up_loss = AverageMeter()
        self.rotation_loss = AverageMeter()
        self.normal_loss = AverageMeter()
        self.total_loss = AverageMeter()
        self.rotation_acc = Accuracy()
        self.normal_acc = Accuracy()

        for split in ["val", "test"]:
            setattr(self, f"{split}_loss", AverageMeter())
            setattr(self, f"{split}_acc", Accuracy())
        
        
def get_model():
    return S2M2_R
