from pytorch_lightning import LightningModule
from . import utils
from architectures import get_backbone
from typing import Tuple, List, Optional, Union, Dict
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AverageMeter
class BaseFewShotModule(LightningModule):
    r"""Template for all few-shot learning models.
    """
    def __init__(
        self,
        backbone_name: str = "resnet12",
        train_way: Optional[int] = 5,
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
        backbone_kwargs: Dict = {}
    ) -> None:
        """
        Args:
            backbone_name: The name of the feature extractor, 
                        which should match the correspond 
                        file name in architectures.feature_extractor
            train_way: The number of classes within one training task.
            val_way: The number of classes within one val task.
            test_way: The number of classes within one testing task.
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
        super().__init__()
        self.save_hyperparameters()
        self.backbone = get_backbone(backbone_name, **backbone_kwargs)
        for mode in ["train","val", "test"]:
            way = getattr(self.hparams, f"{mode}_way")
            setattr(self, f"{mode}_label", torch.arange(way, dtype=torch.int8).repeat(num_query).type(torch.LongTensor).reshape(-1))
        self.set_metrics()

    def train_forward(self, batch):
        r"""Here implements the forward function of training.

        Output: logits
        Args: (can be dynamically adjusted)
            batch: a batch from train_dataloader.
        """
        raise NotImplementedError

    def val_test_forward(self, batch, batch_size, way, shot):
        r"""Here implements the forward function of validation and testing.

        Output: logits
        Args: (can be dynamically adjusted)
            batch: a batch from val_dataloader.
            batch_size: number of tasks during one iteration.
            way: The number of classes within one task.
            shot: The number of samples within each few-shot support class. 
        """
        raise NotImplementedError
    
    def shared_step(self, batch, mode):
        r"""The shared operation across
            validation, testing and potentially training (meta-learning).
            
        Args:
            batch: a batch from val_dataloader.
            mode: train, val or test
        """
        assert mode in ["train", "val", "test"]
        if mode == "train":
            flag = "train"
        else:
            flag = "val_test"
        foward_function = getattr(self, f"{flag}_forward")
        batch_size_per_gpu = getattr(self.hparams, f"{mode}_batch_size_per_gpu")
        shot = getattr(self.hparams, f"{mode}_shot")

        way = getattr(self.hparams, f"{mode}_way")
        logits = foward_function(batch, batch_size_per_gpu,way, shot)
        label = getattr(self, f"{mode}_label")
        label = torch.unsqueeze(self.label, 0).repeat(batch_size_per_gpu, 1).reshape(-1).to(logits.device)
        logits = logits.reshape(label.size(0),-1)
        
        loss = F.cross_entropy(logits, label)
        log_loss = getattr(self, f"{mode}_loss")(loss)
        accuracy = getattr(self, f"{mode}_acc")(logits, label)
        self.log(f"{mode}/loss", log_loss)
        self.log(f"{mode}/acc", accuracy)
        return loss

    def training_step(self, batch, batch_idx):
        if self.hparams.train_shot == None or self.hparams.train_batch_size_per_gpu == None:
            raise RuntimeError("train_shot or train_batch_size not specified.\
                                Please implement your own training step if the\
                                 training is not meta-learning.")
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        _ = self.shared_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        _ = self.shared_step(batch, "test")

    def training_epoch_end(self, outs):
        utils.epoch_wrapup(self, 'train')
    
    def validation_epoch_end(self, outs):
        utils.epoch_wrapup(self, 'val')

    def test_epoch_end(self, outs):
        utils.epoch_wrapup(self, 'test')

    def configure_optimizers(self):
        return utils.set_schedule(self)

    def set_metrics(self):
        r"""Set basic logging metrics for few-shot learning.
        """
        for split in ["train", "val", "test"]:
            setattr(self, f"{split}_loss", AverageMeter())
            setattr(self, f"{split}_acc", Accuracy())
    

