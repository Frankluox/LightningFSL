from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.optim import Adam, SGD



def epoch_wrapup(pl_module: LightningModule, mode: str):
    r"""On the end of each epoch, log information of the whole
        epoch and reset all metrics.
    
    Args:
        pl_module: An instance of LightningModule.
        mode: The current mode (train, val or test).
    """
    assert mode in ["train", "val", "test"]
    value = getattr(pl_module, f"{mode}_loss").compute()
    if mode == 'train':
        pl_module.log(f"{mode}/loss_epoch", value)
    getattr(pl_module, f"{mode}_loss").reset()
    value = getattr(pl_module, f"{mode}_acc").compute()
    if mode == 'train':
        pl_module.log(f"{mode}/acc_epoch", value)
    getattr(pl_module, f"{mode}_acc").reset()

def set_schedule(pl_module):
    r"""Set the optimizer and scheduler for training.

    Supported optimizer:
        Adam and SGD
    Supported scheduler:
        cosine scheduler and decaying on specified epochs

    Args:
        pl_module: An instance of LightningModule.
    """
    lr = pl_module.hparams.lr
    wd = pl_module.hparams.weight_decay
    decay_scheduler = pl_module.hparams.decay_scheduler
    optim_type = pl_module.hparams.optim_type


    if optim_type == "adam":
        optimizer = Adam(pl_module.parameters(),
                                    weight_decay=wd, lr=lr)
    elif optim_type == "sgd":
        optimizer = SGD(pl_module.parameters(),
                                    momentum=0.9, nesterov=True,
                                    weight_decay=wd, lr=lr)
    else:
        raise RuntimeError("optim_type not supported.\
                            Try to implement your own optimizer.")
    
    if decay_scheduler == "cosine":
        if pl_module.trainer.max_steps is None:
            if pl_module.trainer.datamodule.is_meta or\
             not pl_module.trainer.datamodule.is_DDP:
                length_epoch = len(pl_module.trainer.datamodule.train_dataloader())
            else:
                length_epoch = len(pl_module.trainer.datamodule.train_dataloader().sampler)
            max_steps = length_epoch * pl_module.trainer.max_epochs
        else:
            max_steps = pl_module.trainer.max_steps
        scheduler = {'scheduler': CosineAnnealingLR(optimizer,max_steps),
                     'interval': 'step'}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    elif decay_scheduler == "specified_epochs":
        decay_epochs = pl_module.hparams.decay_epochs
        decay_power = pl_module.hparams.decay_power
        assert decay_epochs is not None and decay_power is not None
        scheduler = {'scheduler': 
                     MultiStepLR(optimizer, milestones=decay_epochs, gamma=decay_power),
                     'interval': 'epoch'}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    elif decay_scheduler is None:
        return optimizer
    else:
        raise RuntimeError("decay scheduler not supported.\
                            Try to implement your own scheduler.")

    
        