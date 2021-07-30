from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from typing import Optional, List
from pytorch_lightning.utilities import rank_zero_deprecation, rank_zero_warn
import os
class ModifiedModelCheckpoint(ModelCheckpoint):
    """Modify ModelCheckpoint, making it not a fault when the monitored validation keys do not exist.
        And make it possible to specify epochs to save.
    """
    def __init__(self, save_epochs: Optional[List] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.save_epochs = save_epochs

    def save_checkpoint(self, trainer: 'pl.Trainer', unused: Optional['pl.LightningModule'] = None) -> None:
        """
        Performs the main logic around saving a checkpoint. This method runs on all ranks.
        It is the responsibility of `trainer.save_checkpoint` to correctly handle the behaviour in distributed training,
        i.e., saving only on rank 0 for data parallel use cases.
        """
        if unused is not None:
            rank_zero_deprecation(
                "`ModelCheckpoint.save_checkpoint` signature has changed in v1.3. The `pl_module` parameter"
                " has been removed. Support for the old signature will be removed in v1.5"
            )

        epoch = trainer.current_epoch
        global_step = trainer.global_step

        self._add_backward_monitor_support(trainer)
        exist_key = self._validate_monitor_key(trainer)

        # track epoch when ckpt was last checked
        self._last_global_step_saved = global_step

        # what can be monitored
        monitor_candidates = self._monitor_candidates(trainer, epoch=epoch, step=global_step)

        # callback supports multiple simultaneous modes
        # here we call each mode sequentially
        # Mode 1: save the top k checkpoints
        if exist_key:
            self._save_top_k_checkpoint(trainer, monitor_candidates)
        # Mode 2: save monitor=None checkpoints
        self._save_none_monitor_checkpoint(trainer, monitor_candidates)
        # Mode 3: save last checkpoints
        self._save_last_checkpoint(trainer, monitor_candidates)

        if self.save_epochs is not None and epoch in self.save_epochs:
            save_path = os.path.join(self.dirpath, f"{epoch}th_epoch.ckpt")
            self._save_model(trainer, save_path)

        
    def _validate_monitor_key(self, trainer: 'pl.Trainer') -> None:
        metrics = trainer.logger_connector.callback_metrics

        # validate metric
        if self.monitor is not None and not self._is_valid_monitor_key(metrics):
            m = (
                f"ModelCheckpoint(monitor='{self.monitor}') not found in the returned metrics:"
                f" {list(metrics.keys())}. "
                f"HINT: Did you call self.log('{self.monitor}', value) in the LightningModule?"
            )
            rank_zero_warn(m)
            return False
        return True