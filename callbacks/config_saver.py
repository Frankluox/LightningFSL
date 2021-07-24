from pytorch_lightning.utilities.cli import SaveConfigCallback, LightningArgumentParser
from typing import Dict, Union, Any
from argparse import Namespace
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.core.lightning import LightningModule
import os
class RefinedSaverCallback(SaveConfigCallback):
    def __init__(
        self,
        parser: LightningArgumentParser,
        config: Union[Namespace, Dict[str, Any]],
        config_filename: str = 'config.yaml'
    ) -> None:
        super().__init__(parser, config, config_filename)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        log_dir = trainer.log_dir or trainer.default_root_dir
        # print(trainer.log_dir)
        config_path = os.path.join(log_dir, self.config_filename)
        self.parser.save(self.config, config_path, skip_none=False, overwrite=True, skip_check=True)