from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.seed import seed_everything
import torch.distributed as dist
class SetSeedCallback(Callback):
    """Set different seed for each GPU.
    """
    def __init__(self, seed = 10, is_DDP=False):
        self.seed = seed
        self.is_DDP = is_DDP
    def on_fit_start(self,trainer,pl_module):
        if self.is_DDP:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            seed_everything((dist.get_rank()+1)*self.seed)
        else:
            seed_everything(self.seed)