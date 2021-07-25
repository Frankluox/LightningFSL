from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.seed import seed_everything
import torch.distributed as dist
class SetSeedCallback(Callback):
    """Set different seed for each GPU.
    """
    def __init__(self, seed = 10):
        self.seed = seed
    def on_fit_start(self,trainer,pl_module):
        if dist.is_available():
            seed_everything((dist.get_rank()+1)*self.seed)
        else:
            seed_everything(self.seed)