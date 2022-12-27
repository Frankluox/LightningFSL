from pytorch_lightning import LightningDataModule
from dataset_and_process.datasets import get_dataset
from dataset_and_process.samplers import CategoriesSampler
from typing import Optional, Dict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
class FewShotDataModule(LightningDataModule):
    r"""A general datamodule for few-shot image classification.

    Args:
        train_dataset_name: The name of the training dataset construction class.
        val_test_dataset_name: The name of the val/test dataset construction class.
        train_data_root: Root directory path of train data.
        val_test_data_root: Root directory path of val/test data.
        is_meta: whether implementing meta-learning during training.
        train_batchsize: The batch size of training.
        val_batchsize: The batch size of validation.
        test_batchsize: The batch size of testing.
        train_num_workers: The number of workers during training.
        val_num_workers: The number of workers during validation and testing.
        is_DDP: Whether launch DDP mode for multi-GPU training.
        train_num_task_per_epoch: The number of tasks per epoch during training.
        val_num_task: The number of tasks during one validation loop.
        test_num_task: The number of tasks during one tesing loop.
        way: The number of classes within one task.
        train_shot: The number of samples within each few-shot support class during training.
        val_shot: The number of samples within each few-shot support class during validation.
        test_shot: The number of samples within each few-shot support class during testing.
        num_query: The number of samples within each few-shot query class.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the tasks to make it evenly divisible across the number of
            DDP replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
        num_gpus: The number of gpus.

    .. warning::
        Remember to turn off ``replace_sampler_ddp'' in the Trainer, otherwise the 
        samplers will not be set.
    """
    def __init__(
        self,
        train_dataset_name: str = "miniImageNet",
        val_test_dataset_name: str = "miniImageNet",
        train_data_root: str = '',
        val_test_data_root: str = '',
        is_meta: bool = False,
        train_batchsize: int = 32,
        val_batchsize: int = 4,
        test_batchsize: int = 4,
        train_num_workers: int = 8,
        val_num_workers: int = 8,
        is_DDP: bool = False,
        train_num_task_per_epoch: Optional[int] = 1000,
        val_num_task: int = 600,
        test_num_task: int = 2000,
        train_way: Optional[int] = 5,
        val_way: int = 5,
        test_way: int = 5,
        train_shot: Optional[int] = 5,
        val_shot: int = 5,
        test_shot: int = 5,
        num_query: int = 15,
        drop_last: Optional[bool] = None,
        num_gpus: int = 1,
        train_dataset_params: Dict = {},
        val_test_dataset_params: Dict = {},
    ) -> None:
        super().__init__()
        self.train_data_root = train_data_root
        self.val_test_data_root = val_test_data_root
        self.train_dataset_name = train_dataset_name
        self.val_test_dataset_name = val_test_dataset_name
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.is_DDP = is_DDP
        self.train_batch_size = train_batchsize
        self.val_batch_size = val_batchsize
        self.test_batch_size = test_batchsize
        self.is_meta = is_meta
        self.train_sampler = None
        self.train_batch_sampler = None
        self.train_num_task_per_epoch = train_num_task_per_epoch
        self.val_num_task = val_num_task
        self.test_num_task = test_num_task
        self.train_way = train_way,
        self.val_way = val_way,
        self.test_way = test_way,
        self.train_shot = train_shot
        self.val_shot = val_shot
        self.test_shot = test_shot
        self.num_query = num_query
        self.drop_last = drop_last
        self.num_gpus = num_gpus
        self.train_dataset_params = train_dataset_params
        self.val_test_dataset_params = val_test_dataset_params
        
    @property
    def train_dataset_cls(self):
        """Obtain the dataset class
        """
        return get_dataset(self.train_dataset_name)
    
    @property
    def val_test_dataset_cls(self):
        """Obtain the dataset class
        """
        return get_dataset(self.val_test_dataset_name)

    
    def set_train_dataset(self):
        self.train_dataset = self.train_dataset_cls(
            self.train_data_root,
            mode="train",
            **self.train_dataset_params
        )
    def set_val_dataset(self):
        self.val_dataset = self.val_test_dataset_cls(
            self.val_test_data_root,
            mode="val",
            **self.val_test_dataset_params
        )
    def set_test_dataset(self):
        self.test_dataset = self.val_test_dataset_cls(
            self.val_test_data_root,
            mode="test",
            **self.val_test_dataset_params
        )
    def set_sampler(self):
        """Set sampler for training, validation and testing.
           task-level batches need batch sampler.
        """
        if self.is_meta:
            self.train_batch_sampler = CategoriesSampler(
                self.train_dataset.label, self.train_num_task_per_epoch,
                self.train_way, self.train_shot+self.num_query, self.train_batch_size, 
                self.is_DDP, self.drop_last
                )
        elif self.is_DDP:
            self.train_sampler = DistributedSampler(self.train_dataset)

        self.val_batch_sampler = CategoriesSampler(
            self.val_dataset.label, self.val_num_task,
            self.val_way, self.val_shot+self.num_query, self.val_batch_size, 
            self.is_DDP, self.drop_last
            )

        self.test_batch_sampler = CategoriesSampler(
            self.test_dataset.label, self.test_num_task,
            self.test_way, self.test_shot+self.num_query, self.test_batch_size, 
            self.is_DDP, self.drop_last
            )
        
    def setup(self, stage):
        self.set_train_dataset()
        self.set_val_dataset()
        self.set_test_dataset()
        self.set_sampler()

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size = 1 if self.train_batch_sampler is not None\
                         else self.train_batch_size//self.num_gpus,
            shuffle = False if self.train_batch_sampler is not None\
                      or self.train_sampler is not None else True,
            num_workers = self.train_num_workers,
            batch_sampler = self.train_batch_sampler,
            sampler = self.train_sampler,
            pin_memory = True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            shuffle = False,
            num_workers = self.val_num_workers,
            batch_sampler = self.val_batch_sampler,
            pin_memory = True
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            shuffle = False,
            num_workers = self.val_num_workers,
            batch_sampler = self.test_batch_sampler,
            pin_memory = True
        )
        return loader

if __name__ == '__main__':
    a = FewShotDataModule()
    # a.set_train_dataset()
    # print(a.train_dataloader().sampler)
