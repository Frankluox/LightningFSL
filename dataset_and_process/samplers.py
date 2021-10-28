import torch
import numpy as np
from typing import List, Union, TypeVar, Iterator
from torch.utils.data import Sampler
import torch.distributed as dist
import math
T_co = TypeVar('T_co', covariant=True)

class CategoriesSampler(Sampler[T_co]):
    r"""Sampler that collects data into several few-shot learning tasks
    """
    def __init__(
        self, 
        labels: Union[List, 'numpy.ndarray'], 
        num_task: int, 
        way: int, 
        total_sample_per_class: int,
        total_batch_size: int = 4,
        is_DDP: bool = False,
        drop_last: bool = False
        ) -> None:
        """
         Args:
            labels: The corresponding labels of the whole dataset .
            num_task: The number of tasks within one epoch.
            way: The number of classes within one task.
            total_sample_per_class: The number of samples within each few-shot class(all samples from support and query).
            total_batch_size: The number of tasks to handle per iteration.
            is_DDP: whether in DDP mode. Parts of the code are copied from 
                torch.utils.data.distributed.DistributedSampler.
            drop_last (bool, optional): if ``True``, then the sampler will drop the
                tail of the data to make it evenly divisible across the number of
                replicas. If ``False``, the sampler will add extra indices to make
                the data evenly divisible across the replicas. Default: ``False``.
        """
        self.num_task = num_task
        self.way = way
        self.total_sample_per_class = total_sample_per_class
        self.is_DDP = is_DDP
        self.drop_last = drop_last
        self.total_batch_size = total_batch_size
        self.per_gpu_batch_size = self.total_batch_size

        if self.is_DDP:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            self.num_replicas = dist.get_world_size()
            assert self.total_batch_size % self.num_replicas == 0
            self.per_gpu_batch_size = self.total_batch_size//self.num_replicas


        if self.drop_last and self.num_task % self.total_batch_size != 0:
            self.num_iteration = math.ceil(
                (self.num_task - self.total_batch_size) / self.total_batch_size  # type: ignore
            )
        else:
            self.num_iteration = math.ceil(self.num_task / self.total_batch_size)
    

        labels = np.array(labels)#all data labels
        self.m_ind = []#the data index of each class
        for i in range(max(labels) + 1):
            ind = np.argwhere(labels == i).reshape(-1)# all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self) -> int:
        return self.num_iteration

    def __iter__(self) -> Iterator[T_co]:
        # print(self.num_iteration)
        for _ in range(self.num_iteration):
            tasks = []
            for i in range(self.per_gpu_batch_size):
                task = []
                #random sample num_class indexs,e.g. 5
                classes = torch.randperm(len(self.m_ind))[:self.way]
                # print(f"{j}: {i}: {dist.get_rank()}: {classes}")
                # print(classes)
                for c in classes:
                    #sample total_sample_per_class data index of this class
                    l = self.m_ind[c]#all data indexs of this class
                    pos = torch.randperm(len(l))[:self.total_sample_per_class] 
                    task.append(l[pos])
                tasks.append(torch.stack(task).t().reshape(-1))
            tasks = torch.stack(tasks).reshape(-1)
            yield tasks


