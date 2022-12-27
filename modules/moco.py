from .base_module import BaseFewShotModule
import torch.nn as nn
from typing import Tuple, List, Optional, Union, Dict
import torch.nn.functional as F
from architectures import get_classifier, get_backbone
import torch
class MoCo(BaseFewShotModule):
    r"""The well-known contrastive mode---MoCo (Momentum Contrast for Unsupervised Visual Representation Learning, CVPR 2020)
        , as well as its variant---Exampler (What makes instance discrimination good for transfer learning? ICLR 2021). 
        Surprisingly, pure contrastive-pretrained model performs very well on Few-Shot Learning.
    """
    def __init__(
        self,
        momentum: float = 0.999,
        temparature: float = 0.07,
        queue_len: int = 65536,
        mlp_dim: int = 128, 
        task_classifier_name: str = "proto_head",
        task_classifier_params: Dict = {"learn_scale":False},
        is_Exampler: bool = True,
        is_DDP: bool = True,
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
            momentum: The momentum hyperparameter in MoCo, used for updating the momentum encoder.
            temparature: The temparature hyperparameter in MoCo, used to balance the hardness of negative samples.
            queue_len: The length of the queue containing all negative samples.
            mlp_dim: The dimension of the MLP head after the backbone network.
            task_classifier_name: The name of the classifier for downstream (val, test)
                                  few-shot tasks. It should match the name of file that 
                                  contains the classifier class.
            task_classifier_params: The initial parameters of the classifier for 
                                    downstream (val, test) few-shot tasks.
            is_Exampler: Whether to use Exampler variant of MoCo.
            is_DDP: Whether in DDP mode for multi-GPU training.
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
        self.save_hyperparameters()
        self.backbone_m = get_backbone(backbone_name, **backbone_kwargs)
        for param, param_m in zip(self.backbone.parameters(), self.backbone_m.parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.add_nonlinear = nn.Sequential(
            nn.Linear(self.backbone.outdim, self.backbone.outdim), nn.ReLU(), nn.Linear(self.backbone.outdim, mlp_dim)
        )
        self.add_nonlinear_m = nn.Sequential(
            nn.Linear(self.backbone.outdim, self.backbone.outdim), nn.ReLU(), nn.Linear(self.backbone.outdim, mlp_dim)
        )
        for param, param_m in zip(self.add_nonlinear.parameters(), self.add_nonlinear_m.parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.register_buffer("queue", torch.randn(mlp_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if is_Exampler:
            self.register_buffer("labels", torch.zeros(queue_len, dtype=torch.long))
            self.labels -= 1

        self.val_test_classifier = get_classifier(task_classifier_name, **task_classifier_params)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param, param_m in zip(self.backbone.parameters(), self.backbone_m.parameters()):
            param_m.data = param_m.data * self.hparams.momentum + param.data * (1. - self.hparams.momentum)
        for param, param_m in zip(self.add_nonlinear.parameters(), self.add_nonlinear_m.parameters()):
            param_m.data = param_m.data * self.hparams.momentum + param.data * (1. - self.hparams.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels=None):
        # gather keys before updating queue
        if self.hparams.is_DDP:
            keys = concat_all_gather(keys)
            if self.hparams.is_Exampler:
                labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # print(batch_size)
        assert self.hparams.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        if self.hparams.is_Exampler:
            self.labels[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % self.hparams.queue_len  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle(self, x):
        batch_size = x.shape[0]
        idx_shuffle = torch.randperm(batch_size).cuda()#Returns a 1D tensor of random permutation of integers from 0 to n - 1.
        idx_unshuffle = torch.argsort(idx_shuffle)#Returns the indices that sort a tensor along a given dimension in ascending order by value.
        x = x[idx_shuffle]
        return x, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle(self, x, idx_unshuffle):
        return x[idx_unshuffle]

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def train_forward(self, batch):
        im_one, im_two, labels = batch
        if not self.hparams.is_DDP:
            batch_size = im_one.size(0)
            q_1, q_2 = torch.split(im_one, batch_size//2)
            q_1 = self.backbone(q_1)
            q_2 = self.backbone(q_2)
            q = torch.cat((q_1,q_2), dim = 0)
        # compute query features
        else:
            q = self.backbone(im_one)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)
        # print(q.shape)
        q = nn.functional.adaptive_avg_pool2d(q,1).view(q.size(0), -1)   
        # print(q.shape)     
        q = self.add_nonlinear(q)
        q = nn.functional.normalize(q, dim=1) 

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            if not self.hparams.is_DDP:
                im_two, idx_unshuffle = self._batch_shuffle(im_two)
                im_two_1, im_two_2 = torch.split(im_two, batch_size//2)
                k_1 = self.backbone_m(im_two_1)
                k_2 = self.backbone_m(im_two_2)
                k = torch.cat((k_1,k_2), dim = 0)
                k = nn.functional.normalize(k, dim=1)
                k = nn.functional.adaptive_avg_pool2d(k,1).view(k.size(0), -1)        
                k = self.add_nonlinear_m(k)
                k = nn.functional.normalize(k, dim=1)
                k = self._batch_unshuffle(k, idx_unshuffle)
            else:
                im_two, idx_unshuffle = self._batch_shuffle_ddp(im_two)
                k = self.backbone_m(im_two)  # keys: NxC
                k = nn.functional.normalize(k, dim=1)
                k = nn.functional.adaptive_avg_pool2d(k,1).view(k.size(0), -1)        
                k = self.add_nonlinear_m(k)
                k = nn.functional.normalize(k, dim=1)
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            # undo shuffle
        #n:batch_size,c:dim,k:queue_length    
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        if self.hparams.is_Exampler:
            # print(lables.shape)
            labels_tmp = torch.unsqueeze(labels, -1).repeat(1,self.labels.size(0))
            label_queue =torch.unsqueeze(self.labels, 0).repeat(labels_tmp.size(0), 1)
            heatmap = (labels_tmp==label_queue).long()*(-300)
            l_neg += heatmap
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.hparams.temparature
        target_label = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        self._dequeue_and_enqueue(k, labels)
        return logits, target_label

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



@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)#把每个gpu的tensor装进list中

    output = torch.cat(tensors_gather, dim=0)
    return output

def get_model():
    return MoCo
    
