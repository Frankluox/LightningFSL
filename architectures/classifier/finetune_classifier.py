
import torch
from architectures import get_classifier
import torch.nn.functional as F
from torch import Tensor
import numpy as np
class Finetuner():
    def __init__(self, ft_batchsize, ft_epochs,ft_lr, ft_wd, way, ft_classifier_name, ft_classifier_params):
        self.support_labels = torch.arange(way, dtype=torch.int8).type(torch.LongTensor)
        self.ft_batchsize = ft_batchsize
        self.ft_epochs = ft_epochs
        self.ft_lr = ft_lr
        self.ft_wd = ft_wd
        self.ft_classifier_name = ft_classifier_name
        self.ft_classifier_params = ft_classifier_params
    def forward(self, features_test: Tensor, features_train: Tensor, 
                shot: int) -> Tensor:
        r"""Take one task of few-shot training examples and testing examples as input,
            output the logits of each testing examples.

        Args:
            features_test: Testing examples. size: [1, num_query, c, h, w]
            features_train: Training examples which has labels like:[abcdabcdabcd].
                            size: [1, way*shot, c, h, w]
            shot: The number of training images per class in each few-shot classification
                  task.
        Output:
            classification_scores: The calculated logits of testing examples.
                                   size: [1, num_query, way]
        """
        assert features_train.size(0) == features_test.size(0) == 1
        features_train = torch.squeeze(features_train)
        features_test = torch.squeeze(features_test)


        support_size = features_train.size(0)
        support_labels = self.support_labels.repeat(shot).to(features_test.device)
        finetune_classifier = get_classifier(self.ft_classifier_name, **self.ft_classifier_params).to(features_test.device)
        set_optimizer = torch.optim.SGD(finetune_classifier.parameters(), lr = self.ft_lr, momentum=0.9, dampening=0.9, weight_decay=self.ft_wd)
        with torch.enable_grad():
            for epoch in range(self.ft_epochs+1):
                rand_id = np.random.permutation(support_size)
                for i in range(0, support_size , self.ft_batchsize):
                    set_optimizer.zero_grad()
                    selected_id = torch.from_numpy(rand_id[i: min(i+self.ft_batchsize, support_size)])
                    train_batch = features_train[selected_id]
                    label_batch = support_labels[selected_id] 
                    scores = finetune_classifier(train_batch)
                    loss = F.cross_entropy(scores, label_batch)
                    loss.backward()
                    set_optimizer.step()
        classification_scores = finetune_classifier(features_test).unsqueeze_(0)             
        return classification_scores

