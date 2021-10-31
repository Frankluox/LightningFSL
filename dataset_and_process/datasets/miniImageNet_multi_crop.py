from .miniImageNet import miniImageNet
import torch
from torchvision import transforms
import numpy as np
class miniImageNetMultiCrop(miniImageNet):
    def __init__(self, root, mode, num_patch=9, image_sz=84):
        super().__init__(root, mode)
        self.num_patch=num_patch
        self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_sz),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                        np.array([0.2726, 0.2634, 0.2794]))])
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        patch_list = []
        for _ in range(self.num_patch):
            patch_list.append(self.transform(sample))
        patch_list=torch.stack(patch_list,dim=0)
        return patch_list, target

def return_class():
    return miniImageNetMultiCrop