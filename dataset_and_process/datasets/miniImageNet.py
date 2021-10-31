from torchvision import transforms
import os
import numpy as np
from torchvision.datasets import ImageFolder

class miniImageNet(ImageFolder):
    r"""The standard  dataset for miniImageNet. ::
         
        root
        |
        |
        |---train
        |    |--n01532829
        |    |   |--n0153282900000005.jpg
        |    |   |--n0153282900000006.jpg
        |    |              .
        |    |              .
        |    |--n01558993
        |        .
        |        .
        |---val
        |---test  
    Args:
        root: Root directory path.
        mode: train or val or test
    """
    def __init__(self, root: str, mode: str, image_sz = 84) -> None:
        assert mode in ["train", "val", "test"]
        IMAGE_PATH = os.path.join(root, mode)
        if mode == 'val' or mode == 'test':
            transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_sz),

                transforms.ToTensor(),
                transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                        np.array([0.2726, 0.2634, 0.2794]))])
        elif mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(image_sz),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                        np.array([0.2726, 0.2634, 0.2794]))])
        super().__init__(IMAGE_PATH, transform)
        self.label = self.targets



def return_class():
    return miniImageNet

if __name__ == '__main__':
    a = miniImageNet("../../../BF3S-master/data/mini_imagenet_split/images", "test")
