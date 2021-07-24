from torchvision import transforms
import os
import numpy as np
from torchvision.datasets import ImageFolder


class tieredImageNet(ImageFolder):
    r"""The standard tieredImageNet dataset. ::
            
       root
        |
        |---train
        |    |---n12313324234
        |    |---n00003254543
        |       ...
        |    
        |---val
        |---test

                    
    Args:
        root: Root directory path.
        mode: train or val or test
    """
    def __init__(self, root: str, mode: str):
        assert mode in ["train", "val", "test"]
        IMAGE_PATH = os.path.join(root, mode)
        if mode == 'val' or mode == 'test':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([0.4783, 0.4564, 0.4101]),
                                    np.array([0.2634, 0.2577, 0.2709]))])
        elif mode == 'train':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.4783, 0.4564, 0.4101]),
                                    np.array([0.2634, 0.2577, 0.2709]))])

        super().__init__(IMAGE_PATH, transform)
        self.label = self.targets


def return_class():
    return tieredImageNet

if __name__ == '__main__':
    pass