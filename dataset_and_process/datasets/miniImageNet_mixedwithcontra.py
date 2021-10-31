from torchvision import transforms
import numpy as np
from .miniImageNet import miniImageNet

class miniImageNetMixedwithContrastive(miniImageNet):
    r"""Dataset for contrastive learning mixed with normal training or few-shot learning.
    """
    def __init__(self, root: str, mode: str, image_sz = 84) -> None:
        super().__init__(root, mode)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_sz),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                    np.array([0.2726, 0.2634, 0.2794]))])
        self.transform_con = transforms.Compose([
            transforms.RandomApply([
                    transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # BYOL
                ], p=0.3),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.GaussianBlur((3, 3), (1.0, 2.0))],
                p = 0.2),
            transforms.RandomResizedCrop(image_sz),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                np.array([0.2726, 0.2634, 0.2794]))])
        

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample_normal = self.transform(sample)
        sample_1 = self.transform_con(sample)
        sample_2 = self.transform_con(sample)
        return sample_normal, sample_1, sample_2, target



def return_class():
    return miniImageNetMixedwithContrastive

if __name__ == '__main__':
    pass