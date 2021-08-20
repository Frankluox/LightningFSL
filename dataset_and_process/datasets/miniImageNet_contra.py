from torchvision import transforms
import numpy as np
from .miniImageNet import miniImageNet

class miniImageNetContrastive(miniImageNet):
    r"""miniImageNet for contrastive learning
    """
    def __init__(self, root: str, mode: str) -> None:
        assert mode in ["train", "val", "test"]
        super().__init__(root, mode)
        if mode == 'val' or mode == 'test':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                        np.array([0.2726, 0.2634, 0.2794]))])
        elif mode == 'train':
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomApply([
                        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # BYOL
                    ], p=0.3),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.GaussianBlur((3, 3), (1.0, 2.0))],
                    p = 0.2),
                transforms.RandomResizedCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                    np.array([0.2726, 0.2634, 0.2794]))])

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample_1 = self.transform(sample)
        sample_2 = self.transform(sample)
        return sample_1, sample_2, target



def return_class():
    return miniImageNetContrastive

if __name__ == '__main__':
    pass