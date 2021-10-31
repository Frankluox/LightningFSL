from .miniImageNet import miniImageNet
import torch
import random
import pickle
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as functional

def crop_func(img, crop, ratio = 1.2):
    """
    Given cropping positios, relax for a certain ratio, and return new crops
    , along with the area ratio.
    """
    assert len(crop) == 4
    w,h = functional._get_image_size(img)
    if crop[0] == -1.:
        crop[0],crop[1],crop[2],crop[3]  = 0., 0., h, w
    else:
        crop[0] = max(0, crop[0]-crop[2]*(ratio-1)/2)
        crop[1] = max(0, crop[1]-crop[3]*(ratio-1)/2)
        crop[2] = min(ratio*crop[2], h-crop[0])
        crop[3] = min(ratio*crop[3], w-crop[1])
    return crop, crop[2]*crop[3]/(w*h)

class miniImageNetProbCrop(miniImageNet):
    """
    The dataset ultilizing Fusion Sampling used in the paper
     "Rectifying the Shortcut Learning of Background: Shared Object Concentration
     for Few-Shot Image Recognition".
    """
    def __init__(self, root, mode, feature_image_and_crop_id, position_list, ratio = 1.2, crop_size = 0.08, image_sz = 84):
        """
        feature_image_and_crop_id: The ids of crops of images found by COS algorithms.
        position_list: The position of each crop.
        ratio: The ratio to relax the cropping.
        crop_size: The least croping size of each image.
        """
        super().__init__(root, mode)
        self.image_sz = image_sz
        self.ratio = ratio
        self.crop_size = crop_size
        with open(feature_image_and_crop_id, 'rb') as f:
            self.feature_image_and_crop_id = pickle.load(f)
        self.position_list = np.load(position_list)
        self.get_id_position_map()
        self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                        np.array([0.2726, 0.2634, 0.2794]))])
    
    def get_id_position_map(self):
        self.position_map = {}
        for i, feature_image_and_crop_ids in self.feature_image_and_crop_id.items():
            for clusters in feature_image_and_crop_ids:
                for image in clusters:
                    # print(image)
                    if image[0] in self.position_map:
                        self.position_map[image[0]].append((image[1],image[2]))
                    else:
                        self.position_map[image[0]] = [(image[1],image[2])]
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = self.loader(path)
        x = random.random()
        ran_crop_prob = 1-torch.tensor(self.position_map[idx][0][1]).sum()
        if x > ran_crop_prob:
            crop_ids = self.position_map[idx][0][0]
            if ran_crop_prob <= x < ran_crop_prob+self.position_map[idx][0][1][0]:
                crop_id = crop_ids[0]
            elif ran_crop_prob+self.position_map[idx][0][1][0] <= x < ran_crop_prob+self.position_map[idx][0][1][1]+self.position_map[idx][0][1][0]:
                crop_id = crop_ids[1]
            else:
                crop_id = crop_ids[2]
            crop = self.position_list[idx][crop_id]
            crop, space_ratio  = crop_func(image, crop, ratio = self.ratio)
            image = functional.crop(image,crop[0],crop[1], crop[2],crop[3])
            image = transforms.RandomResizedCrop(self.image_sz, scale = (self.crop_size/space_ratio, 1.0))(image)
        else:
            image = transforms.RandomResizedCrop(self.image_sz)(image)
        image = self.transform(image)
        return image, target


def return_class():
    return miniImageNetProbCrop