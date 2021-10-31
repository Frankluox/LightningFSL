from PIL import Image
import torchvision.transforms as transforms
import os
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from architectures.feature_extractor.resnet12 import ResNet12

from architectures import get_backbone
from torchvision.datasets import ImageFolder
import numbers
from collections.abc import Sequence
import pickle
import numpy as np
import torchvision.transforms.functional as functional
import math
import argparse
import utils

from torchvision.transforms import RandomResizedCrop


def COS(model, dataset_name = "miniImageNet", dataset_path = "", num_crop = 30, batch_sz = 5, alpha = 0.5, num_cluster = 5, save_dir = ".", top_k = top_k): 
    """
    The COS algorithm in the paper
    "Rectifying the Shortcut Learning of Background: Shared Object Concentration
     for Few-Shot Image Recognition".

    This should be used after the pre-training of Eeamplar.

    Produce three objects:
        all_features.npy
        position_list.npy
        feature_image_and_crop_id.pkl
    """
    feature_dim = model.outdim
    dataset = Dataset(crop_num = num_crop, dataset_name = dataset_name, dataset_path = dataset_path)
    num_batch = len(dataset)//batch_sz
    

    label_to_tensor_list = {}#label->[id, tensor_of_the_class]
    count = []#the numbaer of images per class

    for i in range(dataset.num_class):
        label_to_tensor_list[i] = [[], torch.FloatTensor(dataset.num_pic_per_class[i], num_crop+1,feature_dim).cuda()]
        count.append(0)
    print("feature generation.")
    with torch.no_grad():
        if os.path.exists(save_dir+"/all_features.npy"):
            total_features = torch.from_numpy(np.load(save_dir+"/all_features.npy")).cuda()
            labels = dataset.targets
            for i in range(total_features.size(0)):
                label_to_tensor_list[labels[i]][0].append(i)#id
                label_to_tensor_list[labels[i]][1][count[labels[i]]] = total_features[i]#features
                count[labels[i]] += 1 
            total_features = total_features.detach().cpu().numpy()
        else:
            total_features = torch.FloatTensor(len(dataset), num_crop+1,feature_dim).cuda()
            position_list = []
            for batch in range(num_batch):
                print(f"{batch}/{num_batch}")
                x, positions, labels = dataset.fetch(batch*batch_sz, (batch+1)*batch_sz-1)
                x = x.view([-1]+list(x.shape[-3:]))
                position_list.extend(positions)
                features = model(x)
                features = nn.functional.normalize(features, dim=1)
                features = nn.functional.adaptive_avg_pool2d(features,1).view(features.size(0), -1) 
                features = nn.functional.normalize(features, dim=1).view(-1, num_crop+1,features.size(1))
                total_features[batch*batch_sz:(batch+1)*batch_sz] = features
                for i in range(features.size(0)):
                    label_to_tensor_list[labels[i]][0].append(batch*batch_sz+i)
                    label_to_tensor_list[labels[i]][1][count[labels[i]]] = features[i]
                    count[labels[i]] += 1   
            total_features = total_features.detach().cpu().numpy()    
            np.save(save_dir+"/all_features.npy", total_features)
            np.save(save_dir+"/position_list.npy", np.array(position_list))
    
        for i in range(dataset.num_class):
            assert count[i] == dataset.num_pic_per_class[i]
            label_to_tensor_list[i][1] = label_to_tensor_list[i][1].view(-1, feature_dim).detach().cpu().numpy()


    feature_image_and_crop_id = {}#The ids of crops of images found by COS algorithms.
    count = {}#count the number of images in each cluster
    distance_list = []#feature distance to clusters

    print("clustering begins.")
    for i in range(dataset.num_class):
        count[i] = []
        print(f"The {i}-th class.")
        
        feature_image_and_crop_id[i] = []
        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(label_to_tensor_list[i][1])
        cluster_label = kmeans.labels_.reshape((dataset.num_pic_per_class[i], num_crop+1))
        for _ in range(num_cluster):
            count[i].append(0) 
        for k in range(num_cluster):  
            for j in range(dataset.num_pic_per_class[i]):
                if (cluster_label[j] == k).any():
                    count[i][k] += 1

        flag = False
        feature_cluster = []#selected clusters
        for j in range(num_cluster):
            if count[i][j]/dataset.num_pic_per_class[i] > alpha:
                flag = True
                feature_cluster.append(j)
        if not flag:#No clusters satisfy, then choose the best one
            count[i] = torch.Tensor(count[i])
            _, feature_cluster_ = torch.topk(count[i], 1)
            feature_cluster.append(feature_cluster_.item())
        #all distances to the clusters
        distance = kmeans.transform(label_to_tensor_list[i][1]).reshape((dataset.num_pic_per_class[i], num_crop+1, num_cluster))
        distance_list.append(distance)

        image_and_crop_ids = []
        distance_ = torch.from_numpy(distance[:,:,feature_cluster])
        min_distance, _ = torch.min(distance_, dim=2)#each crop, find the closest cluster
        min_distance, idx = torch.topk(min_distance, top_k, dim=1, largest=False)#[600,3]
        for k in range(idx.size(0)):
            idx_ = idx[k].numpy().tolist()
            image_and_crop_ids.append((label_to_tensor_list[i][0][k], idx_, min_distance[k].numpy().tolist()))
        
        feature_image_and_crop_id[i].append(image_and_crop_ids)

    #calculate sampling probabilities
    max_distance = np.concatenate(distance_list).min(axis=2).max()
    for i, image_and_crop_ids in feature_image_and_crop_id.items():
        for l in image_and_crop_ids:
            for j in l:
                sum_ = 0.
                max_ = 0.
                for p,k in enumerate(j[2]):
                    j[2][p] = 1-k/max_distance
                    if max_ < j[2][p]:
                        max_ = j[2][p]
                    sum_ += j[2][p]
                for p,k in enumerate(j[2]):
                    j[2][p] = max_*k/sum_
                
    with open(save_dir+"/feature_image_and_crop_id.pkl", 'wb') as f:
        pickle.dump(feature_image_and_crop_id, f, pickle.HIGHEST_PROTOCOL)

class RandomResizedCrop_revise(RandomResizedCrop):
    """
    Modified from torchvision, return positions of cropping boxes
    """
    def __init__(self, size):
        super().__init__(size = size)
    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return functional.resized_crop(img, i, j, h, w, self.size, self.interpolation), (i,j,h,w)

class Dataset(ImageFolder):
    def __init__(self, dataset_name = "miniImageNet", dataset_path = "", image_sz = 84, crop_way = 'sampling', crop_num = 30):
        super().__init__(dataset_path + "/train")
        assert dataset_name in ["miniImageNet", "tieredImageNet"]
        if dataset_name == "miniImageNet":
            normalize = transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                    np.array([0.2726, 0.2634, 0.2794]))
        else:
            normalize = transforms.Normalize(np.array([0.4783, 0.4564, 0.4101]),
                                    np.array([0.2634, 0.2577, 0.2709]))
        self.crop_way = crop_way
        self.crop_num = crop_num
        self.num_class = len(self.classes)
        self.num_pic_per_class = {}
        for target_class, class_index in self.class_to_idx.items():
            # print(target_class)
            target_dir = os.path.join(dataset_path, "train", target_class)
            if not os.path.isdir(target_dir):
                continue
            for _, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                self.num_pic_per_class[class_index] = len(fnames)
        

        self.crop_func = RandomResizedCrop_revise(image_sz)
        self.transform = transforms.Compose([
            transforms.Resize([image_sz,image_sz]),
            transforms.ToTensor(),
            normalize
        ])
        

    def __getitem__(self, index):
        path, label= self.samples[index]
        image=Image.open(path).convert('RGB')
        patch_list=[]
        positions= []
        patch_list.append(self.transform(image))
        positions.append([-1.,-1.,-1.,-1.])
        for num_patch in range(self.crop_num):
            image_, position = self.crop_func(image)
            positions.append(position)
            patch_list.append(self.transform(image_))
        patch_list=torch.stack(patch_list,dim=0)
        return patch_list, positions, label


    def fetch(self, start: int, end: int):
        """
        obtain data from index 'start' to 'end'.
        """
        position_list = []
        label_list = []
        tensor = torch.FloatTensor(end-start+1, self.crop_num+1, 3, 84, 84).cuda()
        for i in range(start, end+1):
            x, positions, label = self[i]
            tensor[i-start] = x.cuda()
            label_list.append(label)
            position_list.append(positions)
        return tensor, position_list, label_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pretrained_Exemplar_path', 
        type = str, required=True, 
        help='The path of the pretrained Exemplar model'
                        )
    parser.add_argument(
        '--backbone_name', 
        type = str, default="resnet12", 
        help='The name of the backbone.'
                        )
    parser.add_argument('--dataset_path', 
                        type = str, 
                        required=True, 
                        help='The path of the dataset.'
                        )
    parser.add_argument('--dataset_name', 
                        type = str, 
                        default="miniImageNet", 
                        help='The name of the dataset; miniImageNet or tieredImageNet.'
                        )
    parser.add_argument('--num_crop', 
                        type = int, 
                        default=30,
                        help='The number of patches per image.'
                        )
    parser.add_argument('--top_k', 
                        type = int, 
                        default=3,
                        help='The number of selected patches per image.'
                        )
    parser.add_argument('--threshold', 
                        type = float,
                        default=0.5,
                        help='The threshold of preserving clusters.'
                         )
    parser.add_argument('--num_cluster', 
                        type = int,
                        default=5,
                        help='The number of clusters per class.'
                         )
    parser.add_argument('--save_dir', 
                        type = str,
                        default=".",
                        help='The directory to save patches.'
                         )                                         
    args = parser.parse_args()
    model = get_backbone(args.backbone_name).cuda()
    state = torch.load(args.pretrained_Exemplar_path)["state_dict"]
    state = utils.preserve_key(state, "backbone")
    model.load_state_dict(state)
    model.eval()
    COS(model, 
        dataset_name = args.dataset_name,
        dataset_path = args.dataset_path, 
        num_crop = args.num_crop,
        alpha = args.threshold,
        num_cluster = args.num_cluster,
        save_dir = args.save_dir,
        top_k = top_k)