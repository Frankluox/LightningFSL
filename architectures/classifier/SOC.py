import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans

class SOC(nn.Module):
    def __init__(self, num_patch, alpha, beta):
        super().__init__()
        self.num_patch = num_patch
        self.alpha = alpha
        self.beta = beta

    def forward(self, feature_extractor, data, way, shot, batch_size):
        
        # print(shot)
        # print(data.shape)
        num_support_samples = way * shot
        num_patch = data.size(1)
        data = data.reshape([-1]+list(data.shape[-3:]))
        data = feature_extractor(data)
        data = nn.functional.normalize(data, dim=1)
        data = F.adaptive_avg_pool2d(data, 1)
        data = data.reshape([batch_size, -1, num_patch] + list(data.shape[-3:]))
        data = data.permute(0, 1, 3, 2, 4, 5).squeeze(-1)
        features_train = data[:, :num_support_samples]
        features_test = data[:, num_support_samples:]
        #features_train:[B,M,c,h,w]
        #features_test:[B,N,c,h,w]
        M = features_train.shape[1]
        N = features_test.shape[1]
        c = features_train.size(2)
        b = features_train.size(0)
        features_train=F.normalize(features_train, p=2, dim=2, eps=1e-12)
        features_test=F.normalize(features_test, p=2, dim=2, eps=1e-12)
        features_train = features_train.reshape(list(features_train.shape[:3])+[-1])
        num = features_train.size(3)
        patch_num = self.num_patch
        if shot == 1:
            features_focus = features_train     
        else:
            # with torch.no_grad():
            features_focus = []
            #[B,way,shot,c,h*w]
            features_train = features_train.reshape([b,shot,way]+list(features_train.shape[2:]))
            features_train = torch.transpose(features_train,1,2)
            count = 1.
            for l in range(patch_num-1):
                features_train_ = list(torch.split(features_train, 1, dim=2))
                for i in range(shot):
                    features_train_[i] = features_train_[i].squeeze(2)#[B,way,c,h*w]
                    repeat_dim = [1,1,1]
                    for j in range(i):
                        features_train_[i] = features_train_[i].unsqueeze(3)
                        repeat_dim.append(num)
                    repeat_dim.append(1)
                    for j in range(shot-i-1):
                        features_train_[i] = features_train_[i].unsqueeze(-1)
                        repeat_dim.append(num)
                    features_train_[i] = features_train_[i].repeat(repeat_dim)#[B,way,c,(h*w)^shot]
                features_train_ = torch.stack(features_train_, dim=shot+3)#[B,way,c,(h*w)^shot,shot]
                repeat_dim = []
                for _ in range(shot+4):
                    repeat_dim.append(1)
                repeat_dim.append(shot)
                features_train_ = features_train_.unsqueeze(-1).repeat(repeat_dim)
                features_train_ = (features_train_*torch.transpose(features_train_,shot+3,shot+4)).sum(2)
                features_train_ = features_train_.reshape(b,way,-1,shot,shot)
                for i in range(shot):
                    features_train_[:,:,:,i,i] = 0
                sim = features_train_.sum(-1).sum(-1)#[b,way,(h*w)^shot]
                _, idx = torch.max(sim, dim=2)
                best_idx = torch.LongTensor(b,way,shot).cuda()#The closest feature id of each image
                for i in range(shot):
                    best_idx[:,:,shot-i-1] = idx%num
                    idx = idx // num
                #feature_train:[B,way,shot,c,num]
                feature_train_ = features_train.reshape(-1,c,num)
                best_idx_ = best_idx.reshape(-1)
                b_index = torch.LongTensor(range(b*way*shot)).unsqueeze(1).repeat(1,c).unsqueeze(-1).cuda()
                c_index = torch.LongTensor(range(c)).unsqueeze(0).repeat(b*way*shot,1).unsqueeze(-1).cuda()
                num_index = best_idx_.unsqueeze(-1).repeat(1,c).unsqueeze(-1)
                feature_pick = feature_train_[(b_index,c_index,num_index)].squeeze().reshape(b,way,shot,c)#[b,way,shot,c]
                feature_avg = torch.mean(feature_pick,dim=2)#[b,way,c]
                feature_avg = F.normalize(feature_avg, p=2, dim=2, eps=1e-12)
                features_focus.append(count*feature_avg)
                count *= self.alpha
                temp = torch.FloatTensor(b,way,shot,c, num-1).cuda()
                for q in range(b):
                    for w in range(way):
                        for r in range(shot):
                            temp[q,w,r, :, :] = features_train[q,w,r, :, torch.arange(num)!=best_idx[q,w,r].item()]
                features_train = temp
                num = num-1
            features_train = torch.mean(features_train.squeeze(-1),dim=2)
            features_train = F.normalize(features_train, p=2, dim=2, eps=1e-12)
            features_focus.append(count*feature_avg)
            features_focus = torch.stack(features_focus, dim=3)#[b,way,c,num]

                            



        M = way
        features_focus = features_focus.unsqueeze(2)
        features_test= features_test.unsqueeze(1)
        features_test = features_test.reshape(list(features_test.shape[:4])+[-1])
        features_focus = features_focus.repeat(1, 1, N, 1, 1)
        features_test = features_test.repeat(1, M, 1, 1, 1)
        sim = torch.einsum('bmnch,bmncw->bmnhw', features_focus, features_test)
        combination = []
        count = 1.0
        for i in range(patch_num-1):
            combination_,idx_1 = torch.max(sim, dim = 3)
            combination_,idx_2 = torch.max(combination_, dim = 3)#[b,M,N]
            combination.append(F.relu(combination_)*count)
            count *= self.beta
            temp = torch.FloatTensor(b, M, N, sim.size(3)-1, sim.size(4)-1).cuda()
            for q in range(b):
                for w in range(M):
                    for e in range(N):
                        temp[q,w,e,:,:] = sim[q,w,e,torch.arange(sim.size(3))!=idx_1[q,w,e,idx_2[q,w,e]].item(),torch.arange(sim.size(4))!=idx_2[q,w,e].item()]
            sim = temp
        sim = sim.reshape(b, M, N)
        combination.append(F.relu(sim)*count)
        combination = torch.stack(combination, dim = -1).sum(-1)
            

        classification_scores = torch.transpose(combination, 1,2)
        return classification_scores


def create_model(num_patch, alpha, beta):
    return SOC(
        num_patch = num_patch,
        alpha = alpha,
        beta = beta,
    )
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3" 
# x = focus().cuda()
# features_test = torch.load("../../results/miniImageNet/res12_PN/04_23_02_30_CC_scale0.7/query.pt").cuda()
# features_train = torch.load("../../results/miniImageNet/res12_PN/04_23_02_30_CC_scale0.7/sup.pt").cuda()
# for i in range(100000):
#     with torch.no_grad():
#         classification_scores = x(features_test, features_train, 5, 5)